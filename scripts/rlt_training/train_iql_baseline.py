#!/usr/bin/env python
"""IQL (Implicit Q-Learning) baseline on the fixed 2026-05-20 cache.

Algorithm (Kostrikov et al. 2021):
1. V(s) = MLP from state_vec → scalar
2. Q(s, a) = same TwinCritic as TD3 (chunk-flat action input)
3. V loss: expectile regression on Q-V residual
     L_V = E[ L_τ(Q(s,a) - V(s)) ]
     L_τ(u) = |τ - 1(u<0)| u²
4. Q loss: TD bootstrap from V(s'), no actor at target time
     y = R_chunk + γ^C (1-d) V(s')
     L_Q = (Q1-y)² + (Q2-y)²
5. Actor loss: advantage-weighted MSE
     adv = min(Q1, Q2) - V(s)
     w = exp(adv / β_temp).clamp_max(w_max)
     L_π = w * ||π(s, ref) - a_exec||²

Run:
  python scripts/rlt_training/train_iql_baseline.py \
      --cache-dir /home/coder/share/cache/0520_critical_c_rlt_cotrain_fixed \
      --output-dir outputs/iql_baseline_20260521 \
      --gradient-steps 50000
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("iql")


@dataclass
class IQLExperimentConfig:
    name: str = "iql_baseline"
    # actor / critic arch matches AC sweep defaults
    actor_hidden: int = 256
    actor_layers: int = 3
    actor_residual: bool = True
    critic_hidden: int = 256
    critic_layers: int = 3
    critic_residual: bool = True
    # IQL-specific
    expectile_tau: float = 0.7
    awr_temp: float = 3.0
    awr_weight_clip: float = 100.0
    # training
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    gradient_steps: int = 50_000


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, num_layers: int = 3, residual: bool = True):
        super().__init__()
        from lerobot.rlt.actor import ResidualMLP
        from lerobot.rlt.utils import build_mlp
        if residual:
            self.net = ResidualMLP(in_dim=state_dim, hidden_dim=hidden_dim, out_dim=1, num_layers=num_layers)
        else:
            self.net = build_mlp(in_dim=state_dim, hidden_dim=hidden_dim, out_dim=1, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def expectile_loss(residual: torch.Tensor, tau: float) -> torch.Tensor:
    """L_τ(u) = |τ - 1(u<0)| u²  (asymmetric squared loss)."""
    weight = torch.where(residual >= 0, tau, 1.0 - tau)
    return (weight * residual.pow(2)).mean()


def run_iql(exp: IQLExperimentConfig, cache_dir: str, device: str, results_file: Path) -> dict:
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.critic import TwinCritic
    from lerobot.rlt.actor import ChunkActor
    from lerobot.rlt.offline_dataset import load_transition_cache
    from lerobot.rlt.utils import compute_discount_vector, soft_update

    cfg = RLTConfig()
    cfg.action_dim = 12
    cfg.proprio_dim = 12
    cfg.chunk_length = 10
    cfg.rl_token.token_dim = 2048
    cfg.rl_token.num_rl_tokens = 4
    state_dim = cfg.rl_token.token_dim + cfg.proprio_dim
    chunk_dim = cfg.chunk_length * cfg.action_dim
    C = cfg.chunk_length

    # Build nets
    actor = ChunkActor(
        state_dim=state_dim, chunk_dim=chunk_dim,
        hidden_dim=exp.actor_hidden, num_layers=exp.actor_layers,
        residual=exp.actor_residual, ref_dropout_p=exp.ref_dropout_p,
        fixed_std=exp.fixed_std,
    ).to(device)
    critic = TwinCritic(
        state_dim=state_dim, chunk_dim=chunk_dim,
        hidden_dim=exp.critic_hidden, num_layers=exp.critic_layers,
        residual=exp.critic_residual,
    ).to(device)
    target_critic = copy.deepcopy(critic)
    for p in target_critic.parameters():
        p.requires_grad = False
    value_net = ValueNet(state_dim=state_dim, hidden_dim=exp.critic_hidden,
                          num_layers=exp.critic_layers, residual=exp.critic_residual).to(device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=exp.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=exp.critic_lr)
    value_opt = torch.optim.Adam(value_net.parameters(), lr=exp.value_lr)

    train_buf = load_transition_cache(cache_dir, "train", capacity=200_000)
    val_buf = load_transition_cache(cache_dir, "val", capacity=200_000)

    discount_vec = compute_discount_vector(exp.gamma, C, device=device)

    q_losses, v_losses, a_losses = [], [], []
    adv_history = []

    start = time.time()
    for step in range(1, exp.gradient_steps + 1):
        batch = {k: v.to(device) for k, v in train_buf.sample(exp.batch_size).items()}
        x = batch["state_vec"]
        a = batch["exec_chunk_flat"]
        x_next = batch["next_state_vec"]
        reward_seq = batch["reward_seq"]
        done = batch["done"]
        actual_steps = batch["actual_steps"].float()

        # 1) V loss (expectile regression on target Q - V)
        with torch.no_grad():
            q1_t, q2_t = target_critic(x, a)
            q_target_for_v = torch.min(q1_t, q2_t).detach()
        v = value_net(x)
        residual = q_target_for_v - v
        v_loss = expectile_loss(residual, exp.expectile_tau)
        value_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 1.0)
        value_opt.step()
        v_losses.append(v_loss.item())

        # 2) Q loss (TD with V(s') bootstrap, no actor at target time)
        with torch.no_grad():
            v_next = value_net(x_next)
            r = (reward_seq * discount_vec.unsqueeze(0)).sum(dim=1, keepdim=True)
            bootstrap = (exp.gamma ** actual_steps.unsqueeze(-1)) * (1.0 - done.unsqueeze(-1)) * v_next
            y = r + bootstrap
        q1, q2 = critic(x, a)
        q_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()
        q_losses.append(q_loss.item())

        # 3) Actor loss (advantage-weighted MSE to exec action)
        with torch.no_grad():
            q1_a, q2_a = target_critic(x, a)
            q_min = torch.min(q1_a, q2_a)
            v_s = value_net(x)
            adv = (q_min - v_s).squeeze(-1)
            w = torch.exp(adv / exp.awr_temp).clamp_max(exp.awr_weight_clip)
            adv_history.append(float(adv.mean().item()))

        mu, _ = actor.forward(x, batch["ref_chunk_flat"], training=True)
        per_sample_mse = ((mu - a) ** 2).sum(dim=-1)
        actor_loss = (w * per_sample_mse).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()
        a_losses.append(actor_loss.item())

        # Polyak target Q
        soft_update(target_critic, critic, exp.tau)

        if step % 10_000 == 0:
            avg_v = sum(v_losses[-5000:]) / max(len(v_losses[-5000:]), 1)
            avg_q = sum(q_losses[-5000:]) / max(len(q_losses[-5000:]), 1)
            avg_a = sum(a_losses[-5000:]) / max(len(a_losses[-5000:]), 1)
            avg_adv = sum(adv_history[-5000:]) / max(len(adv_history[-5000:]), 1)
            logger.info("[%s] step %d/%d  v=%.4f q=%.4f a=%.4f adv=%.4f",
                        exp.name, step, exp.gradient_steps, avg_v, avg_q, avg_a, avg_adv)

    elapsed = time.time() - start

    # Eval: ref_mse + q_gap (policy vs expert)
    actor.eval(); critic.eval(); value_net.eval()
    with torch.no_grad():
        ref_mse_sum = 0.0
        expert_mse_sum = 0.0
        q_policy_sum = 0.0
        q_expert_sum = 0.0
        n = 0
        for _ in range(20):
            vb = {k: v.to(device) for k, v in val_buf.sample(exp.batch_size).items()}
            x = vb["state_vec"]; a = vb["exec_chunk_flat"]; ref = vb["ref_chunk_flat"]
            mu, _ = actor.forward(x, ref, training=False)
            ref_mse_sum += ((mu - ref) ** 2).mean().item()
            expert_mse_sum += ((mu - a) ** 2).mean().item()
            q_pol = critic.min_q(x, mu).mean().item()
            q_exp = critic.min_q(x, a).mean().item()
            q_policy_sum += q_pol
            q_expert_sum += q_exp
            n += 1
        ref_mse = ref_mse_sum / n
        expert_mse = expert_mse_sum / n
        mean_q_policy = q_policy_sum / n
        mean_q_expert = q_expert_sum / n
        q_gap = mean_q_policy - mean_q_expert

    final_a = sum(a_losses[-2000:]) / max(len(a_losses[-2000:]), 1)
    final_q = sum(q_losses[-2000:]) / max(len(q_losses[-2000:]), 1)
    final_v = sum(v_losses[-2000:]) / max(len(v_losses[-2000:]), 1)
    final_adv = sum(adv_history[-2000:]) / max(len(adv_history[-2000:]), 1)

    result = {
        "algo": "IQL",
        "name": exp.name,
        "config": asdict(exp),
        "final_actor_loss": final_a,
        "final_q_loss": final_q,
        "final_v_loss": final_v,
        "final_adv_mean": final_adv,
        "ref_mse": ref_mse,
        "expert_mse": expert_mse,
        "mean_q_policy": mean_q_policy,
        "mean_q_expert": mean_q_expert,
        "q_gap": q_gap,
        "elapsed_sec": elapsed,
        "steps_per_sec": exp.gradient_steps / elapsed,
    }
    logger.info("[%s] DONE actor=%.4f q=%.4f v=%.4f ref_mse=%.5f q_gap=%.5f time=%.0fs",
                exp.name, final_a, final_q, final_v, ref_mse, q_gap, elapsed)

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    out_dir = results_file.parent
    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "value_net": value_net.state_dict(), "target_critic": target_critic.state_dict(),
                "config": asdict(exp)}, out_dir / f"{exp.name}.pt")

    del actor, critic, target_critic, value_net, train_buf, val_buf
    torch.cuda.empty_cache()
    return result


def build_iql_grid() -> list[IQLExperimentConfig]:
    """4-cell mini-sweep over τ_e and β_awr."""
    return [
        IQLExperimentConfig(name="iql_te0p7_bawr3p0", expectile_tau=0.7, awr_temp=3.0),
        IQLExperimentConfig(name="iql_te0p8_bawr3p0", expectile_tau=0.8, awr_temp=3.0),
        IQLExperimentConfig(name="iql_te0p9_bawr3p0", expectile_tau=0.9, awr_temp=3.0),
        IQLExperimentConfig(name="iql_te0p7_bawr10p0", expectile_tau=0.7, awr_temp=10.0),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", default="outputs/iql_baseline_20260521")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gradient-steps", type=int, default=50_000)
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"

    grid = build_iql_grid()
    for e in grid:
        e.gradient_steps = args.gradient_steps
    logger.info("IQL sweep: %d cells x %d steps", len(grid), args.gradient_steps)
    for i, exp in enumerate(grid):
        if i < args.start_from:
            continue
        logger.info("Cell %d/%d: %s", i + 1, len(grid), exp.name)
        run_iql(exp, args.cache_dir, args.device, results_file)

    if results_file.exists():
        results = sorted(json.loads(results_file.read_text()), key=lambda x: x["ref_mse"])
        logger.info("=" * 70)
        logger.info("IQL SWEEP SUMMARY (by ref_mse)")
        for r in results:
            logger.info("%-30s actor=%.4f q=%.4f ref_mse=%.5f q_gap=%+.5f elapsed=%.0fs",
                        r["name"], r["final_actor_loss"], r["final_q_loss"],
                        r["ref_mse"], r["q_gap"], r["elapsed_sec"])


if __name__ == "__main__":
    main()
