#!/usr/bin/env python
"""CQL (Conservative Q-Learning) baseline.

Algorithm (Kumar et al. 2020), adapted to chunk-level:
1. Q via standard TD3 (twin Q + target + Polyak + smoothing noise) — like our fixed AC
2. Additional conservative penalty:
     L_CQL = α_cql * E[ logsumexp_a Q(s, a) - Q(s, a_exec) ]
   For continuous actions, the logsumexp is approximated by sampling N_ood
   out-of-distribution actions: uniform in [-1, 1] + a few from current policy.
3. Actor loss: TD3-BC (-Q + β * ||π - ref||²)

The CQL penalty pushes Q DOWN at random/policy actions and UP at expert actions.
Helps with offline RL's bootstrapping-from-OOD-actions problem.

Run:
  python scripts/rlt_training/train_cql_baseline.py \
      --cache-dir /home/coder/share/cache/0520_critical_c_rlt_cotrain_fixed \
      --output-dir outputs/cql_baseline_20260521 \
      --gradient-steps 50000
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
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
logger = logging.getLogger("cql")


@dataclass
class CQLExperimentConfig:
    name: str = "cql_baseline"
    actor_hidden: int = 256
    actor_layers: int = 3
    actor_residual: bool = True
    critic_hidden: int = 256
    critic_layers: int = 3
    critic_residual: bool = True
    alpha_cql: float = 1.0
    n_ood_samples: int = 10
    beta_bc: float = 0.3
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    gradient_steps: int = 50_000


def run_cql(exp: CQLExperimentConfig, cache_dir: str, device: str, results_file: Path) -> dict:
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
    target_actor = copy.deepcopy(actor)
    for p in target_actor.parameters():
        p.requires_grad = False

    actor_opt = torch.optim.Adam(actor.parameters(), lr=exp.actor_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=exp.critic_lr)

    train_buf = load_transition_cache(cache_dir, "train", capacity=200_000)
    val_buf = load_transition_cache(cache_dir, "val", capacity=200_000)
    discount_vec = compute_discount_vector(exp.gamma, C, device=device)

    q_losses, a_losses, cql_terms = [], [], []

    start = time.time()
    for step in range(1, exp.gradient_steps + 1):
        batch = {k: v.to(device) for k, v in train_buf.sample(exp.batch_size).items()}
        x = batch["state_vec"]
        a = batch["exec_chunk_flat"]
        x_next = batch["next_state_vec"]
        ref_next = batch["next_ref_flat"]
        reward_seq = batch["reward_seq"]
        done = batch["done"]
        actual_steps = batch["actual_steps"].float()
        B = x.shape[0]

        # 1) TD target via target actor + smoothing noise
        with torch.no_grad():
            mu_next, _ = target_actor.forward(x_next, ref_next)
            noise = torch.randn_like(mu_next) * exp.target_policy_noise
            noise = noise.clamp(-exp.target_noise_clip, exp.target_noise_clip)
            mu_next = (mu_next + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = target_critic(x_next, mu_next)
            q_next = torch.min(q1_t, q2_t).clamp(-100.0, 100.0)
            r = (reward_seq * discount_vec.unsqueeze(0)).sum(dim=1, keepdim=True)
            bootstrap = (exp.gamma ** actual_steps.unsqueeze(-1)) * (1.0 - done.unsqueeze(-1)) * q_next
            y = r + bootstrap

        # 2) Standard TD3 critic loss
        q1, q2 = critic(x, a)
        td_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # 3) CQL penalty: logsumexp over N_ood random + policy actions, minus Q at expert
        N = exp.n_ood_samples
        # Random uniform actions in [-1, 1]
        ood_actions_rand = torch.rand(B * N, chunk_dim, device=device) * 2.0 - 1.0
        # Policy actions (sampled with noise via fixed_std)
        with torch.no_grad():
            mu_pol, _ = actor.forward(x, batch["ref_chunk_flat"], training=False)
            mu_pol_clamped = mu_pol.clamp(-1.0, 1.0)
        # Tile each sample N times → (B*N, chunk_dim) policy actions w/ small noise
        pol_actions = mu_pol_clamped.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        pol_actions = (pol_actions + torch.randn_like(pol_actions) * exp.fixed_std).clamp(-1.0, 1.0)
        # Stack both sets → (B*2N, chunk_dim)
        ood_actions = torch.cat([ood_actions_rand, pol_actions], dim=0)
        x_tiled = x.unsqueeze(1).expand(-1, N, -1).reshape(B * N, -1)
        x_for_ood = torch.cat([x_tiled, x_tiled], dim=0)  # (B*2N, state_dim)

        q1_ood, q2_ood = critic(x_for_ood, ood_actions)
        # Reshape to (2, B, N), logsumexp over N
        q1_ood = q1_ood.view(2, B, N).logsumexp(dim=-1)  # (2, B)
        q2_ood = q2_ood.view(2, B, N).logsumexp(dim=-1)
        # Take avg over the two sets (rand + policy)
        cql_term_q1 = (q1_ood.mean(dim=0) - q1.squeeze(-1)).mean()
        cql_term_q2 = (q2_ood.mean(dim=0) - q2.squeeze(-1)).mean()
        cql_pen = cql_term_q1 + cql_term_q2

        critic_loss = td_loss + exp.alpha_cql * cql_pen
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        critic_opt.step()
        q_losses.append(td_loss.item())
        cql_terms.append(cql_pen.item())

        # Actor: TD3+BC
        mu, _ = actor.forward(x, batch["ref_chunk_flat"], training=True)
        mu_for_q = mu.clamp(-1.0, 1.0)
        q_pol = critic.min_q(x, mu_for_q)
        bc_term = ((mu - batch["ref_chunk_flat"]) ** 2).sum(dim=-1).mean()
        actor_loss = -q_pol.mean() + exp.beta_bc * bc_term
        actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        actor_opt.step()
        a_losses.append(actor_loss.item())

        soft_update(target_critic, critic, exp.tau)
        soft_update(target_actor, actor, exp.tau)

        if step % 10_000 == 0:
            avg_q = sum(q_losses[-5000:]) / max(len(q_losses[-5000:]), 1)
            avg_a = sum(a_losses[-5000:]) / max(len(a_losses[-5000:]), 1)
            avg_cql = sum(cql_terms[-5000:]) / max(len(cql_terms[-5000:]), 1)
            logger.info("[%s] step %d/%d  q=%.4f a=%.4f cql=%.4f",
                        exp.name, step, exp.gradient_steps, avg_q, avg_a, avg_cql)

    elapsed = time.time() - start

    actor.eval(); critic.eval()
    with torch.no_grad():
        rms = ems = qps = qes = 0.0
        n = 0
        for _ in range(20):
            vb = {k: v.to(device) for k, v in val_buf.sample(exp.batch_size).items()}
            x = vb["state_vec"]; a = vb["exec_chunk_flat"]; ref = vb["ref_chunk_flat"]
            mu, _ = actor.forward(x, ref, training=False)
            rms += ((mu - ref) ** 2).mean().item()
            ems += ((mu - a) ** 2).mean().item()
            qps += critic.min_q(x, mu).mean().item()
            qes += critic.min_q(x, a).mean().item()
            n += 1
        ref_mse = rms / n; expert_mse = ems / n
        mean_q_policy = qps / n; mean_q_expert = qes / n
        q_gap = mean_q_policy - mean_q_expert

    final_a = sum(a_losses[-2000:]) / max(len(a_losses[-2000:]), 1)
    final_q = sum(q_losses[-2000:]) / max(len(q_losses[-2000:]), 1)
    final_cql = sum(cql_terms[-2000:]) / max(len(cql_terms[-2000:]), 1)

    result = {
        "algo": "CQL",
        "name": exp.name,
        "config": asdict(exp),
        "final_actor_loss": final_a,
        "final_q_loss": final_q,
        "final_cql_term": final_cql,
        "ref_mse": ref_mse,
        "expert_mse": expert_mse,
        "mean_q_policy": mean_q_policy,
        "mean_q_expert": mean_q_expert,
        "q_gap": q_gap,
        "elapsed_sec": elapsed,
        "steps_per_sec": exp.gradient_steps / elapsed,
    }
    logger.info("[%s] DONE actor=%.4f q=%.4f cql=%.4f ref_mse=%.5f q_gap=%.5f time=%.0fs",
                exp.name, final_a, final_q, final_cql, ref_mse, q_gap, elapsed)

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "target_actor": target_actor.state_dict(), "config": asdict(exp)},
               results_file.parent / f"{exp.name}.pt")
    del actor, critic, target_critic, target_actor, train_buf, val_buf
    torch.cuda.empty_cache()
    return result


def build_cql_grid() -> list[CQLExperimentConfig]:
    return [
        CQLExperimentConfig(name="cql_a1p0_b0p3", alpha_cql=1.0, beta_bc=0.3),
        CQLExperimentConfig(name="cql_a3p0_b0p3", alpha_cql=3.0, beta_bc=0.3),
        CQLExperimentConfig(name="cql_a0p3_b0p3", alpha_cql=0.3, beta_bc=0.3),
        CQLExperimentConfig(name="cql_a1p0_b0p1", alpha_cql=1.0, beta_bc=0.1),
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--output-dir", default="outputs/cql_baseline_20260521")
    p.add_argument("--device", default="cuda")
    p.add_argument("--gradient-steps", type=int, default=50_000)
    p.add_argument("--start-from", type=int, default=0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"

    grid = build_cql_grid()
    for e in grid:
        e.gradient_steps = args.gradient_steps
    logger.info("CQL sweep: %d cells x %d steps", len(grid), args.gradient_steps)
    for i, exp in enumerate(grid):
        if i < args.start_from:
            continue
        logger.info("Cell %d/%d: %s", i + 1, len(grid), exp.name)
        run_cql(exp, args.cache_dir, args.device, results_file)

    if results_file.exists():
        results = sorted(json.loads(results_file.read_text()), key=lambda x: x["ref_mse"])
        logger.info("=" * 70)
        logger.info("CQL SWEEP SUMMARY (by ref_mse)")
        for r in results:
            logger.info("%-30s actor=%.3f q=%.3f cql=%.3f ref_mse=%.5f q_gap=%+.5f elapsed=%.0fs",
                        r["name"], r["final_actor_loss"], r["final_q_loss"], r["final_cql_term"],
                        r["ref_mse"], r["q_gap"], r["elapsed_sec"])


if __name__ == "__main__":
    main()
