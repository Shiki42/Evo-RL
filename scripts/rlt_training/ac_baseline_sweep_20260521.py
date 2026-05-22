#!/usr/bin/env python
"""Baseline AC sweep on the fixed 2026-05-20 cache.

Reproduces docs/rlt/ac_findings.md historical winners on the new
(post B1+B2+B3 bug fix) transition cache, plus a slim widening grid around
the known-best config. 12 cells, 50k gradient steps each.

Run:
  cd /home/coder/code/Evo-RL-quick
  source /home/coder/venv-lerobot/bin/activate
  python scripts/rlt_training/ac_baseline_sweep_20260521.py \
      --cache-dir /home/coder/share/cache/0520_critical_c_rlt_cotrain_fixed \
      --output-dir outputs/ac_baseline_20260521 \
      --gradient-steps 50000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

# Make src importable
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ac_baseline")


@dataclass
class ExperimentConfig:
    name: str
    # actor
    actor_hidden: int = 256
    actor_layers: int = 3
    actor_residual: bool = True
    actor_layer_norm: bool = False
    actor_lr: float = 3e-4
    ref_dropout_p: float = 0.5
    fixed_std: float = 0.05
    # critic
    critic_hidden: int = 256
    critic_layers: int = 3
    critic_residual: bool = True
    critic_layer_norm: bool = False
    critic_lr: float = 3e-4
    # training
    beta: float = 0.3
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    actor_update_interval: int = 2
    gradient_steps: int = 50_000


def build_baseline_grid() -> list[ExperimentConfig]:
    """12 cells: known-best + 1D sweeps around it on the new fixed cache."""
    exps: list[ExperimentConfig] = []

    # 1. Known-best from ac_findings.md (residual MLP, beta=0.3)
    exps.append(ExperimentConfig(name="winner_b0p3_res3L256h_rd0p5"))

    # 2-3. Beta sweep around 0.3
    exps.append(ExperimentConfig(name="b0p1", beta=0.1))
    exps.append(ExperimentConfig(name="b1p0", beta=1.0))

    # 4. Residual ablation
    exps.append(ExperimentConfig(
        name="plain_mlp_b0p3", actor_residual=False, critic_residual=False
    ))

    # 5-6. Depth sweep
    exps.append(ExperimentConfig(
        name="L2_b0p3", actor_layers=2, critic_layers=2
    ))
    exps.append(ExperimentConfig(
        name="L4_b0p3", actor_layers=4, critic_layers=4
    ))

    # 7. Width sweep
    exps.append(ExperimentConfig(
        name="H512_b0p3", actor_hidden=512, critic_hidden=512
    ))

    # 8-9. Ref dropout sweep
    exps.append(ExperimentConfig(name="rd0p3_b0p3", ref_dropout_p=0.3))
    exps.append(ExperimentConfig(name="rd0p7_b0p3", ref_dropout_p=0.7))

    # 10. Fixed std sweep
    exps.append(ExperimentConfig(name="std0p02_b0p3", fixed_std=0.02))

    # 11-12. Tau sweep
    exps.append(ExperimentConfig(name="tau0p001_b0p3", tau=0.001))
    exps.append(ExperimentConfig(name="tau0p01_b0p3", tau=0.01))

    return exps


def run_experiment(exp: ExperimentConfig, cache_dir: str, device: str, results_file: Path) -> dict:
    from lerobot.rlt.algorithm import RLTAlgorithm
    from lerobot.rlt.config import RLTConfig
    from lerobot.rlt.evaluator import evaluate_offline
    from lerobot.rlt.losses import actor_loss, critic_loss
    from lerobot.rlt.offline_dataset import load_transition_cache
    from lerobot.rlt.policy import RLTPolicy
    from lerobot.rlt.utils import soft_update
    from lerobot.rlt.vla_adapter import DummyVLAAdapter

    logger.info("=" * 60)
    logger.info("Running: %s (%d steps)", exp.name, exp.gradient_steps)

    config = RLTConfig()
    config.action_dim = 12
    config.proprio_dim = 12
    config.vla_horizon = 50
    config.chunk_length = 10
    config.rl_token.token_dim = 2048
    config.rl_token.num_rl_tokens = 4
    # match phase3 / cotrained model
    config.rl_token.enc_layers = 3
    config.rl_token.dec_layers = 3
    config.rl_token.ff_dim = 4096

    config.actor.hidden_dim = exp.actor_hidden
    config.actor.num_layers = exp.actor_layers
    config.actor.residual = exp.actor_residual
    config.actor.layer_norm = exp.actor_layer_norm
    config.actor.lr = exp.actor_lr
    config.actor.ref_dropout_p = exp.ref_dropout_p
    config.actor.fixed_std = exp.fixed_std

    config.critic.hidden_dim = exp.critic_hidden
    config.critic.num_layers = exp.critic_layers
    config.critic.residual = exp.critic_residual
    config.critic.layer_norm = exp.critic_layer_norm
    config.critic.lr = exp.critic_lr

    config.training.beta = exp.beta
    config.training.gamma = exp.gamma
    config.training.tau = exp.tau
    config.training.batch_size = exp.batch_size
    config.training.actor_update_interval = exp.actor_update_interval

    vla = DummyVLAAdapter(token_dim=2048, action_dim=12, num_tokens=64, horizon=50)
    policy = RLTPolicy(config, vla).to(device)
    policy.freeze_vla()
    policy.freeze_rl_token_encoder()

    algorithm = RLTAlgorithm(policy, config)
    algorithm.to(device) if hasattr(algorithm, "to") else None
    algorithm.critic.to(device)
    algorithm.target_critic.to(device)

    train_buffer = load_transition_cache(cache_dir, "train", capacity=config.replay.capacity)
    val_buffer = load_transition_cache(cache_dir, "val", capacity=config.replay.capacity)

    actor_opt = torch.optim.Adam(algorithm.policy.actor.parameters(), lr=config.actor.lr)
    critic_opt = torch.optim.Adam(algorithm.critic.parameters(), lr=config.critic.lr)

    C = config.chunk_length
    gamma, beta, tau = config.training.gamma, config.training.beta, config.training.tau
    actor_interval = config.training.actor_update_interval
    critic_update_count = 0
    actor_losses, critic_losses = [], []

    start = time.time()
    for step in range(1, exp.gradient_steps + 1):
        batch = {k: v.to(device) for k, v in train_buffer.sample(exp.batch_size).items()}

        c_loss = critic_loss(algorithm.critic, algorithm.target_critic, algorithm.target_actor, batch, gamma, C, target_policy_noise=config.training.target_policy_noise, target_noise_clip=config.training.target_noise_clip)
        critic_opt.zero_grad()
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(algorithm.critic.parameters(), 1.0)
        critic_opt.step()
        critic_losses.append(c_loss.item())
        critic_update_count += 1

        if critic_update_count % actor_interval == 0:
            a_loss = actor_loss(algorithm.policy.actor, algorithm.critic, batch, beta)
            actor_opt.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(algorithm.policy.actor.parameters(), 1.0)
            actor_opt.step()
            actor_losses.append(a_loss.item())

        algorithm.soft_update_target(tau)

        if step % 10_000 == 0:
            avg_a = sum(actor_losses[-5000:]) / max(len(actor_losses[-5000:]), 1)
            avg_c = sum(critic_losses[-5000:]) / max(len(critic_losses[-5000:]), 1)
            logger.info("[%s] Step %d/%d  c=%.4f  a=%.4f", exp.name, step, exp.gradient_steps, avg_c, avg_a)

    elapsed = time.time() - start

    algorithm.critic.eval()
    algorithm.target_critic.eval()
    algorithm.policy.actor.eval()
    eval_metrics = evaluate_offline(algorithm, val_buffer, config, num_batches=20)

    final_a = sum(actor_losses[-2000:]) / max(len(actor_losses[-2000:]), 1)
    final_c = sum(critic_losses[-2000:]) / max(len(critic_losses[-2000:]), 1)

    result = {
        "name": exp.name,
        "config": asdict(exp),
        "final_actor_loss": final_a,
        "final_critic_loss": final_c,
        "ref_mse": float(eval_metrics.ref_action_mse),
        "expert_mse": float(eval_metrics.expert_action_mse),
        "q_gap": float(eval_metrics.q_gap),
        "mean_q_policy": float(eval_metrics.mean_q_policy),
        "mean_q_expert": float(eval_metrics.mean_q_expert),
        "td_error": float(eval_metrics.mean_critic_td_error),
        "elapsed_sec": elapsed,
        "steps_per_sec": exp.gradient_steps / elapsed,
    }
    logger.info("[%s] DONE actor=%.4f ref_mse=%.5f q_gap=%.5f time=%.0fs",
                exp.name, final_a, result["ref_mse"], result["q_gap"], elapsed)

    results = json.loads(results_file.read_text()) if results_file.exists() else []
    results.append(result)
    results_file.write_text(json.dumps(results, indent=2))

    # Save ckpt for the best so far (by ref_mse)
    out_dir = results_file.parent
    best_file = out_dir / "best.json"
    best_ref = float("inf")
    if best_file.exists():
        best_ref = json.loads(best_file.read_text())["ref_mse"]
    if result["ref_mse"] < best_ref:
        best_file.write_text(json.dumps(result, indent=2))
        torch.save({
            "actor": algorithm.policy.actor.state_dict(),
            "critic": algorithm.critic.state_dict(),
            "target_critic": algorithm.target_critic.state_dict(),
            "config": asdict(exp),
        }, out_dir / "best_checkpoint.pt")
        logger.info("[%s] NEW BEST ref_mse=%.5f", exp.name, result["ref_mse"])

    # Free GPU
    del algorithm, policy, train_buffer, val_buffer, actor_opt, critic_opt
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/ac_baseline_20260521")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gradient-steps", type=int, default=50_000)
    parser.add_argument("--start-from", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_file = out / "results.json"

    exps = build_baseline_grid()
    for e in exps:
        e.gradient_steps = args.gradient_steps

    logger.info("AC baseline sweep: %d cells x %d steps each", len(exps), args.gradient_steps)
    logger.info("Cache: %s", args.cache_dir)
    logger.info("Output: %s", out)

    for i, exp in enumerate(exps):
        if i < args.start_from:
            continue
        logger.info("Cell %d/%d: %s", i + 1, len(exps), exp.name)
        run_experiment(exp, args.cache_dir, args.device, results_file)

    # Final summary
    if results_file.exists():
        results = sorted(json.loads(results_file.read_text()), key=lambda x: x["ref_mse"])
        logger.info("=" * 70)
        logger.info("BASELINE SWEEP SUMMARY (sorted by val ref_mse)")
        for r in results:
            logger.info("%-30s actor=%.4f ref_mse=%.5f q_gap=%.5f elapsed=%.0fs",
                        r["name"], r["final_actor_loss"], r["ref_mse"], r["q_gap"], r["elapsed_sec"])


if __name__ == "__main__":
    main()
