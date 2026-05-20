from __future__ import annotations

import torch
import torch.nn.functional as F

from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.utils import compute_discount_vector


def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discounted return over a chunk: sum_t gamma^t * r_t."""
    C = reward_seq.shape[1]
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    target_actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_policy_noise: float = 0.2,
    target_noise_clip: float = 0.5,
    q_clamp: float = 100.0,
) -> torch.Tensor:
    """TD3-style chunk-level TD loss.

    AC2: target action clamped same as critic-bootstrap clamp.
    AC3: target action comes from a SEPARATE target_actor (Polyak chain on actor too).
    AC4: target action gets clipped Gaussian smoothing noise (TD3 paper).

    Caller passes the target_actor (RLTAlgorithm.target_actor), not the online actor.
    """
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    with torch.no_grad():
        mu_next, _ = target_actor.forward(x_next, ref_next)
        if target_policy_noise > 0.0:
            noise = torch.randn_like(mu_next) * target_policy_noise
            noise = noise.clamp(-target_noise_clip, target_noise_clip)
            mu_next = mu_next + noise
        mu_next = mu_next.clamp(-1.0, 1.0)
        q_next = target_critic.min_q(x_next, mu_next)
        q_next = q_next.clamp(-q_clamp, q_clamp)
        r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        if actual_steps is not None:
            bootstrap_exp = actual_steps.unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = r + bootstrap

    q1, q2 = critic(x, a)
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(
    actor: ChunkActor,
    critic: TwinCritic,
    batch: dict[str, torch.Tensor],
    beta: float,
) -> torch.Tensor:
    """Q-maximization + BC anchor to VLA reference.

    AC2: mu is clamped to [-1, 1] before being queried against critic.
    """
    x = batch["state_vec"]
    ref = batch["ref_chunk_flat"]
    mu, _ = actor.forward(x, ref, training=True)
    mu_for_q = mu.clamp(-1.0, 1.0)
    q = critic.min_q(x, mu_for_q)
    bc_reg = ((mu - ref) ** 2).sum(dim=-1).mean()
    return -q.mean() + beta * bc_reg
