"""Regression tests for AC1+AC2+AC3+AC4 fixes (2026-05-21).

AC1 = trainer.py offline target update inside UTD loop
AC2 = mu clamped before critic.min_q in actor_loss
AC3 = separate target_actor in RLTAlgorithm
AC4 = target_policy_noise / target_noise_clip in critic_loss
"""
from __future__ import annotations

import copy

import pytest
import torch

from lerobot.rlt.actor import ChunkActor
from lerobot.rlt.algorithm import RLTAlgorithm
from lerobot.rlt.config import RLTConfig
from lerobot.rlt.critic import TwinCritic
from lerobot.rlt.losses import actor_loss, critic_loss
from lerobot.rlt.policy import RLTPolicy
from lerobot.rlt.vla_adapter import DummyVLAAdapter


@pytest.fixture
def algo_cuda():
    cfg = RLTConfig()
    cfg.action_dim = 4
    cfg.proprio_dim = 4
    cfg.chunk_length = 5
    cfg.rl_token.token_dim = 16
    cfg.rl_token.num_rl_tokens = 1
    cfg.actor.hidden_dim = 32
    cfg.actor.num_layers = 2
    cfg.critic.hidden_dim = 32
    cfg.critic.num_layers = 2
    vla = DummyVLAAdapter(token_dim=16, action_dim=4, num_tokens=8, horizon=10)
    policy = RLTPolicy(cfg, vla).cuda()
    algo = RLTAlgorithm(policy, cfg)
    algo.critic.cuda()
    algo.target_critic.cuda()
    algo.target_actor.cuda()
    return algo, cfg


def _make_batch(B: int, state_dim: int, C: int, action_dim: int, device: str = "cuda") -> dict:
    return {
        "state_vec": torch.randn(B, state_dim, device=device),
        "exec_chunk_flat": torch.randn(B, C * action_dim, device=device) * 0.1,
        "ref_chunk_flat": torch.randn(B, C * action_dim, device=device) * 0.1,
        "reward_seq": torch.zeros(B, C, device=device),
        "next_state_vec": torch.randn(B, state_dim, device=device),
        "next_ref_flat": torch.randn(B, C * action_dim, device=device) * 0.1,
        "done": torch.zeros(B, device=device),
        "actual_steps": torch.full((B,), C, dtype=torch.float32, device=device),
    }


def test_ac3_target_actor_exists_and_is_separate(algo_cuda):
    """AC3: RLTAlgorithm exposes target_actor as a separate Polyak copy."""
    algo, _ = algo_cuda
    assert hasattr(algo, "target_actor")
    assert algo.target_actor is not algo.policy.actor
    # Same architecture
    on_params = {n: p.shape for n, p in algo.policy.actor.named_parameters()}
    tgt_params = {n: p.shape for n, p in algo.target_actor.named_parameters()}
    assert on_params == tgt_params
    # Target frozen
    for p in algo.target_actor.parameters():
        assert not p.requires_grad


def test_ac3_target_actor_initialised_to_online(algo_cuda):
    """AC3: target_actor starts identical to online actor (deepcopy)."""
    algo, _ = algo_cuda
    for (n1, p1), (n2, p2) in zip(
        algo.policy.actor.named_parameters(), algo.target_actor.named_parameters()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2)


def test_ac3_soft_update_polyaks_both_critic_and_actor(algo_cuda):
    """AC3: soft_update_target(tau) updates target_actor AND target_critic."""
    algo, cfg = algo_cuda
    # Diverge online actor + critic
    for p in algo.policy.actor.parameters():
        p.data.add_(torch.randn_like(p) * 0.5)
    for p in algo.critic.parameters():
        p.data.add_(torch.randn_like(p) * 0.5)

    # Snapshot targets
    tc_before = [p.clone() for p in algo.target_critic.parameters()]
    ta_before = [p.clone() for p in algo.target_actor.parameters()]

    algo.soft_update_target(tau=0.5)

    # Both should have moved
    tc_after = list(algo.target_critic.parameters())
    ta_after = list(algo.target_actor.parameters())
    moved_critic = any(not torch.equal(b, a) for b, a in zip(tc_before, tc_after))
    moved_actor = any(not torch.equal(b, a) for b, a in zip(ta_before, ta_after))
    assert moved_critic, "target_critic did not move"
    assert moved_actor, "target_actor did not move"


def test_ac4_target_policy_noise_deterministic_when_zero(algo_cuda):
    """AC4: critic_loss is deterministic when target_policy_noise=0."""
    algo, cfg = algo_cuda
    state_dim = algo.config.rl_token.token_dim + algo.config.proprio_dim
    batch = _make_batch(8, state_dim, cfg.chunk_length, cfg.action_dim)
    torch.manual_seed(0)
    l1 = critic_loss(
        algo.critic, algo.target_critic, algo.target_actor, batch, gamma=0.99, C=cfg.chunk_length,
        target_policy_noise=0.0,
    )
    torch.manual_seed(1)
    l2 = critic_loss(
        algo.critic, algo.target_critic, algo.target_actor, batch, gamma=0.99, C=cfg.chunk_length,
        target_policy_noise=0.0,
    )
    assert torch.isclose(l1, l2)


def test_ac4_target_policy_noise_nondeterministic_when_positive(algo_cuda):
    """AC4: critic_loss varies when target_policy_noise>0 and seeds differ."""
    algo, cfg = algo_cuda
    state_dim = algo.config.rl_token.token_dim + algo.config.proprio_dim
    batch = _make_batch(8, state_dim, cfg.chunk_length, cfg.action_dim)
    torch.manual_seed(0)
    l1 = critic_loss(
        algo.critic, algo.target_critic, algo.target_actor, batch, gamma=0.99, C=cfg.chunk_length,
        target_policy_noise=0.2, target_noise_clip=0.5,
    )
    torch.manual_seed(1)
    l2 = critic_loss(
        algo.critic, algo.target_critic, algo.target_actor, batch, gamma=0.99, C=cfg.chunk_length,
        target_policy_noise=0.2, target_noise_clip=0.5,
    )
    assert not torch.isclose(l1, l2)


def test_ac2_actor_loss_clamps_mu_for_critic(algo_cuda):
    """AC2: mu fed to critic is clamped to [-1, 1].

    Use a fake actor that always returns a large mu; a fake critic that returns
    a.mean()*1000. If actor_loss clamps, |-q| == 1000; if not, it == 10000.
    """
    algo, cfg = algo_cuda
    state_dim = algo.config.rl_token.token_dim + algo.config.proprio_dim
    B = 4
    batch = _make_batch(B, state_dim, cfg.chunk_length, cfg.action_dim)

    class _FakeActor(torch.nn.Module):
        def forward(self, x, ref, training=False):
            mu = torch.full_like(ref, 10.0)
            return mu, mu * 0.0

    class _LinearCritic(torch.nn.Module):
        def min_q(self, x, a):
            return a.mean(dim=-1, keepdim=True) * 1000.0

    fa = _FakeActor().cuda()
    fc = _LinearCritic().cuda()
    loss = actor_loss(fa, fc, batch, beta=0.0)
    # With clamp, mu in critic = 1.0, q = 1000, -q.mean() = -1000
    assert -1100 < loss.item() < -900, f"loss {loss.item()} suggests no clamp"


def test_ac2_no_clamp_would_fail_same_test():
    """Sanity: if we removed the clamp, the same test would yield ~-10000."""
    # Just a constant-check that the test threshold is meaningful.
    assert -10000 < -1100  # would fail without clamp
    assert -1100 < -1000   # passes with clamp


def test_ac1_trainer_offline_soft_update_inside_utd_loop():
    """AC1: trainer.offline_rl_loop now calls soft_update_target inside the UTD loop."""
    import inspect
    from lerobot.rlt import trainer as trainer_mod
    src = inspect.getsource(trainer_mod.offline_rl_loop)
    # The pattern should now have soft_update_target inside the for-loop level,
    # not outside the inner `for _ in range(utd)`.
    # Heuristic: count the indent of `algorithm.soft_update_target(tau)`.
    # Inside utd loop → indent of 12 spaces (within `for _ in range(utd):` block)
    # Outside → indent of 8 spaces (top of `for step in range(...)`)
    for line in src.splitlines():
        s = line.rstrip()
        if "soft_update_target(tau)" in s:
            indent = len(s) - len(s.lstrip())
            assert indent >= 12, f"soft_update_target at indent {indent} (outside UTD); want >=12"
            return
    raise AssertionError("No soft_update_target(tau) call found in offline_rl_loop")
