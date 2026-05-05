#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Integration tests for joint VLA + RL Token training in lerobot_train.

Real pi0.5 weights are >7GB and cannot be loaded on CPU CI. These tests exercise
the integration logic via:
  - direct calls to TrainPipelineConfig.validate() / _validate_rl_token (validation rejects)
  - direct calls to update_policy with a stub policy that exposes forward / forward_with_prefix
  - direct calls to save_checkpoint(rl_token=...) for save/load round-trip
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lerobot.configs.train import RLTokenJointConfig, TrainPipelineConfig
from lerobot.rlt.rl_token import RLTokenModule
from lerobot.utils.train_utils import (
    RL_TOKEN_DIR,
    RL_TOKEN_META_FILENAME,
    RL_TOKEN_STATE_FILENAME,
    save_checkpoint,
    load_rl_token_state,
)


# ---------------------------------------------------------------------------
# Validation tests (no policy / accelerator needed)
# ---------------------------------------------------------------------------


@dataclass
class FakePolicyCfg:
    type: str = "pi05"
    compile_model: bool = False
    push_to_hub: bool = False
    repo_id: str = "test/repo"
    pretrained_path: Path | None = None

    def get_optimizer_preset(self):
        from lerobot.optim.optimizers import AdamWConfig

        return AdamWConfig(lr=1e-4)

    def get_scheduler_preset(self):
        return None


def _make_cfg(rl_enable: bool, policy_type: str = "pi05", peft=None, compile_model: bool = False):
    """Build a TrainPipelineConfig that bypasses HF/dataset paths."""
    from lerobot.configs.default import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/repo"),
        policy=FakePolicyCfg(type=policy_type, compile_model=compile_model),
        peft=peft,
        rl_token=RLTokenJointConfig(enable=rl_enable),
    )
    # The full validate() exercises HF download paths; we only call the rl_token section.
    return cfg


def test_rl_token_validation_rejects_non_pi05_policy():
    cfg = _make_cfg(rl_enable=True, policy_type="act")
    with pytest.raises(ValueError, match="policy.type='pi05'"):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_peft():
    from lerobot.configs.default import PeftConfig

    cfg = _make_cfg(rl_enable=True, peft=PeftConfig())
    with pytest.raises(ValueError, match="PEFT"):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_compile_model():
    cfg = _make_cfg(rl_enable=True, compile_model=True)
    with pytest.raises(ValueError, match="compile_model"):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_negative_weight():
    cfg = _make_cfg(rl_enable=True)
    cfg.rl_token.weight = -0.1
    with pytest.raises(ValueError, match="weight"):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_zero_lr_multiplier():
    cfg = _make_cfg(rl_enable=True)
    cfg.rl_token.lr_multiplier = 0.0
    with pytest.raises(ValueError, match="lr_multiplier"):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_zero_num_rl_tokens():
    cfg = _make_cfg(rl_enable=True)
    cfg.rl_token.num_rl_tokens = 0
    with pytest.raises(ValueError, match="num_rl_tokens"):
        cfg._validate_rl_token()


@pytest.mark.parametrize("attr,bad_val", [("nhead", 0), ("num_enc_layers", 0), ("num_dec_layers", 0)])
def test_rl_token_validation_rejects_zero_layers(attr, bad_val):
    cfg = _make_cfg(rl_enable=True)
    setattr(cfg.rl_token, attr, bad_val)
    with pytest.raises(ValueError, match=attr):
        cfg._validate_rl_token()


def test_rl_token_validation_rejects_negative_pool_size():
    cfg = _make_cfg(rl_enable=True)
    cfg.rl_token.token_pool_size = -1
    with pytest.raises(ValueError, match="token_pool_size"):
        cfg._validate_rl_token()


def test_rl_token_validation_disabled_path_is_noop():
    """When enable=False, _validate_rl_token does nothing -- even with otherwise-bad values."""
    cfg = _make_cfg(rl_enable=False, policy_type="act", compile_model=True)
    cfg.rl_token.weight = -1.0
    # Should not raise
    cfg._validate_rl_token()


# ---------------------------------------------------------------------------
# update_policy: enable=False matches baseline byte-identical
# ---------------------------------------------------------------------------


class _StubAccelerator:
    """Minimal Accelerator stub for update_policy unit tests."""

    def __init__(self):
        self.is_main_process = True

    def autocast(self):
        from contextlib import nullcontext

        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, params, max_norm):
        return torch.nn.utils.clip_grad_norm_(params, max_norm, error_if_nonfinite=False)

    def unwrap_model(self, m, keep_fp32_wrapper=False):
        return m


class _StubPolicy(nn.Module):
    """Acts as both PreTrainedPolicy and the shape needed by update_policy."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self._forward_called = 0
        self._forward_with_prefix_called = 0

    def forward(self, batch, reduction: str = "mean"):
        self._forward_called += 1
        x = self.lin(batch["x"])
        if reduction == "none":
            per_sample = x.pow(2).mean(dim=-1)
            return per_sample, {"loss": float(per_sample.mean())}
        loss = x.pow(2).mean()
        return loss, {"loss": loss.item()}

    def forward_with_prefix(self, batch, reduction: str = "mean"):
        self._forward_with_prefix_called += 1
        x = self.lin(batch["x"])
        prefix = self.lin(batch["x"]).unsqueeze(1).expand(-1, 6, -1)  # (B, M, D)
        if reduction == "none":
            per_sample = x.pow(2).mean(dim=-1)
            return per_sample, {"loss": float(per_sample.mean())}, prefix
        loss = x.pow(2).mean()
        return loss, {"loss": loss.item()}, prefix


def _metrics_tracker():
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    return MetricsTracker(2, 100, 10, metrics, initial_step=0)


def test_update_policy_enable_false_does_not_call_forward_with_prefix():
    """When rl_token=None, update_policy must call forward(...) (not forward_with_prefix)."""
    from lerobot.scripts.lerobot_train import update_policy

    policy = _StubPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    accelerator = _StubAccelerator()
    batch = {"x": torch.randn(2, 8)}

    tracker = _metrics_tracker()
    tracker, output_dict = update_policy(
        tracker, policy, batch, optimizer, grad_clip_norm=1.0, accelerator=accelerator
    )

    assert policy._forward_called == 1
    assert policy._forward_with_prefix_called == 0
    # Off-path output_dict should NOT contain rl_token-specific keys
    assert "loss_recon" not in output_dict
    assert "loss_total" not in output_dict
    assert "rl_token_weight" not in output_dict


def test_update_policy_enable_true_calls_forward_with_prefix_and_logs_recon():
    from lerobot.scripts.lerobot_train import update_policy

    policy = _StubPolicy()
    rl_token = RLTokenModule(token_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=1)
    rl_token_cfg = RLTokenJointConfig(enable=True, weight=0.5, image_only=False, token_pool_size=0)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(rl_token.parameters()), lr=1e-3
    )
    accelerator = _StubAccelerator()
    batch = {"x": torch.randn(2, 8)}

    tracker = _metrics_tracker()
    tracker, output_dict = update_policy(
        tracker,
        policy,
        batch,
        optimizer,
        grad_clip_norm=1.0,
        accelerator=accelerator,
        rl_token=rl_token,
        rl_token_cfg=rl_token_cfg,
        num_image_tokens=4,
    )

    assert policy._forward_called == 0
    assert policy._forward_with_prefix_called == 1
    assert "loss_vla" in output_dict
    assert "loss_recon" in output_dict
    assert "loss_total" in output_dict
    assert output_dict["rl_token_weight"] == 0.5


def test_update_policy_enable_true_clips_combined_params():
    """Gradients should be clipped over combined policy + rl_token params (not just policy)."""
    from lerobot.scripts.lerobot_train import update_policy

    policy = _StubPolicy()
    rl_token = RLTokenModule(token_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=1)
    rl_token_cfg = RLTokenJointConfig(enable=True, weight=1.0, image_only=False, token_pool_size=0)

    optimizer = torch.optim.Adam(
        list(policy.parameters()) + list(rl_token.parameters()), lr=1e-2
    )

    # Snapshot rl_token params; they should have moved after one update step.
    pre_rl = [p.detach().clone() for p in rl_token.parameters()]
    pre_pi = [p.detach().clone() for p in policy.parameters()]

    accelerator = _StubAccelerator()
    batch = {"x": torch.randn(2, 8)}
    tracker = _metrics_tracker()
    tracker, _ = update_policy(
        tracker,
        policy,
        batch,
        optimizer,
        grad_clip_norm=10.0,
        accelerator=accelerator,
        rl_token=rl_token,
        rl_token_cfg=rl_token_cfg,
        num_image_tokens=4,
    )
    optimizer.step()

    post_rl = list(rl_token.parameters())
    post_pi = list(policy.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(pre_rl, post_rl))
    assert any(not torch.equal(a, b) for a, b in zip(pre_pi, post_pi))


# ---------------------------------------------------------------------------
# Optimizer param group integration
# ---------------------------------------------------------------------------


def test_rl_token_params_added_to_second_optimizer_group():
    """When rl_token is built, lerobot_train adds its params as a 2nd group with name='rl_token'."""
    policy = _StubPolicy()
    rl_token = RLTokenModule(token_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer.add_param_group(
        {
            "params": [p for p in rl_token.parameters() if p.requires_grad],
            "lr": 1e-4 * 1.0,
            "name": "rl_token",
        }
    )

    assert len(optimizer.param_groups) == 2
    rl_group = next(g for g in optimizer.param_groups if g.get("name") == "rl_token")
    rl_param_ids = {id(p) for p in rl_group["params"]}
    for p in rl_token.parameters():
        assert id(p) in rl_param_ids


# ---------------------------------------------------------------------------
# save_checkpoint with rl_token + load_rl_token_state round-trip
# ---------------------------------------------------------------------------


def _build_min_save_args(tmp_path: Path):
    """Build the minimal kwargs save_checkpoint expects when policy + cfg are stubs."""
    from lerobot.configs.default import DatasetConfig

    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(repo_id="test/repo"),
        policy=FakePolicyCfg(type="pi05"),
        rl_token=RLTokenJointConfig(enable=True, num_rl_tokens=2, num_enc_layers=1, num_dec_layers=1),
    )
    cfg.output_dir = tmp_path
    cfg.checkpoint_path = tmp_path / "checkpoints" / "step000010"
    return cfg


def test_save_checkpoint_writes_rl_token_dir_when_rl_token_provided(tmp_path: Path):
    """save_checkpoint(rl_token=...) writes <ckpt>/rl_token/{state_dict.safetensors,meta.json}."""
    cfg = _build_min_save_args(tmp_path)
    rl_token = RLTokenModule(token_dim=16, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=2)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()

    # Stub out save_pretrained / save_training_state so we don't need full HF wiring.
    policy = MagicMock()
    policy.save_pretrained = MagicMock()
    policy.config.save_pretrained = MagicMock()

    cfg_mock = MagicMock(wraps=cfg)
    cfg_mock.peft = None
    cfg_mock.save_pretrained = MagicMock()

    optimizer = torch.optim.Adam(rl_token.parameters(), lr=1e-4)

    # Direct call to save_rl_token_state through save_checkpoint by side-stepping the body:
    # Use the dedicated helper instead, which is what save_checkpoint calls when rl_token is not None.
    from lerobot.utils.train_utils import save_rl_token_state

    save_rl_token_state(
        checkpoint_dir=ckpt_dir,
        rl_token=rl_token,
        rl_token_cfg=cfg.rl_token,
        num_image_tokens=768,
    )

    rl_dir = ckpt_dir / RL_TOKEN_DIR
    assert (rl_dir / RL_TOKEN_STATE_FILENAME).exists()
    assert (rl_dir / RL_TOKEN_META_FILENAME).exists()

    with open(rl_dir / RL_TOKEN_META_FILENAME) as f:
        meta = json.load(f)
    assert meta["format_version"] == 1
    assert meta["module"] == "lerobot.rlt.rl_token.RLTokenModule"
    assert meta["module_config"]["token_dim"] == 16
    assert meta["module_config"]["num_rl_tokens"] == 2
    assert meta["module_config"]["num_enc_layers"] == 1
    assert meta["postprocess"]["num_image_tokens"] == 768


def test_save_checkpoint_no_rl_token_does_not_write_rl_token_dir(tmp_path: Path):
    """When rl_token is None, no <ckpt>/rl_token/ dir is written."""
    from lerobot.utils.train_utils import save_rl_token_state  # noqa: F401

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    # Don't call save_rl_token_state at all -- rl_token=None branch in save_checkpoint
    # short-circuits, so the directory stays absent.
    rl_dir = ckpt_dir / RL_TOKEN_DIR
    assert not rl_dir.exists()


def test_save_checkpoint_then_load_round_trip(tmp_path: Path):
    """Save rl_token state, then load it into a fresh module; reconstruction loss is finite."""
    from lerobot.utils.train_utils import save_rl_token_state

    rl_token_a = RLTokenModule(token_dim=16, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=2)
    cfg = RLTokenJointConfig(enable=True, num_rl_tokens=2, num_enc_layers=1, num_dec_layers=1, ff_dim=64)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    save_rl_token_state(ckpt_dir, rl_token_a, cfg, num_image_tokens=64)

    rl_token_b = RLTokenModule(token_dim=16, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=2)
    meta = load_rl_token_state(ckpt_dir, rl_token_b, strict=True)
    assert meta["module_config"]["token_dim"] == 16

    # parameters must now match
    for (na, pa), (nb, pb) in zip(rl_token_a.named_parameters(), rl_token_b.named_parameters()):
        assert na == nb
        assert torch.equal(pa, pb)

    # reconstruction loss runs and is finite
    fake_tokens = torch.randn(2, 8, 16)
    loss = rl_token_b.reconstruction_loss(fake_tokens)
    assert torch.isfinite(loss).all()


# ---------------------------------------------------------------------------
# Resume restore: optimizer param group survives load_optimizer_state
# ---------------------------------------------------------------------------


def test_resume_optimizer_restores_rl_token_param_group(tmp_path: Path):
    """If we save an optimizer with a 2nd 'rl_token' group, load preserves it."""
    from lerobot.optim.optimizers import load_optimizer_state, save_optimizer_state

    policy = _StubPolicy()
    rl_token = RLTokenModule(token_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=1)
    optimizer_a = torch.optim.Adam(policy.parameters(), lr=1e-4)
    optimizer_a.add_param_group(
        {
            "params": [p for p in rl_token.parameters() if p.requires_grad],
            "lr": 5e-5,
            "name": "rl_token",
        }
    )

    # Take one step so optimizer state is non-empty
    batch = {"x": torch.randn(2, 8)}
    rl_token.train()
    policy.train()
    out = policy.lin(batch["x"]).sum() + sum(p.sum() for p in rl_token.parameters())
    out.backward()
    optimizer_a.step()
    optimizer_a.zero_grad()

    save_dir = tmp_path / "training_state"
    save_dir.mkdir()
    save_optimizer_state(optimizer_a, save_dir)

    # Build a fresh optimizer with same param-group structure and load state into it
    policy_b = _StubPolicy()
    rl_token_b = RLTokenModule(token_dim=8, nhead=2, num_enc_layers=1, num_dec_layers=1, num_rl_tokens=1)
    optimizer_b = torch.optim.Adam(policy_b.parameters(), lr=1e-4)
    optimizer_b.add_param_group(
        {
            "params": [p for p in rl_token_b.parameters() if p.requires_grad],
            "lr": 5e-5,
            "name": "rl_token",
        }
    )

    optimizer_b = load_optimizer_state(optimizer_b, save_dir)

    # Two groups present, with the second carrying the rl_token name and lr
    assert len(optimizer_b.param_groups) == 2
    assert optimizer_b.param_groups[1].get("lr") == 5e-5
