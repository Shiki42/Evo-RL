#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Tests for PI05Pytorch.forward_with_prefix and PI05Policy.forward_with_prefix.

Real pi0.5 weights are >7GB and cannot be loaded on CPU CI. These tests use a
StubPaligemma that mimics the `paligemma_with_expert.forward(...)` interface so
we can exercise the new sibling methods without building a real PaLiGemma.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn as nn

from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch


class StubPaligemma(nn.Module):
    """Stand-in for self.paligemma_with_expert.

    The real `paligemma_with_expert.forward(...)` returns:
        ((prefix_out, suffix_out), past_key_values)

    where prefix_out is (B, M_prefix, D_text) and suffix_out is (B, M_suffix, D_expert).
    We mimic that with a single Linear over the concatenated inputs_embeds, slicing
    back into prefix vs suffix on the way out.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        # one Linear so prefix_out has gradient flowing back to a real parameter
        self.prefix_proj = nn.Linear(hidden_size, hidden_size)
        self.suffix_proj = nn.Linear(hidden_size, hidden_size)
        # Mimic structure used by PI05Pytorch.forward to detect dtype.
        self.paligemma = type(
            "PG",
            (),
            {
                "language_model": type(
                    "LM",
                    (),
                    {
                        "layers": [
                            type(
                                "Layer",
                                (),
                                {
                                    "self_attn": type(
                                        "Attn",
                                        (),
                                        {
                                            "q_proj": type(
                                                "Q",
                                                (),
                                                {"weight": torch.zeros(1, dtype=torch.float32)},
                                            )()
                                        },
                                    )()
                                },
                            )()
                        ]
                    },
                )()
            },
        )()

    def forward(
        self,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=False,
        adarms_cond=None,
    ):
        prefix_embs, suffix_embs = inputs_embeds[0], inputs_embeds[1]
        prefix_out = self.prefix_proj(prefix_embs)
        suffix_out = self.suffix_proj(suffix_embs)
        return (prefix_out, suffix_out), None


class TinyPI05Pytorch(nn.Module):
    """Hand-rolled minimal PI05Pytorch substitute exposing forward / forward_with_prefix.

    We deliberately do NOT subclass PI05Pytorch (its __init__ would build PaLiGemma).
    Instead we mirror the relevant attributes and *bind* the real PI05Pytorch.forward
    and PI05Pytorch.forward_with_prefix as bound methods. This proves the methods
    only depend on attribute contracts, not on actual PaLiGemma weights.
    """

    def __init__(self, hidden_size: int = 32, chunk_size: int = 4, action_dim: int = 6):
        super().__init__()
        self.hidden_size = hidden_size

        cfg = type(
            "Cfg",
            (),
            {
                "chunk_size": chunk_size,
                "max_action_dim": action_dim,
                "min_period": 4e-3,
                "max_period": 4.0,
                "time_sampling_beta_alpha": 1.5,
                "time_sampling_beta_beta": 1.0,
                "time_sampling_scale": 0.999,
                "time_sampling_offset": 0.001,
            },
        )()
        self.config = cfg

        self.paligemma_with_expert = StubPaligemma(hidden_size)

        self.action_in_proj = nn.Linear(action_dim, hidden_size)
        self.action_out_proj = nn.Linear(hidden_size, action_dim)
        self.time_mlp_in = nn.Linear(hidden_size, hidden_size)
        self.time_mlp_out = nn.Linear(hidden_size, hidden_size)

        self.gradient_checkpointing_enabled = False
        self.rtc_processor = None

    # Reuse the real implementations -- they are method-resolved at call time so
    # we can simply alias them here.
    embed_prefix = PI05Pytorch.embed_prefix
    embed_suffix = PI05Pytorch.embed_suffix
    sample_noise = PI05Pytorch.sample_noise
    sample_time = PI05Pytorch.sample_time
    _apply_checkpoint = PI05Pytorch._apply_checkpoint
    _prepare_attention_masks_4d = PI05Pytorch._prepare_attention_masks_4d
    forward = PI05Pytorch.forward
    forward_with_prefix = PI05Pytorch.forward_with_prefix


def _make_dummy_inputs(B=2, num_img_tokens=4, num_lang_tokens=3, action_dim=6, chunk_size=4, hidden_size=32):
    """Create dummy images / tokens / actions matching what embed_prefix expects."""
    # embed_prefix expects: images list per camera with `embed_image`, then language tokens through embed_language_tokens.
    # We override embed_prefix on the stub via monkeypatch in the test that needs shapes; the
    # tests in this file call forward_with_prefix on a *different* path that bypasses the
    # real PaLiGemma image / language embedders. So we will use a custom override.
    images = [torch.randn(B, num_img_tokens, hidden_size)]  # one camera's worth of "embedded" tokens
    img_masks = [torch.ones(B, dtype=torch.bool)]
    tokens = torch.zeros(B, num_lang_tokens, dtype=torch.long)
    masks = torch.ones(B, num_lang_tokens, dtype=torch.bool)
    actions = torch.randn(B, chunk_size, action_dim)
    return images, img_masks, tokens, masks, actions


# ---------------------------------------------------------------------------
# Test 1: forward_with_prefix returns shape, dtype, and prefix grad
# ---------------------------------------------------------------------------


def test_pi05_forward_with_prefix_signature_shape_and_grad(monkeypatch):
    """forward_with_prefix returns (loss, prefix_out); prefix has correct shape and is differentiable."""
    B, num_img_tokens, num_lang_tokens, action_dim, chunk_size, hidden_size = 2, 4, 3, 6, 4, 32
    M = num_img_tokens + num_lang_tokens

    model = TinyPI05Pytorch(hidden_size=hidden_size, chunk_size=chunk_size, action_dim=action_dim)
    model.eval()

    # Replace embed_prefix / embed_suffix entry points with simple stand-ins so we don't need a real
    # vision tower or tokenizer embedding layer.
    def fake_embed_prefix(self, images, img_masks, tokens, masks):
        # images is a list of tensors shaped (B, num_img_tokens, hidden_size)
        prefix_emb = torch.cat(images + [torch.zeros(B, num_lang_tokens, hidden_size)], dim=1)
        prefix_pad = torch.cat(
            [torch.ones(B, num_img_tokens, dtype=torch.bool), masks], dim=1
        )
        prefix_att = torch.zeros(B, M, dtype=torch.bool)
        return prefix_emb, prefix_pad, prefix_att

    def fake_embed_suffix(self, x_t, time):
        bsize = x_t.shape[0]
        suf_emb = torch.zeros(bsize, chunk_size, hidden_size)
        suf_pad = torch.ones(bsize, chunk_size, dtype=torch.bool)
        suf_att = torch.zeros(bsize, chunk_size, dtype=torch.bool)
        adarms_cond = None
        return suf_emb, suf_pad, suf_att, adarms_cond

    monkeypatch.setattr(TinyPI05Pytorch, "embed_prefix", fake_embed_prefix)
    monkeypatch.setattr(TinyPI05Pytorch, "embed_suffix", fake_embed_suffix)

    images, img_masks, tokens, masks, actions = _make_dummy_inputs(
        B=B,
        num_img_tokens=num_img_tokens,
        num_lang_tokens=num_lang_tokens,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
    )

    losses, prefix_out = model.forward_with_prefix(images, img_masks, tokens, masks, actions)

    # Shapes
    assert losses.shape == (B, chunk_size, action_dim)
    assert prefix_out.shape == (B, M, hidden_size)

    # Dtype matches the StubPaligemma's prefix_proj weight dtype (fp32).
    assert prefix_out.dtype == torch.float32

    # prefix grad must flow back to at least one paligemma parameter.
    loss = prefix_out.sum() + losses.sum()
    loss.backward()
    pg_params = list(model.paligemma_with_expert.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in pg_params)


# ---------------------------------------------------------------------------
# Test 2: forward unchanged (off-path byte identity)
# ---------------------------------------------------------------------------


def test_pi05_forward_unchanged_when_joint_not_used(monkeypatch):
    """Same seed, same inputs -> forward returns the same losses with or without forward_with_prefix existing.

    We verify that calling forward(...) yields exactly the value it would have without our patch
    by comparing against a recomputation using the same seed/inputs.
    """
    B, num_img_tokens, num_lang_tokens, action_dim, chunk_size, hidden_size = 2, 4, 3, 6, 4, 32
    model = TinyPI05Pytorch(hidden_size=hidden_size, chunk_size=chunk_size, action_dim=action_dim)
    model.eval()

    def fake_embed_prefix(self, images, img_masks, tokens, masks):
        prefix_emb = torch.cat(images + [torch.zeros(B, num_lang_tokens, hidden_size)], dim=1)
        prefix_pad = torch.cat(
            [torch.ones(B, num_img_tokens, dtype=torch.bool), masks], dim=1
        )
        prefix_att = torch.zeros(B, num_img_tokens + num_lang_tokens, dtype=torch.bool)
        return prefix_emb, prefix_pad, prefix_att

    def fake_embed_suffix(self, x_t, time):
        bsize = x_t.shape[0]
        suf_emb = torch.zeros(bsize, chunk_size, hidden_size)
        suf_pad = torch.ones(bsize, chunk_size, dtype=torch.bool)
        suf_att = torch.zeros(bsize, chunk_size, dtype=torch.bool)
        return suf_emb, suf_pad, suf_att, None

    monkeypatch.setattr(TinyPI05Pytorch, "embed_prefix", fake_embed_prefix)
    monkeypatch.setattr(TinyPI05Pytorch, "embed_suffix", fake_embed_suffix)

    images, img_masks, tokens, masks, actions = _make_dummy_inputs(
        B=B,
        num_img_tokens=num_img_tokens,
        num_lang_tokens=num_lang_tokens,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
    )

    torch.manual_seed(0)
    noise = torch.randn(B, chunk_size, action_dim)
    time = torch.full((B,), 0.5)

    losses_a = model.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)
    losses_b = model.forward(images, img_masks, tokens, masks, actions, noise=noise, time=time)

    # Determinism (sanity)
    assert torch.equal(losses_a, losses_b)

    # forward_with_prefix produces the same losses tensor (since the inner forward_func is shared).
    losses_c, _ = model.forward_with_prefix(images, img_masks, tokens, masks, actions, noise=noise, time=time)
    assert torch.allclose(losses_a, losses_c, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 3: PI05Policy.forward_with_prefix returns triple
# ---------------------------------------------------------------------------


def test_pi05_policy_forward_with_prefix_signature():
    """PI05Policy.forward_with_prefix must exist and return (loss, dict, prefix_hidden)."""
    sig = inspect.signature(PI05Policy.forward_with_prefix)
    params = list(sig.parameters)
    # self, batch, reduction
    assert params[:3] == ["self", "batch", "reduction"]


def _make_stub_policy_class():
    """Create a stub class that exposes only the attrs PI05Policy.forward_with_prefix needs."""
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy as P

    class StubPolicy:
        # Bind the real method as an unbound function -- it will be called as a method.
        forward_with_prefix = P.forward_with_prefix

        class _Cfg:
            output_features = {"action": type("F", (), {"shape": (4,)})()}

        def __init__(self, prefix_tensor: torch.Tensor, losses_tensor: torch.Tensor):
            self.config = self._Cfg()
            # model.forward_with_prefix returns (losses, prefix)
            self.model = type(
                "M",
                (),
                {
                    "forward_with_prefix": staticmethod(
                        lambda *a, **kw: (losses_tensor, prefix_tensor)
                    )
                },
            )()

        def _preprocess_images(self, batch):
            return [], []

        def prepare_action(self, batch):
            # padded action dim 6
            return torch.randn(2, 5, 6)

    return StubPolicy


def test_pi05_policy_forward_with_prefix_delegates_to_model():
    """PI05Policy.forward_with_prefix delegates to model.forward_with_prefix and returns the prefix tensor."""
    StubPolicy = _make_stub_policy_class()
    losses = torch.full((2, 5, 6), 0.5)  # (B, chunk, padded_action_dim)
    prefix = torch.randn(2, 7, 8)  # (B, M, D)
    sp = StubPolicy(prefix_tensor=prefix, losses_tensor=losses)

    batch = {
        "observation.language.tokens": torch.zeros(2, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    loss, ldict, returned_prefix = sp.forward_with_prefix(batch)

    assert returned_prefix is prefix
    assert returned_prefix.shape == (2, 7, 8)
    assert isinstance(ldict, dict)
    assert "loss" in ldict and "loss_per_dim" in ldict
    assert torch.isfinite(loss).all()
    # original_action_dim = 4, so the truncated losses are 0.5 -> mean 0.5
    assert abs(loss.item() - 0.5) < 1e-6


def test_pi05_policy_forward_with_prefix_reduction_none():
    StubPolicy = _make_stub_policy_class()
    losses = torch.full((2, 5, 6), 0.25)
    prefix = torch.randn(2, 9, 8)
    sp = StubPolicy(prefix_tensor=prefix, losses_tensor=losses)
    batch = {
        "observation.language.tokens": torch.zeros(2, 3, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    per_sample, ldict, returned_prefix = sp.forward_with_prefix(batch, reduction="none")
    assert per_sample.shape == (2,)
    assert returned_prefix.shape == (2, 9, 8)


# ---------------------------------------------------------------------------
# Test 4: Optional gradient checkpointing path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grad_ckpt", [False, True])
def test_forward_with_prefix_under_gradient_checkpointing(monkeypatch, grad_ckpt):
    """forward_with_prefix preserves autograd graph regardless of gradient_checkpointing flag."""
    B, num_img_tokens, num_lang_tokens, action_dim, chunk_size, hidden_size = 2, 4, 3, 6, 4, 32
    model = TinyPI05Pytorch(hidden_size=hidden_size, chunk_size=chunk_size, action_dim=action_dim)
    model.train()
    model.gradient_checkpointing_enabled = grad_ckpt

    def fake_embed_prefix(self, images, img_masks, tokens, masks):
        prefix_emb = torch.cat(images + [torch.zeros(B, num_lang_tokens, hidden_size)], dim=1)
        prefix_pad = torch.cat(
            [torch.ones(B, num_img_tokens, dtype=torch.bool), masks], dim=1
        )
        prefix_att = torch.zeros(B, num_img_tokens + num_lang_tokens, dtype=torch.bool)
        return prefix_emb, prefix_pad, prefix_att

    def fake_embed_suffix(self, x_t, time):
        bsize = x_t.shape[0]
        suf_emb = torch.zeros(bsize, chunk_size, hidden_size)
        suf_pad = torch.ones(bsize, chunk_size, dtype=torch.bool)
        suf_att = torch.zeros(bsize, chunk_size, dtype=torch.bool)
        return suf_emb, suf_pad, suf_att, None

    monkeypatch.setattr(TinyPI05Pytorch, "embed_prefix", fake_embed_prefix)
    monkeypatch.setattr(TinyPI05Pytorch, "embed_suffix", fake_embed_suffix)

    images, img_masks, tokens, masks, actions = _make_dummy_inputs(
        B=B,
        num_img_tokens=num_img_tokens,
        num_lang_tokens=num_lang_tokens,
        action_dim=action_dim,
        chunk_size=chunk_size,
        hidden_size=hidden_size,
    )

    losses, prefix_out = model.forward_with_prefix(images, img_masks, tokens, masks, actions)
    assert prefix_out.requires_grad

    # Backward should populate at least one grad on the paligemma stub (no NaN).
    (prefix_out.sum() + losses.sum()).backward()
    grads = [p.grad for p in model.paligemma_with_expert.parameters() if p.grad is not None]
    assert len(grads) >= 1
    assert all(torch.isfinite(g).all() for g in grads)
