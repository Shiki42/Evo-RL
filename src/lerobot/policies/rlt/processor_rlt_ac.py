from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rlt.configuration_rlt_ac import ChunkACPolicyConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def _build_pi05_proxy(config: ChunkACPolicyConfig) -> PI05Config:
    """Construct a PI05Config that mirrors the AC config for processor reuse.

    AC training operates on precomputed chunk transitions (no raw images); the
    preprocessor pipeline is largely a no-op at training time but produces a
    deploy-time pipeline byte-identical to SFT pi0.5. This is the fix for the
    train/deploy action-space mismatch (root cause of "robot moves fast then
    snaps back to origin").
    """
    proxy = PI05Config(
        dtype=config.vla_dtype,
        chunk_size=config.chunk_length,
        n_action_steps=config.chunk_length,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        image_resolution=tuple(config.image_resolution),
        tokenizer_max_length=config.tokenizer_max_length,
        device=config.device,
    )
    proxy.normalization_mapping = dict(config.normalization_mapping)
    proxy.input_features = dict(config.input_features)
    proxy.output_features = dict(config.output_features)
    return proxy


def make_rlt_ac_pre_post_processors(
    config: ChunkACPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    proxy = _build_pi05_proxy(config)
    return make_pi05_pre_post_processors(config=proxy, dataset_stats=dataset_stats)
