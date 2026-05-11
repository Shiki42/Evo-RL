from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.policies.rlt.configuration_rlt_token import RLTokenPolicyConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def _build_pi05_proxy(config: RLTokenPolicyConfig) -> PI05Config:
    """Construct a PI05Config that mirrors the RLT config for processor reuse.

    Goal: the saved policy_preprocessor.json is byte-identical to an SFT pi05
    run, so training and deployment share one normalization pipeline. The proxy
    is used ONLY by the processor factory; no pi05 weights are loaded here.
    """
    proxy = PI05Config(
        dtype=config.vla_dtype,
        chunk_size=config.chunk_size,
        n_action_steps=config.chunk_size,
        max_state_dim=config.max_state_dim,
        max_action_dim=config.max_action_dim,
        image_resolution=tuple(config.image_resolution),
        tokenizer_max_length=config.tokenizer_max_length,
        device=config.device,
    )
    # Preserve RLT's normalization mapping (which is set to match SFT pi05).
    proxy.normalization_mapping = dict(config.normalization_mapping)
    # Wire input/output features so the normalizer step sees the same keys.
    proxy.input_features = dict(config.input_features)
    proxy.output_features = dict(config.output_features)
    return proxy


def make_rlt_token_pre_post_processors(
    config: RLTokenPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    proxy = _build_pi05_proxy(config)
    return make_pi05_pre_post_processors(config=proxy, dataset_stats=dataset_stats)
