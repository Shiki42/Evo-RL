from __future__ import annotations

from typing import Any

import torch

from lerobot.policies.rlt.configuration_rlt_ac import ChunkACPolicyConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def _identity_to_transition(batch):
    """Pass the precomputed-chunk-transition dict through unchanged."""
    return batch


def _identity_to_output(transition):
    return transition


def make_rlt_ac_pre_post_processors(
    config: ChunkACPolicyConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Training-time processor pipeline for ChunkACPolicy.

    The dataset (ChunkTransitionDataset) emits precomputed chunk transitions
    that ChunkACPolicy.forward consumes directly. There is nothing to
    normalize, tokenize, or pad — bypass all built-in processor logic by
    returning a no-step pipeline whose to_transition/to_output are identity.

    For deploy-time observation preprocessing, lerobot-record loads the SFT
    pi05 preprocessor directly from the pi0.5 ckpt dir referenced by
    config.vla_pretrained_path. The AC ckpt does not carry a pi05 pipeline.
    """
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
            to_transition=_identity_to_transition,
            to_output=_identity_to_output,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
