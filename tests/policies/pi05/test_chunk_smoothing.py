#!/usr/bin/env python
from __future__ import annotations

from types import SimpleNamespace

import torch

from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.constants import OBS_STATE


def _policy(**config_overrides) -> PI05Policy:
    policy = object.__new__(PI05Policy)
    config = SimpleNamespace(
        n_action_steps=2,
        chunk_size=4,
        chunk_overlap_ensemble_prev_weight=0.0,
        chunk_boundary_bridge_steps=0,
        chunk_boundary_bridge_to_state=False,
        chunk_boundary_bridge_anchor="previous_action",
    )
    for name, value in config_overrides.items():
        setattr(config, name, value)
    policy.config = config
    policy._previous_action_chunk = None
    policy._last_selected_action = None
    policy._last_action_debug = {}
    return policy


def test_chunk_overlap_ensemble_blends_previous_tail() -> None:
    policy = _policy(chunk_overlap_ensemble_prev_weight=0.25)
    policy._previous_action_chunk = torch.tensor([[[0.0], [0.0], [8.0], [12.0]]])
    actions = torch.tensor([[[4.0], [4.0], [4.0], [4.0]]])

    blended, prev_overlap, overlap_len = policy._apply_chunk_overlap_ensemble(actions)

    assert overlap_len == 2
    torch.testing.assert_close(prev_overlap, torch.tensor([[[8.0], [12.0]]]))
    torch.testing.assert_close(blended[:, :2], torch.tensor([[[5.0], [6.0]]]))
    torch.testing.assert_close(blended[:, 2:], actions[:, 2:])


def test_chunk_boundary_bridge_uses_last_selected_action() -> None:
    policy = _policy(chunk_boundary_bridge_steps=2)
    policy._last_selected_action = torch.tensor([[0.0]])
    actions = torch.tensor([[[9.0], [9.0], [9.0], [9.0]]])

    bridged = policy._apply_chunk_boundary_bridge({}, actions)

    torch.testing.assert_close(bridged[:, :2], torch.tensor([[[3.0], [6.0]]]))
    torch.testing.assert_close(bridged[:, 2:], actions[:, 2:])


def test_chunk_boundary_bridge_can_anchor_first_chunk_to_state() -> None:
    policy = _policy(chunk_boundary_bridge_steps=2, chunk_boundary_bridge_to_state=True)
    actions = torch.tensor([[[9.0], [9.0], [9.0], [9.0]]])
    batch = {OBS_STATE: torch.tensor([[3.0]])}

    bridged = policy._apply_chunk_boundary_bridge(batch, actions)

    torch.testing.assert_close(bridged[:, :2], torch.tensor([[[5.0], [7.0]]]))
    torch.testing.assert_close(bridged[:, 2:], actions[:, 2:])


class _QueuePolicy(PI05Policy):
    def __init__(self, chunks, config_overrides=None):
        config = {
            "n_action_steps": 2,
            "chunk_size": 4,
            "chunk_overlap_ensemble_prev_weight": 0.0,
            "chunk_boundary_bridge_steps": 0,
            "chunk_boundary_bridge_to_state": False,
            "chunk_boundary_bridge_anchor": "previous_action",
            "output_features": {"action": SimpleNamespace(shape=(1,))},
        }
        if config_overrides:
            config.update(config_overrides)
        self.config = SimpleNamespace(**config)
        self.chunks = list(chunks)
        self.eval_called = False
        PI05Policy.reset(self)

    def _rtc_enabled(self):
        return False

    def eval(self):
        self.eval_called = True

    def predict_action_chunk(self, batch):
        return self.chunks.pop(0).clone()


def test_select_action_refills_queue_and_debugs_boundary() -> None:
    policy = _QueuePolicy(
        [
            torch.tensor([[[1.0], [2.0], [3.0], [4.0]]]),
            torch.tensor([[[10.0], [20.0], [30.0], [40.0]]]),
        ],
        {"chunk_boundary_bridge_steps": 1, "chunk_boundary_bridge_to_state": True},
    )
    batch = {OBS_STATE: torch.tensor([[0.0]])}

    first = PI05Policy.select_action(policy, batch)
    second = PI05Policy.select_action(policy, batch)
    third = PI05Policy.select_action(policy, batch)

    torch.testing.assert_close(first, torch.tensor([[1.0]]))
    torch.testing.assert_close(second, torch.tensor([[2.0]]))
    torch.testing.assert_close(third, torch.tensor([[11.0]]))
    assert policy.eval_called
    assert policy._last_action_debug["is_boundary"]
    torch.testing.assert_close(policy._last_action_debug["raw_action0"], torch.tensor([[10.0]]))
    torch.testing.assert_close(policy._last_action_debug["executed_action0"], torch.tensor([[11.0]]))


def test_bridge_prefers_previous_action_for_later_chunks() -> None:
    policy = _policy(chunk_boundary_bridge_steps=2, chunk_boundary_bridge_to_state=True)
    policy._last_selected_action = torch.tensor([[0.0]])
    actions = torch.tensor([[[9.0], [9.0], [9.0], [9.0]]])
    batch = {OBS_STATE: torch.tensor([[3.0]])}

    bridged = policy._apply_chunk_boundary_bridge(batch, actions)

    torch.testing.assert_close(bridged[:, :2], torch.tensor([[[3.0], [6.0]]]))
    torch.testing.assert_close(bridged[:, 2:], actions[:, 2:])


def test_bridge_can_prefer_state_for_later_chunks() -> None:
    policy = _policy(
        chunk_boundary_bridge_steps=2,
        chunk_boundary_bridge_to_state=True,
        chunk_boundary_bridge_anchor="state",
    )
    policy._last_selected_action = torch.tensor([[0.0]])
    actions = torch.tensor([[[9.0], [9.0], [9.0], [9.0]]])
    batch = {OBS_STATE: torch.tensor([[3.0]])}

    bridged = policy._apply_chunk_boundary_bridge(batch, actions)

    torch.testing.assert_close(bridged[:, :2], torch.tensor([[[5.0], [7.0]]]))
    torch.testing.assert_close(bridged[:, 2:], actions[:, 2:])
