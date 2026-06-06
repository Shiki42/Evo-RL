#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.bi_so_follower import BiSOFollower, BiSOFollowerConfig
from lerobot.robots.so_follower import SOFollowerConfig


def _make_arm_mock(name: str) -> MagicMock:
    arm = MagicMock(name=name)
    arm.is_connected = False
    arm.is_calibrated = True
    arm.observation_features = {"joint.pos": float}
    arm.action_features = {"joint.pos": float}
    arm.cameras = {}
    return arm


@pytest.fixture
def bi_follower():
    left_arm = _make_arm_mock("left_arm")
    right_arm = _make_arm_mock("right_arm")

    with patch(
        "lerobot.robots.bi_so_follower.bi_so_follower.SOFollower",
        side_effect=[left_arm, right_arm],
    ):
        robot = BiSOFollower(
            BiSOFollowerConfig(
                left_arm_config=SOFollowerConfig(port="/dev/left"),
                right_arm_config=SOFollowerConfig(port="/dev/right"),
            )
        )
        yield robot, left_arm, right_arm


def test_init_sets_arm_diagnostic_labels(bi_follower):
    _, left_arm, right_arm = bi_follower

    assert left_arm.diagnostic_label == "left follower arm"
    assert right_arm.diagnostic_label == "right follower arm"
