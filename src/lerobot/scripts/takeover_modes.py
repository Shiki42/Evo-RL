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

"""Dual-mode takeover system for corrective teleoperation.

Two modes for mapping leader (teleop) actions to follower (robot) commands
during human intervention:

- **joint_coupled**: direct passthrough (the existing behaviour).
- **ee_delta_ik**: leader motion is interpreted as an end-effector delta
  relative to the moment the operator grabs the leader arm. The delta is
  applied to the follower's EE pose at that instant, and IK produces
  joint-space commands.
"""

import abc
import logging

import numpy as np

from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation


class TakeoverMode(abc.ABC):
    """Strategy interface for mapping leader actions to follower commands."""

    @abc.abstractmethod
    def on_enter(self, leader_action: RobotAction, follower_obs: RobotObservation) -> None:
        """Called once when the operator enters intervention (S0 -> S1)."""

    @abc.abstractmethod
    def compute_action(self, leader_action: RobotAction, follower_obs: RobotObservation) -> RobotAction:
        """Called every tick while the operator is in control (S1)."""

    @abc.abstractmethod
    def on_exit(self) -> None:
        """Called once when the operator releases control (S1 -> S2)."""


class JointCoupledTakeover(TakeoverMode):
    """Transparent passthrough -- leader joint positions are used as-is."""

    def on_enter(self, leader_action: RobotAction, follower_obs: RobotObservation) -> None:
        pass

    def compute_action(self, leader_action: RobotAction, follower_obs: RobotObservation) -> RobotAction:
        return leader_action

    def on_exit(self) -> None:
        pass


def _extract_joint_array(data: RobotAction | RobotObservation, motor_names: list[str]) -> np.ndarray:
    """Extract ordered joint positions (degrees) by explicit key lookup."""
    return np.array([float(data[f"{name}.pos"]) for name in motor_names], dtype=float)


def _clip_position(pos: np.ndarray, bounds_min: np.ndarray | None, bounds_max: np.ndarray | None) -> np.ndarray:
    if bounds_min is not None:
        pos = np.maximum(pos, bounds_min)
    if bounds_max is not None:
        pos = np.minimum(pos, bounds_max)
    return pos


def _limit_step(pos: np.ndarray, ref: np.ndarray, max_step: float) -> np.ndarray:
    delta = pos - ref
    dist = float(np.linalg.norm(delta))
    if dist > max_step and dist > 0:
        pos = ref + delta * (max_step / dist)
    return pos


class EEDeltaIKTakeover(TakeoverMode):
    """Map leader EE delta onto follower EE, then IK to joint space.

    On enter the leader and follower EE poses are latched as references.
    Each tick the translation and rotation deltas of the leader relative to
    its reference are computed and applied to the follower reference to
    produce a target EE pose.  IK converts that target into joint commands.
    """

    def __init__(
        self,
        leader_kinematics: RobotKinematics,
        follower_kinematics: RobotKinematics,
        motor_names: list[str],
        max_ee_step_m: float = 0.05,
        ee_bounds_min: list[float] | None = None,
        ee_bounds_max: list[float] | None = None,
        deadband_m: float = 0.0005,
        ik_pos_tolerance_m: float = 0.02,
        ik_orientation_weight: float = 0.001,
    ):
        self.leader_kin = leader_kinematics
        self.follower_kin = follower_kinematics
        self.motor_names = motor_names
        self.max_ee_step_m = max_ee_step_m
        self.ee_bounds_min = np.array(ee_bounds_min, dtype=float) if ee_bounds_min is not None else None
        self.ee_bounds_max = np.array(ee_bounds_max, dtype=float) if ee_bounds_max is not None else None
        self.deadband_m = deadband_m
        self.ik_pos_tolerance_m = ik_pos_tolerance_m
        self.ik_orientation_weight = ik_orientation_weight

        self._leader_ref: np.ndarray | None = None
        self._follower_ref: np.ndarray | None = None
        self._last_safe_q: np.ndarray | None = None
        self._last_target_pos: np.ndarray | None = None

    def on_enter(self, leader_action: RobotAction, follower_obs: RobotObservation) -> None:
        leader_q = _extract_joint_array(leader_action, self.motor_names)
        follower_q = _extract_joint_array(follower_obs, self.motor_names)

        self._leader_ref = self.leader_kin.forward_kinematics(leader_q)
        self._follower_ref = self.follower_kin.forward_kinematics(follower_q)
        self._last_safe_q = follower_q.copy()
        self._last_target_pos = self._follower_ref[:3, 3].copy()
        logging.info("EEDeltaIKTakeover: references latched.")

    def on_exit(self) -> None:
        self._leader_ref = None
        self._follower_ref = None
        self._last_safe_q = None
        self._last_target_pos = None
        logging.info("EEDeltaIKTakeover: latched state cleared.")

    def compute_action(self, leader_action: RobotAction, follower_obs: RobotObservation) -> RobotAction:
        if self._leader_ref is None or self._follower_ref is None:
            raise RuntimeError("EEDeltaIKTakeover.compute_action called before on_enter.")

        leader_q = _extract_joint_array(leader_action, self.motor_names)
        leader_now = self.leader_kin.forward_kinematics(leader_q)

        target = self._compute_target_ee(leader_now)
        joint_result = self._solve_ik(follower_obs, target)
        return self._build_action_dict(joint_result, leader_action)

    def _compute_target_ee(self, leader_now: np.ndarray) -> np.ndarray:
        delta_pos = leader_now[:3, 3] - self._leader_ref[:3, 3]

        if float(np.linalg.norm(delta_pos)) < self.deadband_m:
            delta_pos = np.zeros(3)

        # Rotation delta in world frame. Assumes leader and follower share the same
        # world-frame orientation (or are close enough). If bases are rotated relative
        # to each other, use local-frame delta: R_ref^T @ R_now applied as R_fol @ delta.
        delta_rot = leader_now[:3, :3] @ self._leader_ref[:3, :3].T

        target = np.eye(4, dtype=float)
        target[:3, 3] = self._follower_ref[:3, 3] + delta_pos
        target[:3, :3] = delta_rot @ self._follower_ref[:3, :3]

        target[:3, 3] = _clip_position(target[:3, 3], self.ee_bounds_min, self.ee_bounds_max)
        target[:3, 3] = _limit_step(target[:3, 3], self._last_target_pos, self.max_ee_step_m)
        self._last_target_pos = target[:3, 3].copy()

        return target

    def _solve_ik(self, follower_obs: RobotObservation, target: np.ndarray) -> np.ndarray:
        # Use last IK solution as initial guess for solver stability (Test 2 pattern).
        # Falls back to live joint positions only on first call after on_enter.
        initial_guess = self._last_safe_q
        q_result = self.follower_kin.inverse_kinematics(
            initial_guess, target, orientation_weight=self.ik_orientation_weight
        )

        # Validate IK solution via FK roundtrip (placo solver always returns a result
        # but may not converge; this check catches divergent solutions)
        fk_check = self.follower_kin.forward_kinematics(q_result)
        pos_err = float(np.linalg.norm(fk_check[:3, 3] - target[:3, 3]))
        if pos_err > self.ik_pos_tolerance_m:
            logging.warning(
                "IK roundtrip error %.4fm exceeds tolerance %.4fm; holding last safe joints.",
                pos_err,
                self.ik_pos_tolerance_m,
            )
            return self._last_safe_q

        self._last_safe_q = q_result.copy()
        return q_result

    def _build_action_dict(self, joint_result: np.ndarray, leader_action: RobotAction) -> RobotAction:
        action: RobotAction = {}
        for i, name in enumerate(self.motor_names):
            action[f"{name}.pos"] = float(joint_result[i])

        # Pass through gripper and any non-motor keys from leader
        for k, v in leader_action.items():
            if k.endswith(".pos") and k.removesuffix(".pos") in self.motor_names:
                continue
            action[k] = v

        return action


def make_takeover_mode(
    mode: str,
    *,
    leader_urdf: str | None = None,
    follower_urdf: str | None = None,
    ee_frame: str = "gripper_frame_link",
    motor_names: list[str] | None = None,
    max_ee_step_m: float = 0.05,
    ee_bounds_min: list[float] | None = None,
    ee_bounds_max: list[float] | None = None,
    deadband_m: float = 0.0005,
    ik_pos_tolerance_m: float = 0.02,
    ik_orientation_weight: float = 0.001,
) -> TakeoverMode:
    """Instantiate a TakeoverMode by name.

    For ``ee_delta_ik``, creates RobotKinematics from URDF paths internally.
    """
    if mode == "joint_coupled":
        return JointCoupledTakeover()

    if mode == "ee_delta_ik":
        if leader_urdf is None or follower_urdf is None:
            raise ValueError("ee_delta_ik mode requires both leader_urdf and follower_urdf.")
        if motor_names is None:
            raise ValueError("ee_delta_ik mode requires motor_names.")

        leader_kin = RobotKinematics(leader_urdf, target_frame_name=ee_frame, joint_names=motor_names)
        follower_kin = RobotKinematics(follower_urdf, target_frame_name=ee_frame, joint_names=motor_names)

        return EEDeltaIKTakeover(
            leader_kinematics=leader_kin,
            follower_kinematics=follower_kin,
            motor_names=motor_names,
            max_ee_step_m=max_ee_step_m,
            ee_bounds_min=ee_bounds_min,
            ee_bounds_max=ee_bounds_max,
            deadband_m=deadband_m,
            ik_pos_tolerance_m=ik_pos_tolerance_m,
            ik_orientation_weight=ik_orientation_weight,
        )

    raise ValueError(f"Unknown takeover mode: {mode!r}. Expected 'joint_coupled' or 'ee_delta_ik'.")
