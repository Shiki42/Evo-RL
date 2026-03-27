#!/usr/bin/env python3
"""Diagnostic script for ee_delta_ik teleop debugging.

Connects to a leader+follower arm pair and continuously logs:
- Joint positions (degrees) for both arms
- FK end-effector positions for both arms
- Delta from latched reference
- IK roundtrip error
- IK solution details

Usage:
    cd SO101/  # placo needs mesh files in CWD
    python scripts/debug_ee_delta_ik.py \
        --follower_port /dev/ttyACM2 --follower_id bi_so101_follower_right \
        --leader_port /dev/ttyACM0  --leader_id bi_so101_leader_right \
        --urdf ./so101_new_calib.urdf
"""

import argparse
import sys
import time

import numpy as np

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def extract_joints(obs, names=JOINT_NAMES):
    return np.array([float(obs[n + ".pos"]) for n in names])


def main():
    parser = argparse.ArgumentParser(description="Debug ee_delta_ik")
    parser.add_argument("--follower_port", required=True)
    parser.add_argument("--follower_id", required=True)
    parser.add_argument("--leader_port", required=True)
    parser.add_argument("--leader_id", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=5.0, help="Log frequency")
    parser.add_argument("--orientation_weight", type=float, default=0.01)
    parser.add_argument("--position_weight", type=float, default=1.0)
    parser.add_argument("--disable_collisions", action="store_true",
                        help="Try to disable placo collision checking")
    args = parser.parse_args()

    from lerobot.model.kinematics import RobotKinematics
    from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot.robots.so_follower.so_follower import SO101Follower
    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    # --- Connect hardware ---
    follower_cfg = SO101FollowerConfig(port=args.follower_port, id=args.follower_id, cameras={})
    follower = SO101Follower(follower_cfg)
    follower.connect(calibrate=False)
    print("Follower connected: %s" % args.follower_port)

    leader_cfg = SO101LeaderConfig(port=args.leader_port, id=args.leader_id)
    leader = SO101Leader(leader_cfg)
    leader.connect(calibrate=False)
    print("Leader connected: %s" % args.leader_port)

    # --- Init kinematics ---
    leader_kin = RobotKinematics(args.urdf, target_frame_name="gripper_frame_link", joint_names=JOINT_NAMES)
    follower_kin = RobotKinematics(args.urdf, target_frame_name="gripper_frame_link", joint_names=JOINT_NAMES)

    print("Kinematics initialized.")
    print("  orientation_weight = %.4f" % args.orientation_weight)
    print("  position_weight = %.4f" % args.position_weight)

    if args.disable_collisions:
        for kin in [leader_kin, follower_kin]:
            try:
                kin.solver.enable_self_collisions(False)
                print("  Self-collisions DISABLED for %s" % kin)
            except AttributeError:
                print("  WARNING: placo solver does not support enable_self_collisions()")

    # --- Latch references ---
    leader_obs = leader.get_action()
    follower_obs = follower.get_observation()
    leader_q = extract_joints(leader_obs)
    follower_q = extract_joints(follower_obs)

    leader_ref = leader_kin.forward_kinematics(leader_q)
    follower_ref = follower_kin.forward_kinematics(follower_q)

    print()
    print("=" * 90)
    print("References latched:")
    print("  Leader  q (deg): %s" % np.round(leader_q, 2))
    print("  Leader  EE (m):  %s" % np.round(leader_ref[:3, 3], 4))
    print("  Follower q (deg): %s" % np.round(follower_q, 2))
    print("  Follower EE (m):  %s" % np.round(follower_ref[:3, 3], 4))
    print("=" * 90)
    print()

    # --- FK/IK roundtrip at reference ---
    print("--- FK/IK Roundtrip at Reference Pose ---")
    for name, kin, q in [("Leader", leader_kin, leader_q), ("Follower", follower_kin, follower_q)]:
        t_fk = kin.forward_kinematics(q)
        q_ik = kin.inverse_kinematics(q, t_fk, args.position_weight, args.orientation_weight)
        t_rt = kin.forward_kinematics(q_ik)
        pos_err = float(np.linalg.norm(t_rt[:3, 3] - t_fk[:3, 3]))
        joint_err = np.abs(q_ik[:len(JOINT_NAMES)] - q)
        print("  %s: pos_err=%.6fm  max_joint_err=%.4fdeg  q_ik=%s" % (
            name, pos_err, float(np.max(joint_err)), np.round(q_ik[:len(JOINT_NAMES)], 2)))
    print()

    # --- Sweep orientation weights ---
    print("--- IK Accuracy vs Orientation Weight (at follower reference) ---")
    t_target = follower_kin.forward_kinematics(follower_q)
    for ow in [0.0, 0.001, 0.01, 0.1, 1.0]:
        q_ik = follower_kin.inverse_kinematics(follower_q, t_target, args.position_weight, ow)
        t_rt = follower_kin.forward_kinematics(q_ik)
        pos_err = float(np.linalg.norm(t_rt[:3, 3] - t_target[:3, 3]))
        rot_err = float(np.linalg.norm(t_rt[:3, :3] - t_target[:3, :3], "fro"))
        print("  orient_w=%.3f: pos_err=%.6fm  rot_err=%.6f  q_ik=%s" % (
            ow, pos_err, rot_err, np.round(q_ik[:len(JOINT_NAMES)], 2)))
    print()

    # --- Real-time monitoring ---
    print("--- Real-time Monitoring (%.1f Hz for %.0fs) ---" % (args.hz, args.duration))
    print("  Columns: t | leader_delta_pos(mm) | target_ee(mm) | ik_pos_err(mm) | ik_rot_err | follower_q_now")
    print("-" * 120)

    dt = 1.0 / args.hz
    start_t = time.time()
    last_safe_q = follower_q.copy()

    while time.time() - start_t < args.duration:
        tick_start = time.time()

        # Read current state
        leader_action = leader.get_action()
        follower_obs_now = follower.get_observation()
        leader_q_now = extract_joints(leader_action)
        follower_q_now = extract_joints(follower_obs_now)

        # FK leader
        leader_now = leader_kin.forward_kinematics(leader_q_now)
        delta_pos = leader_now[:3, 3] - leader_ref[:3, 3]
        delta_rot = leader_now[:3, :3] @ leader_ref[:3, :3].T

        # Compute target
        target = np.eye(4)
        target[:3, 3] = follower_ref[:3, 3] + delta_pos
        target[:3, :3] = delta_rot @ follower_ref[:3, :3]

        # IK solve
        q_ik = follower_kin.inverse_kinematics(
            follower_q_now, target, args.position_weight, args.orientation_weight)

        # FK roundtrip
        t_rt = follower_kin.forward_kinematics(q_ik)
        pos_err = float(np.linalg.norm(t_rt[:3, 3] - target[:3, 3]))
        rot_err = float(np.linalg.norm(t_rt[:3, :3] - target[:3, :3], "fro"))

        elapsed = time.time() - start_t
        delta_mm = delta_pos * 1000
        target_mm = target[:3, 3] * 1000
        status = "OK" if pos_err < 0.02 else "WARN"

        print("[%s] t=%.1fs | delta=(%+6.1f,%+6.1f,%+6.1f)mm | target=(%+7.1f,%+7.1f,%+7.1f)mm | ik_pos_err=%.1fmm rot_err=%.4f | fol_q=%s" % (
            status, elapsed,
            delta_mm[0], delta_mm[1], delta_mm[2],
            target_mm[0], target_mm[1], target_mm[2],
            pos_err * 1000, rot_err,
            np.round(follower_q_now, 1)))

        # Sleep to maintain hz
        sleep_t = dt - (time.time() - tick_start)
        if sleep_t > 0:
            time.sleep(sleep_t)

    print()
    print("Done. Disconnecting...")
    leader.disconnect()
    follower.disconnect()


if __name__ == "__main__":
    main()
