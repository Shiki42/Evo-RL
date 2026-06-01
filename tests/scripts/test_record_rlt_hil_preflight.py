from __future__ import annotations

from pathlib import Path

from scripts import record_rlt_hil_wo_prefix as record_script


class _FakeDevice:
    instances: list["_FakeDevice"] = []

    def __init__(self, config):
        self.config = config
        self.is_connected = False
        self.calls: list[str] = []
        _FakeDevice.instances.append(self)

    def connect(self, calibrate: bool = True) -> None:
        self.calls.append(f"connect:{calibrate}")
        self.is_connected = True

    def disconnect(self) -> None:
        self.calls.append("disconnect")
        self.is_connected = False


def test_preflight_checks_followers_and_leaders_before_record(monkeypatch, tmp_path):
    _FakeDevice.instances = []
    monkeypatch.setattr(record_script, "BiSOFollower", _FakeDevice)
    monkeypatch.setattr(record_script, "BiSOLeader", _FakeDevice)

    followers = [{"port": "/dev/left-follower"}, {"port": "/dev/right-follower"}]
    leaders = [{"port": "/dev/left-leader"}, {"port": "/dev/right-leader"}]
    robot_cal_dir = tmp_path / "robot-cal"
    leader_cal_dir = tmp_path / "leader-cal"

    record_script._preflight_motor_connections(
        followers,
        leaders,
        str(robot_cal_dir),
        str(leader_cal_dir),
    )

    robot, teleop = _FakeDevice.instances
    assert robot.config.id == "bimanual"
    assert robot.config.calibration_dir == Path(robot_cal_dir)
    assert robot.config.left_arm_config.port == "/dev/left-follower"
    assert robot.config.right_arm_config.port == "/dev/right-follower"
    assert robot.config.left_arm_config.use_degrees is True
    assert robot.config.right_arm_config.use_degrees is True
    assert robot.calls == ["connect:True", "disconnect"]

    assert teleop.config.id == "bimanual_leader"
    assert teleop.config.calibration_dir == Path(leader_cal_dir)
    assert teleop.config.left_arm_config.port == "/dev/left-leader"
    assert teleop.config.right_arm_config.port == "/dev/right-leader"
    assert teleop.config.left_arm_config.use_degrees is True
    assert teleop.config.right_arm_config.use_degrees is True
    assert teleop.calls == ["connect:True", "disconnect"]


def test_preflight_skips_leaders_when_teleop_disabled(monkeypatch, tmp_path):
    _FakeDevice.instances = []
    monkeypatch.setattr(record_script, "BiSOFollower", _FakeDevice)
    monkeypatch.setattr(record_script, "BiSOLeader", _FakeDevice)

    followers = [{"port": "/dev/left-follower"}, {"port": "/dev/right-follower"}]

    record_script._preflight_motor_connections(
        followers,
        [],
        str(tmp_path / "robot-cal"),
        None,
    )

    assert len(_FakeDevice.instances) == 1
    assert _FakeDevice.instances[0].calls == ["connect:True", "disconnect"]
