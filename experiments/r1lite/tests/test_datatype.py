"""Tests for utils/message/datatype.py – RobotAction & Trajectory."""

import pytest


class TestRobotAction:
    def test_default_values(self):
        sensor_msgs = pytest.importorskip("sensor_msgs")  # noqa: F841
        from utils.message.datatype import RobotAction

        action = RobotAction()
        assert action.left_arm is None
        assert action.right_arm is None
        assert action.torso is None
        assert action.left_gripper is None
        assert action.right_gripper is None
        assert action.chassis is None
        assert action.left_ee_pose is None
        assert action.right_ee_pose is None


class TestTrajectory:
    def test_construction(self):
        sensor_msgs = pytest.importorskip("sensor_msgs")  # noqa: F841
        from utils.message.datatype import Trajectory

        traj = Trajectory()
        assert isinstance(traj.timestamp, float)
        assert len(traj.actions) == 0
        assert traj.actions.maxlen == 100
