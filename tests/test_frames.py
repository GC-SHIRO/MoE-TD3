#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""坐标换算测试：动作方向、上升符号和 PX4 姿态换算。"""

import math
import unittest

import numpy as np

from core.frames import (
    HOLD_YAW_NED,
    action_to_ned_velocity,
    enu_to_ned,
    ned_to_enu,
    px4_attitude_to_enu_rotation,
    rotation_from_rpy,
    rotation_to_rpy,
    yaw_enu_to_ned,
)


class FrameTests(unittest.TestCase):
    def test_forward_action_flies_world_east(self):
        """动作 (1,0,0) 对应 NED (0,1,0)，即沿 Gazebo 世界 +x 飞。"""
        np.testing.assert_allclose(action_to_ned_velocity([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0])
        np.testing.assert_allclose(action_to_ned_velocity([0.0, 1.0, 0.0]), [1.0, 0.0, 0.0])

    def test_positive_z_action_climbs(self):
        """SEARCH 使用正 z 上升，对应 NED 中负的 v_down。"""
        self.assertLess(action_to_ned_velocity([0.0, 0.0, 0.25])[2], 0.0)

    def test_enu_ned_round_trip(self):
        vector = np.array([1.5, -2.0, 3.25])
        np.testing.assert_allclose(ned_to_enu(enu_to_ned(vector)), vector)
        np.testing.assert_allclose(enu_to_ned(ned_to_enu(vector)), vector)

    def test_hold_yaw_faces_world_x(self):
        self.assertAlmostEqual(yaw_enu_to_ned(0.0), HOLD_YAW_NED)

    def test_px4_attitude_facing_east_is_identity_in_enu(self):
        """PX4 偏航 π/2（朝东）的姿态在 ENU/FLU 下应为单位旋转。"""
        half = HOLD_YAW_NED / 2.0
        q = [math.cos(half), 0.0, 0.0, math.sin(half)]
        np.testing.assert_allclose(px4_attitude_to_enu_rotation(q), np.eye(3), atol=1e-12)

    def test_rpy_round_trip(self):
        rpy = (0.1, -0.2, 0.3)
        np.testing.assert_allclose(rotation_to_rpy(rotation_from_rpy(*rpy)), rpy, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
