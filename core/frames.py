#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS/Gazebo（ENU/FLU）与 PX4（NED/FRD）之间的坐标换算。

所有坐标换算只放在这里。策略动作的语义：偏航固定、机头朝世界 +x 时的机体系速度，
因此动作直接等于世界 ENU 速度 (vx, vy, vz)，z 向上为正。
"""

import math
from typing import Sequence, Tuple

import numpy as np


# 机头朝 Gazebo 世界 +x（东）时，PX4 NED 中的偏航角。
HOLD_YAW_NED = math.pi / 2.0


def enu_to_ned(vector: Sequence[float]) -> np.ndarray:
    """ENU (东, 北, 天) → NED (北, 东, 地)。位置和速度通用。"""
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([y, x, -z], dtype=np.float64)


def ned_to_enu(vector: Sequence[float]) -> np.ndarray:
    """NED (北, 东, 地) → ENU (东, 北, 天)。该换算是自身的逆。"""
    n, e, d = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.array([e, n, -d], dtype=np.float64)


def action_to_ned_velocity(action: Sequence[float]) -> np.ndarray:
    """策略动作 → PX4 TrajectorySetpoint 的 NED 速度。"""
    return enu_to_ned(action)


def yaw_enu_to_ned(yaw_enu: float) -> float:
    """ENU 偏航（从东起逆时针）→ NED 偏航（从北起顺时针）。"""
    return wrap_angle(math.pi / 2.0 - float(yaw_enu))


def yaw_ned_to_enu(yaw_ned: float) -> float:
    return wrap_angle(math.pi / 2.0 - float(yaw_ned))


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ENU 世界系下 Z-Y-X 欧拉角的旋转矩阵，列向量为机体轴在世界系下的方向。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quaternion_to_rotation(q_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise ValueError("四元数范数为 0")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


# NED→ENU 与 FRD→FLU 都是对称的轴置换矩阵。
_NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
_FRD_TO_FLU = np.diag([1.0, -1.0, -1.0])


def px4_attitude_to_enu_rotation(q_ned_frd_wxyz: Sequence[float]) -> np.ndarray:
    """PX4 姿态四元数（FRD 机体系 → NED 世界系）→ FLU 机体系 → ENU 世界系的旋转矩阵。"""
    rotation_ned_frd = quaternion_to_rotation(q_ned_frd_wxyz)
    return _NED_TO_ENU @ rotation_ned_frd @ _FRD_TO_FLU


def rotation_to_rpy(rotation: np.ndarray) -> Tuple[float, float, float]:
    """rotation_from_rpy 的逆，返回 (roll, pitch, yaw)。"""
    pitch = -math.asin(float(np.clip(rotation[2, 0], -1.0, 1.0)))
    roll = math.atan2(rotation[2, 1], rotation[2, 2])
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    return float(roll), float(pitch), float(yaw)
