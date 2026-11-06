#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定下视相机的可见性判定与真值伪检测。

mock 后端和真实后端的 "truth" 检测源共用这里，用于在 YOLO 接入前打通控制链路，
也可作为对比实验的"理想感知"。

检测位置语义（已确认，见 doc/interfaces.md）：旧 YOLO 节点输出预测目标的相对距离，
即米制的"无人机相对 marker"三维位置，轴向与世界 ENU 对齐（偏航固定为 0 时即机体轴）。
旧数据中 observation 前三维随降落从约 (5, 3, 9) 收敛到约 (0, 0, 0.5)，与此一致。
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ==================== 相机参数（需与 landing_sim 中的相机模型一致） ====================
CAMERA_HFOV_DEG = 80.0
CAMERA_ASPECT = 4.0 / 3.0            # 宽 / 高
CAMERA_MOUNT_OFFSET = (0.0, 0.0, -0.05)   # 机体 FLU 系下相机相对机体原点的位置
MAX_DETECTION_RANGE = 30.0
MIN_DETECTION_RANGE = 0.20           # 过近时 marker 超出画面或失焦

# ==================== 伪检测噪声参数 ====================
# 噪声为逐帧独立高斯，差分后会放大到速度和加速度；需用旧 YOLO 日志标定后再调整。
POSITION_NOISE_BASE = 0.01           # 米
POSITION_NOISE_PER_METER = 0.003
EDGE_MISS_PROBABILITY = 0.3          # 处在画面外 10% 边缘带时的漏检概率
RANDOM_MISS_PROBABILITY = 0.02
CONFIDENCE_BASE = 0.95
CONFIDENCE_DROP_PER_METER = 0.01
CONFIDENCE_NOISE = 0.03


@dataclass
class Detection:
    visible: bool
    position: np.ndarray        # 无人机相对 marker，米，世界 ENU 轴向
    confidence: float


def marker_in_view(
    drone_position: Sequence[float],
    drone_rotation: np.ndarray,
    marker_position: Sequence[float],
    edge_margin: float = 1.0,
) -> Optional[np.ndarray]:
    """marker 在视场内时返回归一化像平面坐标 (x/x_max, y/y_max)，否则返回 None。"""
    rotation = np.asarray(drone_rotation, dtype=np.float64).reshape(3, 3)
    camera = np.asarray(drone_position, dtype=np.float64) + rotation @ np.asarray(CAMERA_MOUNT_OFFSET)
    body = rotation.T @ (np.asarray(marker_position, dtype=np.float64) - camera)
    depth = -body[2]                      # 光轴朝机体 -z
    distance = float(np.linalg.norm(body))
    if depth <= 0.0 or not (MIN_DETECTION_RANGE <= distance <= MAX_DETECTION_RANGE):
        return None
    tan_half_h = math.tan(math.radians(CAMERA_HFOV_DEG) / 2.0)
    tan_half_v = tan_half_h / CAMERA_ASPECT
    normalized = np.array([body[1] / depth / tan_half_h, body[0] / depth / tan_half_v])
    if np.any(np.abs(normalized) > edge_margin):
        return None
    return normalized


def truth_detection(
    drone_position: Sequence[float],
    drone_rotation: np.ndarray,
    marker_position: Sequence[float],
    rng: np.random.RandomState,
) -> Detection:
    """按视场、距离和随机漏检生成伪检测。"""
    drone_position = np.asarray(drone_position, dtype=np.float64)
    marker_position = np.asarray(marker_position, dtype=np.float64)
    normalized = marker_in_view(drone_position, drone_rotation, marker_position)
    if normalized is None:
        return Detection(False, np.zeros(3), 0.0)
    miss = RANDOM_MISS_PROBABILITY
    if np.any(np.abs(normalized) > 0.9):
        miss = max(miss, EDGE_MISS_PROBABILITY)
    if rng.uniform() < miss:
        return Detection(False, np.zeros(3), 0.0)

    relative = drone_position - marker_position
    distance = float(np.linalg.norm(relative))
    sigma = POSITION_NOISE_BASE + POSITION_NOISE_PER_METER * distance
    noisy = relative + rng.normal(0.0, sigma, size=3)
    confidence = CONFIDENCE_BASE - CONFIDENCE_DROP_PER_METER * distance + rng.normal(0.0, CONFIDENCE_NOISE)
    return Detection(True, noisy, float(np.clip(confidence, 0.05, 1.0)))
