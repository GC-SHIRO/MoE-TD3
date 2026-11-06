#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""十维视觉运动 observation、检测新鲜度判定和奖励。

语义与旧仓库 Sampling/collect_global_expert.py 保持一致，旧数据与新数据才能共用训练代码。
"""

import math
from typing import Any, Optional, Sequence, Tuple

import numpy as np


# ==================== observation 参数 ====================
TIME_DELTA = 0.1                 # 控制周期，按仿真时间计
STATE_DIM = 10
ACTION_DIM = 3
# 限幅只抑制偶发检测跳变，不能替代检测器标定。
MAX_VISUAL_SPEED = 10.0
MAX_VISUAL_ACCELERATION = 100.0
# 允许检测短暂漏帧；超过该时间才视为失检。
DETECTION_LOST_TIMEOUT = 0.5

# ==================== 奖励参数 ====================
SUCCESS_REWARD = 300.0
FAILURE_REWARD = -200.0
DISTANCE_REWARD_SCALE = -0.1


class DetectionTracker:
    """按仿真时间判断检测是否新鲜，失检期间保留最近一次有效位置。"""

    def __init__(self, lost_timeout: float = DETECTION_LOST_TIMEOUT) -> None:
        self.lost_timeout = float(lost_timeout)
        self.reset()

    def reset(self) -> None:
        self.last_position = np.zeros(3, dtype=np.float64)
        self.last_confidence = 0.0
        self.last_detection_time: Optional[float] = None

    def update(
        self,
        visible: bool,
        position: Sequence[float],
        confidence: float,
        sim_time: float,
    ) -> Tuple[np.ndarray, bool, float]:
        """输入本步检测，返回 (位置, 是否新鲜, 置信度)。

        空检测帧的置信度立即归零，不沿用上一帧数值；位置在超时前保持最近有效值。
        """
        if visible:
            self.last_position = vector3(position, "detection_position")
            self.last_confidence = clip_confidence(confidence)
            self.last_detection_time = float(sim_time)
        fresh = (
            self.last_detection_time is not None
            and float(sim_time) - self.last_detection_time <= self.lost_timeout
        )
        current_confidence = self.last_confidence if (fresh and visible) else 0.0
        return self.last_position.copy(), bool(fresh), float(current_confidence)


class VisualMotionObservation:
    """将连续检测位置转换为十维策略 observation。"""

    def __init__(self, time_delta: float = TIME_DELTA) -> None:
        if not math.isfinite(time_delta) or time_delta <= 0.0:
            raise ValueError("视觉状态差分时间间隔必须为正的有限数值")
        self.time_delta = float(time_delta)
        self.position = np.zeros(3, dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.acceleration = np.zeros(3, dtype=np.float32)
        self.has_previous_velocity = False

    def initialize(self, position: Sequence[float], confidence: float) -> np.ndarray:
        """每局第一帧只记录位置和置信度，导数统一置零。"""
        self.position = vector3(position, "initial_position").astype(np.float32)
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)
        self.has_previous_velocity = False
        return self._compose(confidence)

    def update(
        self, raw_position: Sequence[float], marker_visible: bool, confidence: float
    ) -> np.ndarray:
        """可见时差分；失检时保持位置并清零导数和置信度。"""
        if not marker_visible:
            self.velocity.fill(0.0)
            self.acceleration.fill(0.0)
            self.has_previous_velocity = False
            return self._compose(0.0)

        next_position = vector3(raw_position, "raw_position").astype(np.float32)
        next_velocity = np.clip(
            (next_position - self.position) / self.time_delta,
            -MAX_VISUAL_SPEED,
            MAX_VISUAL_SPEED,
        ).astype(np.float32)
        if self.has_previous_velocity:
            next_acceleration = np.clip(
                (next_velocity - self.velocity) / self.time_delta,
                -MAX_VISUAL_ACCELERATION,
                MAX_VISUAL_ACCELERATION,
            ).astype(np.float32)
        else:
            # 第一条速度及失检重获后的首条速度没有前一有效速度可比较。
            next_acceleration = np.zeros(3, dtype=np.float32)

        self.position = next_position
        self.velocity = next_velocity
        self.acceleration = next_acceleration
        self.has_previous_velocity = True
        return self._compose(confidence)

    def _compose(self, confidence: float) -> np.ndarray:
        return np.concatenate(
            (self.position, self.velocity, self.acceleration, [clip_confidence(confidence)])
        ).astype(np.float32)


def compute_reward(next_observation: Sequence[float], done: bool, success: bool) -> float:
    """连续距离奖励，并保留真实终止奖励。"""
    if done:
        return SUCCESS_REWARD if success else FAILURE_REWARD
    observation = state_vector(next_observation, "next_observation")
    distance = float(np.sum(np.abs(observation[:3]) ** 3) ** (1.0 / 3.0))
    return DISTANCE_REWARD_SCALE * distance


def vector3(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (3,):
        raise ValueError(f"{name} 必须是 3 维，当前形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限值: {vector}")
    return vector


def state_vector(value: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (STATE_DIM,):
        raise ValueError(f"{name} 必须是 {STATE_DIM} 维，当前形状为 {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} 包含非有限值: {vector}")
    return vector


def clip_confidence(value: Any) -> float:
    """将检测置信度收敛到策略输入范围，无效值视为 0。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(confidence):
        return 0.0
    return float(np.clip(confidence, 0.0, 1.0))
