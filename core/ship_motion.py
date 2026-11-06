#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""船甲板运动模型：水平运动叠加升沉、横摇、纵摇。

Python 模型是甲板真值的唯一来源；Gazebo 插件 landing_sim/src/DeckMotion.cc 使用同一套公式
（命令格式见 encode_plugin_command），两者必须同步修改。
船体偏航固定为 0（船头朝世界 +x），与旧场景一致；直线航向只决定速度方向。
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from core.frames import rotation_from_rpy


# ==================== 水平运动参数（沿用旧采集器） ====================
MOTION_CLASSES: Tuple[str, ...] = ("static", "line_constant", "line_variable")
MAX_SHIP_SPEED = 0.80
MIN_MOVING_SPEED = 0.20
VARIABLE_MIN_SPEED_UPPER = 0.50          # 变速下限的随机上界
MAX_HEADING_DEVIATION_DEG = 120.0        # 航向在船头 ±120° 内随机，避免倒退驶向障碍物
SPEED_SEGMENT_SECONDS = (2.0, 5.0)       # 变速分段时长范围
SCHEDULE_HORIZON_SECONDS = 120.0         # 必须大于单局最长仿真时间

# ==================== 摇荡参数（按海况等级） ====================
# 每个等级给出升沉振幅（米）和横摇、纵摇振幅（度）的上限；实际振幅在 [0.5, 1.0] 倍上限内随机。
SEA_STATES: Dict[int, Dict[str, float]] = {
    0: {"heave": 0.00, "roll_deg": 0.0, "pitch_deg": 0.0},
    1: {"heave": 0.10, "roll_deg": 2.0, "pitch_deg": 1.0},
    2: {"heave": 0.25, "roll_deg": 4.0, "pitch_deg": 2.0},
    3: {"heave": 0.50, "roll_deg": 7.0, "pitch_deg": 3.0},
    4: {"heave": 0.80, "roll_deg": 10.0, "pitch_deg": 5.0},
}
WAVE_PERIOD_RANGE = {"heave": (5.0, 9.0), "roll": (6.0, 12.0), "pitch": (4.0, 8.0)}
WAVE_COMPONENTS_PER_AXIS = 2
WAVE_AXES = ("heave", "roll", "pitch")

# ==================== 甲板几何（甲板坐标系，原点为船体模型原点） ====================
# marker 偏移沿用旧参数 MARKER_OFFSET_*；甲板半尺寸需与 landing_sim 中的船模型一致。
MARKER_OFFSET = (-0.23, 0.0, 1.3)
DECK_HALF_SIZE = (1.0, 1.0)
SHIP_BASE_Z = 0.1

_NUMERIC_EPS = 1e-3


@dataclass
class DeckState:
    """某一时刻的甲板真值，全部在世界 ENU 系。"""

    time: float
    position: np.ndarray            # 船体原点
    rpy: np.ndarray                 # 横摇、纵摇、偏航
    velocity: np.ndarray            # 船体原点线速度
    marker_position: np.ndarray
    marker_velocity: np.ndarray
    normal: np.ndarray              # 甲板法向（单位向量）

    def tilt_deg(self) -> float:
        return float(math.degrees(math.acos(float(np.clip(self.normal[2], -1.0, 1.0)))))

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "position": self.position.astype(float).tolist(),
            "rpy": self.rpy.astype(float).tolist(),
            "velocity": self.velocity.astype(float).tolist(),
            "marker_position": self.marker_position.astype(float).tolist(),
            "marker_velocity": self.marker_velocity.astype(float).tolist(),
            "tilt_deg": self.tilt_deg(),
        }


def scenario_label(motion_class: str, sea_state: int) -> str:
    return f"{motion_class}/sea{int(sea_state)}"


def sample_scenario(motion_class: str, sea_state: int, seed: int) -> Dict[str, Any]:
    """为一局生成可 JSON 序列化的甲板运动参数。"""
    if motion_class not in MOTION_CLASSES:
        raise ValueError(f"未知的运动类别: {motion_class}")
    if int(sea_state) not in SEA_STATES:
        raise ValueError(f"未知的海况等级: {sea_state}")
    rng = np.random.RandomState(int(seed))
    max_deviation = math.radians(MAX_HEADING_DEVIATION_DEG)
    heading = float(rng.uniform(-max_deviation, max_deviation))

    schedule: List[List[float]] = []
    if motion_class == "line_constant":
        schedule = [[0.0, float(rng.uniform(MIN_MOVING_SPEED, MAX_SHIP_SPEED))]]
    elif motion_class == "line_variable":
        speed_min = float(rng.uniform(MIN_MOVING_SPEED, VARIABLE_MIN_SPEED_UPPER))
        speed_max = float(rng.uniform(speed_min, MAX_SHIP_SPEED))
        t = 0.0
        while t <= SCHEDULE_HORIZON_SECONDS:
            schedule.append([t, float(rng.uniform(speed_min, speed_max))])
            t += float(rng.uniform(*SPEED_SEGMENT_SECONDS))

    limits = SEA_STATES[int(sea_state)]
    waves: List[Dict[str, Any]] = []
    for axis in WAVE_AXES:
        key = "heave" if axis == "heave" else f"{axis}_deg"
        limit = float(limits[key])
        if limit <= 0.0:
            continue
        total = limit * float(rng.uniform(0.5, 1.0))
        shares = rng.dirichlet(np.ones(WAVE_COMPONENTS_PER_AXIS))
        for share in shares:
            amplitude = total * float(share)
            if axis != "heave":
                amplitude = math.radians(amplitude)
            waves.append({
                "axis": axis,
                "amplitude": amplitude,
                "period": float(rng.uniform(*WAVE_PERIOD_RANGE[axis])),
                "phase": float(rng.uniform(0.0, 2.0 * math.pi)),
            })

    return {
        "motion_class": motion_class,
        "sea_state": int(sea_state),
        "label": scenario_label(motion_class, sea_state),
        "seed": int(seed),
        "heading": heading,
        "speed_schedule": schedule,
        "waves": waves,
    }


class DeckMotion:
    """由场景参数和仿真时间解析计算甲板位姿；t 从本局开始计时。"""

    def __init__(
        self,
        scenario: Dict[str, Any],
        origin_xy: Sequence[float],
        base_z: float = SHIP_BASE_Z,
        marker_offset: Sequence[float] = MARKER_OFFSET,
        deck_half_size: Sequence[float] = DECK_HALF_SIZE,
    ) -> None:
        self.scenario = scenario
        self.origin = np.array([float(origin_xy[0]), float(origin_xy[1]), float(base_z)])
        self.marker_offset = np.asarray(marker_offset, dtype=np.float64).reshape(3)
        self.deck_half_size = np.asarray(deck_half_size, dtype=np.float64).reshape(2)
        heading = float(scenario["heading"])
        self.direction = np.array([math.cos(heading), math.sin(heading), 0.0])
        self.schedule = [(float(t), float(v)) for t, v in scenario["speed_schedule"]]
        self.waves = list(scenario["waves"])

    # ---------- 水平运动 ----------
    def _speed(self, t: float) -> float:
        speed = 0.0
        for start, value in self.schedule:
            if t < start:
                break
            speed = value
        return speed

    def _distance(self, t: float) -> float:
        """分段常速的解析积分。"""
        distance = 0.0
        for index, (start, value) in enumerate(self.schedule):
            if t <= start:
                break
            end = self.schedule[index + 1][0] if index + 1 < len(self.schedule) else math.inf
            distance += value * (min(t, end) - start)
        return distance

    # ---------- 摇荡 ----------
    def _wave(self, axis: str, t: float) -> Tuple[float, float]:
        value = rate = 0.0
        for wave in self.waves:
            if wave["axis"] != axis:
                continue
            omega = 2.0 * math.pi / float(wave["period"])
            angle = omega * t + float(wave["phase"])
            value += float(wave["amplitude"]) * math.sin(angle)
            rate += float(wave["amplitude"]) * omega * math.cos(angle)
        return value, rate

    def pose(self, t: float) -> Tuple[np.ndarray, np.ndarray]:
        t = max(0.0, float(t))
        heave, _ = self._wave("heave", t)
        roll, _ = self._wave("roll", t)
        pitch, _ = self._wave("pitch", t)
        position = self.origin + self.direction * self._distance(t)
        position[2] += heave
        return position, np.array([roll, pitch, 0.0])

    def marker_position(self, t: float) -> np.ndarray:
        position, rpy = self.pose(t)
        return position + rotation_from_rpy(*rpy) @ self.marker_offset

    def state(self, t: float) -> DeckState:
        t = max(0.0, float(t))
        position, rpy = self.pose(t)
        rotation = rotation_from_rpy(*rpy)
        _, heave_rate = self._wave("heave", t)
        velocity = self.direction * self._speed(t)
        velocity[2] += heave_rate
        # marker 速度包含姿态角速度带来的切向分量，用中心差分避免手推角速度换算。
        earlier = max(0.0, t - _NUMERIC_EPS)
        later = t + _NUMERIC_EPS
        marker_velocity = (self.marker_position(later) - self.marker_position(earlier)) / (later - earlier)
        return DeckState(
            time=t,
            position=position,
            rpy=rpy,
            velocity=velocity,
            marker_position=position + rotation @ self.marker_offset,
            marker_velocity=marker_velocity,
            normal=rotation[:, 2].copy(),
        )

    def deck_contact(self, point: Sequence[float], t: float, tolerance: float = 0.0) -> bool:
        """点是否落在甲板平面上或以下，且位于甲板范围内（mock 后端的接触判定）。"""
        point = np.asarray(point, dtype=np.float64).reshape(3)
        position, rpy = self.pose(t)
        rotation = rotation_from_rpy(*rpy)
        # 甲板面以 marker 所在平面为准，范围以 marker 为中心。
        deck_center = position + rotation @ self.marker_offset
        local = rotation.T @ (point - deck_center)
        inside = bool(np.all(np.abs(local[:2]) <= self.deck_half_size))
        return inside and local[2] <= tolerance


def encode_plugin_command(scenario: Dict[str, Any], start_time: float, origin_xy: Sequence[float],
                          base_z: float = SHIP_BASE_Z) -> str:
    """生成 DeckMotion Gazebo 插件读取的空格分隔命令。

    格式：start_time x0 y0 z0 heading n_seg [t v]*n_seg n_wave [axis amplitude period phase]*n_wave
    axis 编码：0=heave，1=roll，2=pitch。
    """
    axis_code = {"heave": 0, "roll": 1, "pitch": 2}
    fields: List[str] = [
        f"{float(start_time):.6f}",
        f"{float(origin_xy[0]):.6f}",
        f"{float(origin_xy[1]):.6f}",
        f"{float(base_z):.6f}",
        f"{float(scenario['heading']):.9f}",
        str(len(scenario["speed_schedule"])),
    ]
    for t, v in scenario["speed_schedule"]:
        fields += [f"{float(t):.6f}", f"{float(v):.9f}"]
    fields.append(str(len(scenario["waves"])))
    for wave in scenario["waves"]:
        fields += [
            str(axis_code[wave["axis"]]),
            f"{float(wave['amplitude']):.9f}",
            f"{float(wave['period']):.9f}",
            f"{float(wave['phase']):.9f}",
        ]
    return " ".join(fields)
