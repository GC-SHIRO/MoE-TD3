#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单回合的终止判定、早停、类别平衡和入库检查，与仿真后端无关。"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from core.observation import ACTION_DIM, state_vector, vector3
from core.ship_motion import DeckState


# ==================== 判定参数（沿用旧采集器） ====================
LANDING_XY_THRESHOLD = 0.30
MAX_WORLD_DISTANCE = 60.0        # 相对 slot 原点
MAX_FLIGHT_HEIGHT = 12.0
VISUAL_XY_LIMIT = 20.0           # 检测位置水平分量越界即终止
MIN_SUCCESS_STEPS = 15

# ==================== 早停参数（沿用旧并行采集器） ====================
MAX_SEARCH_STREAK = 40
STALL_WINDOW_STEPS = 150
MIN_DESCENT_PROGRESS = 0.3


@dataclass
class Truth:
    """专家与判定使用的全局真值，世界 ENU 系。"""

    drone_position: np.ndarray
    drone_velocity: np.ndarray
    drone_rotation: np.ndarray       # FLU 机体系 → ENU 世界系
    drone_yaw: float
    deck: DeckState

    def expert_inputs(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
        return (
            self.drone_position,
            self.drone_velocity,
            float(self.drone_yaw),
            self.deck.marker_position,
            self.deck.marker_velocity,
        )

    def relative_height(self) -> float:
        return float(self.drone_position[2] - self.deck.marker_position[2])

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "drone_position": self.drone_position.astype(float).tolist(),
            "drone_velocity": self.drone_velocity.astype(float).tolist(),
            "drone_yaw": float(self.drone_yaw),
            "target_position": self.deck.marker_position.astype(float).tolist(),
            "target_velocity": self.deck.marker_velocity.astype(float).tolist(),
            "deck": self.deck.to_metadata(),
        }


@dataclass
class StepJudgement:
    done: bool = False
    success: bool = False
    terminal_reason: str = "RUNNING"
    metrics: Dict[str, Any] = field(default_factory=dict)


def judge_step(
    truth: Truth,
    deck_contact: bool,
    detection_fresh: bool,
    detection_position: Sequence[float],
    slot_origin: Sequence[float],
    approach: Optional[Truth] = None,
) -> StepJudgement:
    """动作执行后判定是否结束。接触甲板即结束，位置在阈值内才算成功。

    approach 为执行动作前的真值：触地后速度会被接触改变，冲击速度应取接触前的相对速度。
    """
    relative = truth.drone_position - truth.deck.marker_position
    velocity_source = approach if approach is not None else truth
    relative_velocity = velocity_source.drone_velocity - velocity_source.deck.marker_velocity
    xy_distance = float(math.hypot(relative[0], relative[1]))
    metrics = {
        "relative_xy_distance": xy_distance,
        "relative_height": float(relative[2]),
        "relative_xy_speed": float(math.hypot(relative_velocity[0], relative_velocity[1])),
        # 相对法向速度为负表示正在接近甲板面；触地时它反映冲击强度。
        "relative_normal_velocity": float(np.dot(relative_velocity, truth.deck.normal)),
        "deck_tilt_deg": truth.deck.tilt_deg(),
        "deck_contact": bool(deck_contact),
    }

    if deck_contact:
        success = xy_distance <= LANDING_XY_THRESHOLD
        reason = "DECK_CONTACT_LANDED" if success else "DECK_CONTACT_OFF_TARGET"
        return StepJudgement(True, success, reason, metrics)

    detection_position = np.asarray(detection_position, dtype=np.float64)
    if detection_fresh and np.any(np.abs(detection_position[:2]) > VISUAL_XY_LIMIT):
        return StepJudgement(True, False, "VISUAL_OUT_OF_BOUNDS", metrics)

    offset = truth.drone_position - np.asarray(slot_origin, dtype=np.float64)
    if (
        float(np.linalg.norm(offset)) > MAX_WORLD_DISTANCE
        or truth.drone_position[2] > MAX_FLIGHT_HEIGHT
        or truth.drone_position[2] < 0.0
    ):
        # z < 0 表示落水，必须立即终止。
        return StepJudgement(True, False, "WORLD_OUT_OF_BOUNDS", metrics)

    return StepJudgement(False, False, "RUNNING", metrics)


class EarlyAbort:
    """连续 SEARCH 过长或相对高度长期不下降时提前结束回合。"""

    def __init__(
        self,
        max_search_streak: int = MAX_SEARCH_STREAK,
        stall_window_steps: int = STALL_WINDOW_STEPS,
        min_descent_progress: float = MIN_DESCENT_PROGRESS,
    ) -> None:
        if max_search_streak <= 0 or stall_window_steps <= 0 or min_descent_progress <= 0.0:
            raise ValueError("早停阈值必须为正数")
        self.max_search_streak = int(max_search_streak)
        self.stall_window_steps = int(stall_window_steps)
        self.min_descent_progress = float(min_descent_progress)
        self.reset()

    def reset(self) -> None:
        self.step = 0
        self.search_streak = 0
        self.best_height: Optional[float] = None
        self.best_step = 0

    def update(self, phase: str, relative_height: float) -> Optional[str]:
        """每步在执行动作前调用一次；返回提前终止原因，继续执行返回 None。"""
        self.step += 1
        self.search_streak = self.search_streak + 1 if phase == "SEARCH" else 0
        if self.search_streak > self.max_search_streak:
            return "EARLY_ABORT_SEARCH"
        height = float(relative_height)
        if self.best_height is None or height <= self.best_height - self.min_descent_progress:
            self.best_height, self.best_step = height, self.step
        elif self.step - self.best_step >= self.stall_window_steps:
            return "EARLY_ABORT_STALL"
        return None


def choose_label(saved_counts: Dict[str, int], rng: np.random.RandomState) -> str:
    """优先补齐已保存数量最少的场景标签。"""
    minimum = min(saved_counts.values())
    candidates = sorted(name for name, count in saved_counts.items() if count == minimum)
    return candidates[int(rng.randint(len(candidates)))]


def episode_is_trainable(episode: Sequence[Dict[str, Any]]) -> Tuple[bool, str]:
    """进入训练文件前的必要结构检查。"""
    if len(episode) < MIN_SUCCESS_STEPS:
        return False, "回合长度不足"
    if not bool(episode[-1].get("success", False)):
        return False, "回合没有成功落地"
    for index, step in enumerate(episode):
        try:
            next_observation = state_vector(step["next_observation"], "next_observation")
            state_vector(step["observation"], "observation")
            action = vector3(step["action"], "action")
            reward = float(step["reward"])
        except (KeyError, TypeError, ValueError) as error:
            return False, f"第 {index} 步字段错误: {error}"
        if action.shape != (ACTION_DIM,) or np.any(np.abs(action) > 1.0 + 1e-6):
            return False, f"第 {index} 步 action 超出 [-1, 1]"
        if not math.isfinite(reward):
            return False, f"第 {index} 步 reward 非有限"
        if bool(step["done"]) != (index == len(episode) - 1):
            return False, f"第 {index} 步 done 位置错误"
        if index + 1 < len(episode):
            following = state_vector(episode[index + 1]["observation"], "following_observation")
            if not np.array_equal(next_observation, following):
                return False, f"第 {index} 步 next_observation 链断裂"
    return True, "ok"
