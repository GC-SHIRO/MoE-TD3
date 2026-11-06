#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""纯 Python 运动学后端：用于在没有 Gazebo/PX4/YOLO 时打通采集、训练和评估全链路。

无人机用一阶速度响应 + 加速度限幅近似 PX4 速度环，姿态由水平加速度反推，只用于相机可见性。
数据只能用于链路测试，不能作为正式训练数据。
"""

import math
from typing import Any, Dict, Optional

import numpy as np

from backends.base import Backend, StepResult
from core.camera import Detection, truth_detection
from core.episode import Truth
from core.frames import rotation_from_rpy
from core.observation import TIME_DELTA
from core.ship_motion import DeckMotion
from core.slots import sample_start_position, ship_origin_xy, slot_origin


# ==================== mock 动力学参数 ====================
VELOCITY_TIME_CONSTANT = 0.35        # 秒，速度环一阶响应时间常数
MAX_HORIZONTAL_ACCELERATION = 3.0    # m/s^2
MAX_VERTICAL_ACCELERATION = 2.0
SUBSTEPS = 10
GRAVITY = 9.81
INITIAL_DETECTION_TRIES = 20
CONTACT_TOLERANCE = 0.0


class MockBackend(Backend):
    def __init__(self, slot: int, config: Dict[str, Any]) -> None:
        self.slot = int(slot)
        self.config = dict(config)
        self.slot_origin = slot_origin(self.slot)
        self.rng = np.random.RandomState(int(config.get("seed", 0)) + 7919 * self.slot)
        self.quadrant = 0
        self.deck: Optional[DeckMotion] = None
        self.time = 0.0
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.contact = False
        self._detection = Detection(False, np.zeros(3), 0.0)

    def reset(self, scenario: Dict[str, Any], seed: int) -> Optional[Detection]:
        self.rng = np.random.RandomState(int(seed))
        self.deck = DeckMotion(scenario, ship_origin_xy(self.slot))
        self.time = 0.0
        self.position = sample_start_position(self.slot, self.quadrant, self.rng)
        self.quadrant += 1
        self.velocity = np.zeros(3)
        self.rotation = np.eye(3)
        self.contact = False
        marker = self.deck.state(0.0).marker_position
        for _ in range(INITIAL_DETECTION_TRIES):
            detection = truth_detection(self.position, self.rotation, marker, self.rng)
            if detection.visible:
                self._detection = detection
                return detection
        return None

    def step(self, action: np.ndarray) -> StepResult:
        if self.deck is None:
            raise RuntimeError("step 之前必须先 reset")
        command = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1.0, 1.0)
        dt = TIME_DELTA / SUBSTEPS
        end_time = self.time + TIME_DELTA
        for _ in range(SUBSTEPS):
            if self.contact:
                break
            acceleration = (command - self.velocity) / VELOCITY_TIME_CONSTANT
            horizontal = np.linalg.norm(acceleration[:2])
            if horizontal > MAX_HORIZONTAL_ACCELERATION:
                acceleration[:2] *= MAX_HORIZONTAL_ACCELERATION / horizontal
            acceleration[2] = float(np.clip(acceleration[2], -MAX_VERTICAL_ACCELERATION, MAX_VERTICAL_ACCELERATION))
            self.velocity = self.velocity + acceleration * dt
            self.position = self.position + self.velocity * dt
            self.time += dt
            # 水平加速度对应的倾角：向 +x 加速需要机头下压（ENU 中 pitch 为正）。
            pitch = math.atan2(acceleration[0], GRAVITY)
            roll = -math.atan2(acceleration[1], GRAVITY)
            self.rotation = rotation_from_rpy(roll, pitch, 0.0)
            if self.deck.deck_contact(self.position, self.time, CONTACT_TOLERANCE):
                self.contact = True
                # 触地后随甲板运动，速度取甲板速度。
                self.velocity = self.deck.state(self.time).marker_velocity.copy()
        # 触地提前结束子步时也推进完整周期，保证每步固定 TIME_DELTA。
        self.time = end_time
        marker = self.deck.state(self.time).marker_position
        self._detection = truth_detection(self.position, self.rotation, marker, self.rng)
        return StepResult(sim_time=float(self.time), deck_contact=bool(self.contact))

    def truth(self) -> Truth:
        if self.deck is None:
            raise RuntimeError("truth 之前必须先 reset")
        return Truth(
            drone_position=self.position.copy(),
            drone_velocity=self.velocity.copy(),
            drone_rotation=self.rotation.copy(),
            drone_yaw=0.0,
            deck=self.deck.state(self.time),
        )

    def detection(self) -> Detection:
        return self._detection

    def hold(self) -> None:
        self.velocity = np.zeros(3)
