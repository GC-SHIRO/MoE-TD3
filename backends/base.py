#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""仿真后端接口：采集和评估只依赖这一层，mock 与 ROS2/Gazebo 两个实现可互换。"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from core.camera import Detection
from core.episode import Truth


@dataclass
class StepResult:
    sim_time: float          # 本局开始以来的仿真时间
    deck_contact: bool


class Backend:
    """一个 slot（一架无人机 + 一块甲板）的仿真接口。"""

    slot_origin: np.ndarray

    def reset(self, scenario: Dict[str, Any], seed: int) -> Optional[Detection]:
        """复位到新的场景并等待首个有效检测；在限定时间内看不到 marker 时返回 None。"""
        raise NotImplementedError

    def step(self, action: np.ndarray) -> StepResult:
        """下发一步动作，推进 TIME_DELTA 仿真时间。"""
        raise NotImplementedError

    def truth(self) -> Truth:
        raise NotImplementedError

    def detection(self) -> Detection:
        """本步最新的检测结果；没有新帧或空检测时 visible=False。"""
        raise NotImplementedError

    def hold(self) -> None:
        """回合结束后让无人机悬停，等待下一次复位。"""

    def close(self) -> None:
        pass


def make_backend(name: str, slot: int, config: Dict[str, Any]) -> Backend:
    """按名称创建后端；ROS2 依赖只在真正使用时导入，离线测试无需安装。"""
    if name == "mock":
        from backends.mock import MockBackend

        return MockBackend(slot, config)
    if name == "ros2_gz":
        from backends.ros2_gz import Ros2GzBackend

        return Ros2GzBackend(slot, config)
    raise ValueError(f"未知后端: {name}")
