#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单世界多 slot 的布局与起始点采样；mock 后端、真实后端和 world 生成脚本共用。"""

from typing import Tuple

import numpy as np


# ==================== slot 布局参数 ====================
SLOT_SPACING = 200.0                 # slot 之间沿世界 x 的间距，需远大于 MAX_WORLD_DISTANCE
SHIP_OFFSET = (10.0, 5.0)            # 船初始位置相对 slot 原点（无人机出生点），沿用旧场景
START_HALF_RANGE = 4.0               # 起始点在船周围 ±4 m，保证起始时能看到 marker
START_HEIGHT_RANGE = (7.5, 9.0)


def slot_origin(slot: int) -> np.ndarray:
    """slot 原点即该 slot 无人机的出生点（世界 ENU，z=0）。"""
    return np.array([float(slot) * SLOT_SPACING, 0.0, 0.0])


def ship_origin_xy(slot: int) -> Tuple[float, float]:
    origin = slot_origin(slot)
    return float(origin[0] + SHIP_OFFSET[0]), float(origin[1] + SHIP_OFFSET[1])


def sample_start_position(slot: int, quadrant: int, rng: np.random.RandomState) -> np.ndarray:
    """四象限轮流采样起始点，保证各方向出现概率相同（与旧环境一致）。"""
    cx, cy = ship_origin_xy(slot)
    eps = 1e-6
    sign_x = 1.0 if quadrant % 4 in (0, 3) else -1.0
    sign_y = 1.0 if quadrant % 4 in (0, 1) else -1.0
    x = cx + sign_x * rng.uniform(eps, START_HALF_RANGE)
    y = cy + sign_y * rng.uniform(eps, START_HALF_RANGE)
    z = rng.uniform(*START_HEIGHT_RANGE)
    return np.array([x, y, z], dtype=np.float64)
