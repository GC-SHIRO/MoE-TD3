#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局真值专家自动采集入口。

先修改下方参数，再运行：
    python collect.py           正式采集
    python collect.py --test    少量样本冒烟测试，写入独立测试文件
"""

import argparse
import multiprocessing
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backends.base import make_backend
from core.dataset import append_episode, count_lines, merge_files, prepare_output_files
from core.episode import EarlyAbort, choose_label, episode_is_trainable
from core.expert import ExpertConfig, GlobalLandingExpert
from core.rollout import run_episode
from core.ship_motion import MOTION_CLASSES, sample_scenario, scenario_label


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== 采集参数 ====================
BACKEND = "mock"                     # "mock" 只用于链路测试；接入仿真后改为 "ros2_gz"
NUM_SLOTS = 1                        # 并行 slot 数，每个 slot 一个 worker 进程
TARGET_SAVED_EPISODES = 300          # 全部 slot 合计，平均分配
MAX_ATTEMPTS = 1500                  # 全部 slot 合计，平均分配
MAX_STEPS = 600
RANDOM_SEED = 42
SLOT_SEED_STRIDE = 100_000           # slot k 的随机种子为 RANDOM_SEED + k * stride
SEA_STATES_TO_COLLECT = (0, 1, 2, 3)
EARLY_ABORT_ENABLED = True
CLEAR_OUTPUT_ON_START = False        # False 表示追加；开始新实验时改为 True 或换文件名

# ==================== 输出路径（按后端分目录，mock 数据不会混入正式数据） ====================
DATA_DIR = os.path.join(REPO_ROOT, "data", "expert", BACKEND)
TRAINING_OUTPUT = os.path.join(DATA_DIR, "global_expert.jsonl")
RAW_OUTPUT = os.path.join(DATA_DIR, "global_expert_raw.jsonl")
WORKER_DIR = os.path.join(DATA_DIR, "workers")

# ==================== 冒烟测试参数（--test） ====================
TEST_TARGET_SAVED_EPISODES = 2
TEST_MAX_ATTEMPTS = 5
TEST_TRAINING_OUTPUT = os.path.join(DATA_DIR, "global_expert_test.jsonl")
TEST_RAW_OUTPUT = os.path.join(DATA_DIR, "global_expert_test_raw.jsonl")

# ==================== 专家参数（沿用旧采集器） ====================
EXPERT_CONFIG = ExpertConfig(
    kp_xy=0.85, kd_xy=0.35, kp_z=0.85,
    align_xy_threshold=1.0, descend_xy_threshold=0.30, descend_vxy_threshold=0.25,
    stable_steps_required=2,
    tracking_height=2.0, flare_height=0.60, touchdown_height=0.15, near_ground_height=0.60,
    max_xy_speed=1.0, max_descent_speed=0.50, max_climb_speed=0.40,
    flare_descent_speed=0.20, touchdown_descent_speed=0.08, search_climb_speed=0.25,
    max_xy_delta=0.15, max_z_delta=0.10, max_action=1.0,
)

# ==================== 后端参数（只对 ros2_gz 生效，含义见 backends/ros2_gz.py） ====================
BACKEND_CONFIG: Dict[str, Any] = {
    "detection_source": "truth",     # "truth" 真值伪检测；YOLO 接入后改为 "yolo"
}


def parse_args(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全局真值专家自动采集")
    parser.add_argument("--test", action="store_true", help="少量样本冒烟测试，写入独立测试文件")
    return parser.parse_args(arguments)


def split_evenly(total: int, parts: int) -> List[int]:
    base, remainder = divmod(int(total), int(parts))
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def scenario_labels() -> List[str]:
    return [scenario_label(motion, sea) for sea in SEA_STATES_TO_COLLECT for motion in MOTION_CLASSES]


def expert_policy(expert: GlobalLandingExpert):
    def policy(truth, observation, marker_visible, last_action):
        command = expert.compute_action(
            *truth.expert_inputs(), marker_visible=marker_visible, last_action=last_action
        )
        return command.action, command.to_metadata()
    return policy


def run_worker(slot: int, target: int, max_attempts: int, training_output: str,
               raw_output: str, test_mode: bool) -> Dict[str, int]:
    """一个 slot 的采集循环，返回各场景标签的成功数量。"""
    seed = RANDOM_SEED + slot * SLOT_SEED_STRIDE
    backend = make_backend(BACKEND, slot, {**BACKEND_CONFIG, "seed": seed})
    expert = GlobalLandingExpert(EXPERT_CONFIG)
    early_abort = EarlyAbort() if EARLY_ABORT_ENABLED else None
    label_rng = np.random.RandomState(seed)
    saved_counts = {label: 0 for label in scenario_labels()}
    saved = attempt = 0
    try:
        while saved < target and attempt < max_attempts:
            attempt += 1
            label = choose_label(saved_counts, label_rng)
            motion_class, sea_text = label.split("/sea")
            episode_seed = seed + attempt - 1
            scenario = sample_scenario(motion_class, int(sea_text), episode_seed)
            expert.reset()
            episode_id = slot * SLOT_SEED_STRIDE + attempt
            try:
                episode, summary = run_episode(
                    backend, expert_policy(expert), "expert", scenario, episode_seed, MAX_STEPS,
                    episode_id, "test" if test_mode else "normal", early_abort,
                )
            except Exception as error:
                print(f"[slot {slot}] 第 {attempt:04d} 局复位失败: {error}")
                continue
            if not episode:
                print(f"[slot {slot}] 第 {attempt:04d} 局初始未看到 marker，重新开始")
                continue
            append_episode(raw_output, episode)
            trainable, reason = episode_is_trainable(episode)
            if summary["success"] and trainable:
                append_episode(training_output, episode)
                saved += 1
                saved_counts[label] += 1
                result = "已保存"
            else:
                result = f"未保存（{reason}）"
            print(
                f"[slot {slot}] 第 {attempt:04d} 局 | {label:22s} | {result} | 步数={summary['steps']:3d} | "
                f"SEARCH={summary['search_steps']:3d} | 终止={summary['terminal_reason']} | 进度={saved}/{target}"
            )
    finally:
        backend.close()
    if saved < target:
        print(f"[slot {slot}] 达到最大尝试次数，只保存了 {saved}/{target} 个成功回合")
    return saved_counts


def _worker_entry(arguments):
    return run_worker(*arguments)


def main() -> None:
    args = parse_args()
    if args.test:
        target, attempts = TEST_TARGET_SAVED_EPISODES, TEST_MAX_ATTEMPTS
        training_output, raw_output, clear = TEST_TRAINING_OUTPUT, TEST_RAW_OUTPUT, True
    else:
        target, attempts = TARGET_SAVED_EPISODES, MAX_ATTEMPTS
        training_output, raw_output, clear = TRAINING_OUTPUT, RAW_OUTPUT, CLEAR_OUTPUT_ON_START
    prepare_output_files(training_output, raw_output, clear)

    print("=" * 72)
    print("全局真值专家自动采集" + ("（冒烟测试）" if args.test else ""))
    print(f"后端: {BACKEND} | slot 数: {NUM_SLOTS} | 目标成功回合: {target} | 最大尝试: {attempts}")
    print(f"场景标签: {', '.join(scenario_labels())}")
    print(f"训练数据: {training_output}")
    print(f"原始数据: {raw_output}")
    print("=" * 72)

    if NUM_SLOTS == 1:
        counts = [run_worker(0, target, attempts, training_output, raw_output, args.test)]
    else:
        tag = "test_" if args.test else ""
        worker_files = [
            (os.path.join(WORKER_DIR, f"{tag}slot{k}.jsonl"), os.path.join(WORKER_DIR, f"{tag}slot{k}_raw.jsonl"))
            for k in range(NUM_SLOTS)
        ]
        leftovers = [path for pair in worker_files for path in pair if os.path.exists(path)]
        if leftovers:
            raise RuntimeError(f"存在上次未合并的 worker 文件，请先检查并手动合并: {leftovers}")
        jobs = [
            (k, t, a, worker_files[k][0], worker_files[k][1], args.test)
            for k, (t, a) in enumerate(zip(split_evenly(target, NUM_SLOTS), split_evenly(attempts, NUM_SLOTS)))
        ]
        with multiprocessing.Pool(NUM_SLOTS) as pool:
            counts = pool.map(_worker_entry, jobs)
        merged = merge_files([pair[0] for pair in worker_files], training_output)
        merge_files([pair[1] for pair in worker_files], raw_output)
        print(f"已合并 {merged} 个成功回合到 {training_output}")

    total = {label: sum(count[label] for count in counts) for label in scenario_labels()}
    print(f"各场景成功数量: {total}")
    print(f"训练文件现有回合数: {count_lines(training_output)}")


if __name__ == "__main__":
    main()
