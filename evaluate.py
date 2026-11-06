#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭环评估入口：按场景标签统计成功率、落点误差和触地冲击。

先修改下方参数，再运行：
    python evaluate.py
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from backends.base import make_backend
from core.expert import GlobalLandingExpert
from core.ship_motion import MOTION_CLASSES, sample_scenario, scenario_label
from core.rollout import run_episode


REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== 评估参数 ====================
BACKEND = "mock"
POLICY = "expert"                    # "expert"（特权专家上限）或 "moe"（训练存档）
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "MoE_TD3", "experiment_01", "stage2", "final.pt")
MOE_MODE = "hard"                    # MoE 动作取最大权重专家（hard）或加权平均（soft）
EPISODES_PER_LABEL = 10
SEA_STATES_TO_EVALUATE = (0, 1, 2, 3, 4)
MAX_STEPS = 600
EVAL_SEED = 10_000_000               # 与采集种子区间错开，避免评估用到训练场景
SLOT = 0

# ==================== 输出路径 ====================
RUN_NAME = f"{BACKEND}_{POLICY}"
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "evaluation", RUN_NAME)

# 专家参数与采集保持一致。
from collect import BACKEND_CONFIG, EXPERT_CONFIG, expert_policy  # noqa: E402


def moe_policy(checkpoint_path: str, mode: str):
    """读取训练存档，返回只使用 observation 的策略函数。"""
    from model.moe_td3 import Args, OfflineMoETD3BC, phase_name
    from scripts.train.common import load_checkpoint

    payload = load_checkpoint(checkpoint_path)
    args = Args(**payload["args"])
    agent = OfflineMoETD3BC(args, single_head=payload["stage"] == 0)
    agent.actor.load_state_dict(payload["actor"])
    mean = np.asarray(payload["data"]["mean"], dtype=np.float32)
    std = np.asarray(payload["data"]["std"], dtype=np.float32)
    state = {"history": [], "phase": 0}

    def reset() -> None:
        state["history"] = []
        state["phase"] = 0

    def policy(truth, observation, marker_visible, last_action):
        state["history"].append((np.asarray(observation, dtype=np.float32) - mean) / std)
        state["history"] = state["history"][-args.seq_len:]
        action, phase = agent.choose_action(np.stack(state["history"]), state["phase"], mode=mode)
        state["phase"] = int(phase)
        return action, {"phase": phase_name(phase) if not agent.single_head else "SINGLE_HEAD"}

    return policy, reset


def summarize(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in summaries:
        by_label[item["label"]].append(item)
    by_label["ALL"] = list(summaries)
    table = {}
    for label, items in by_label.items():
        successes = [item for item in items if item["success"]]
        touchdowns = [item["touchdown"] for item in successes if item["touchdown"]]
        reasons: Dict[str, int] = defaultdict(int)
        for item in items:
            reasons[item["terminal_reason"]] += 1
        table[label] = {
            "episodes": len(items),
            "success_rate": len(successes) / len(items) if items else None,
            "mean_xy_error": _mean([t["relative_xy_distance"] for t in touchdowns]),
            "mean_abs_normal_velocity": _mean([abs(t["relative_normal_velocity"]) for t in touchdowns]),
            "mean_deck_tilt_deg": _mean([t["deck_tilt_deg"] for t in touchdowns]),
            "mean_steps": _mean([item["steps"] for item in successes]),
            "mean_search_steps": _mean([item["search_steps"] for item in items]),
            "terminal_reasons": dict(reasons),
        }
    return table


def _mean(values):
    return float(np.mean(values)) if values else None


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    backend = make_backend(BACKEND, SLOT, {**BACKEND_CONFIG, "seed": EVAL_SEED})
    if POLICY == "expert":
        expert = GlobalLandingExpert(EXPERT_CONFIG)
        policy, reset, key = expert_policy(expert), expert.reset, "expert"
    elif POLICY == "moe":
        policy, reset = moe_policy(CHECKPOINT, MOE_MODE)
        key = "policy"
    else:
        raise ValueError(f"未知策略: {POLICY}")

    labels = [scenario_label(m, s) for s in SEA_STATES_TO_EVALUATE for m in MOTION_CLASSES]
    episodes_path = os.path.join(OUTPUT_DIR, "episodes.jsonl")
    open(episodes_path, "w", encoding="utf-8").close()
    summaries: List[Dict[str, Any]] = []
    episode_id = 0
    try:
        for label in labels:
            motion_class, sea_text = label.split("/sea")
            for repeat in range(EPISODES_PER_LABEL):
                episode_id += 1
                seed = EVAL_SEED + episode_id
                scenario = sample_scenario(motion_class, int(sea_text), seed)
                reset()
                _, summary = run_episode(backend, policy, key, scenario, seed, MAX_STEPS, episode_id, "evaluation")
                summaries.append(summary)
                with open(episodes_path, "a", encoding="utf-8") as output:
                    output.write(json.dumps(summary, ensure_ascii=False) + "\n")
                print(f"{label:22s} #{repeat + 1:02d} | 成功={summary['success']} | 终止={summary['terminal_reason']} | 步数={summary['steps']}")
    finally:
        backend.close()

    table = summarize(summaries)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as output:
        json.dump({"backend": BACKEND, "policy": POLICY, "checkpoint": CHECKPOINT if POLICY == "moe" else None,
                   "table": table}, output, ensure_ascii=False, indent=2)
    print(f"\n{'场景':22s} {'回合':>4s} {'成功率':>7s} {'落点误差':>8s} {'法向速度':>8s} {'倾角':>6s}")
    for label, row in table.items():
        def fmt(value, pattern):
            return pattern.format(value) if value is not None else "-"
        print(f"{label:22s} {row['episodes']:4d} {fmt(row['success_rate'], '{:7.1%}')} "
              f"{fmt(row['mean_xy_error'], '{:8.3f}')} {fmt(row['mean_abs_normal_velocity'], '{:8.3f}')} "
              f"{fmt(row['mean_deck_tilt_deg'], '{:6.2f}')}")
    print(f"结果: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
