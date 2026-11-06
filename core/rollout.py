#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""采集与闭环评估共用的单回合执行循环。

策略接口：policy(truth, observation, marker_visible, last_action) -> (action, metadata)，
metadata 必须包含 "phase"。专家使用全局真值，学习策略只使用 observation。
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.episode import EarlyAbort, Truth, judge_step
from core.observation import DetectionTracker, VisualMotionObservation, compute_reward

Policy = Callable[[Truth, np.ndarray, bool, np.ndarray], Tuple[np.ndarray, Dict[str, Any]]]


def run_episode(
    backend: Any,
    policy: Policy,
    policy_key: str,
    scenario: Dict[str, Any],
    seed: int,
    max_steps: int,
    episode_id: int,
    collection_mode: str,
    early_abort: Optional[EarlyAbort] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """执行一局并返回 (逐步记录, 回合摘要)。复位失败时返回空列表。"""
    first = backend.reset(scenario, seed)
    if first is None:
        return [], {"terminal_reason": "NO_INITIAL_DETECTION", "success": False, "steps": 0}

    tracker = DetectionTracker()
    tracker.update(True, first.position, first.confidence, 0.0)
    builder = VisualMotionObservation()
    observation = builder.initialize(first.position, first.confidence)
    if early_abort is not None:
        early_abort.reset()

    episode: List[Dict[str, Any]] = []
    marker_visible = True
    last_action = np.zeros(3, dtype=np.float32)
    success = False
    terminal_reason = "RUNNING"
    final_metrics: Dict[str, Any] = {}
    try:
        for step_index in range(max_steps):
            truth_before = backend.truth()
            action, metadata = policy(truth_before, observation, marker_visible, last_action)
            action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)
            abort_reason = None
            if early_abort is not None:
                abort_reason = early_abort.update(str(metadata.get("phase")), truth_before.relative_height())

            result = backend.step(action)
            detection = backend.detection()
            raw_position, fresh, confidence = tracker.update(
                detection.visible, detection.position, detection.confidence, result.sim_time
            )
            next_observation = builder.update(raw_position, fresh, confidence)
            truth_after = backend.truth()
            judgement = judge_step(
                truth_after, result.deck_contact, fresh, raw_position, backend.slot_origin, approach=truth_before
            )

            done, success, terminal_reason = judgement.done, judgement.success, judgement.terminal_reason
            # 早停在执行动作后才结束回合，保证最后一条 transition 完整、done 位于末尾。
            if not done and abort_reason is not None:
                done, terminal_reason = True, abort_reason
            if not done and step_index + 1 >= max_steps:
                done, terminal_reason = True, "MAX_STEPS"
            final_metrics = judgement.metrics

            episode.append({
                "observation": observation.astype(float).tolist(),
                "action": action.astype(float).tolist(),
                "reward": float(compute_reward(next_observation, done, success)),
                "next_observation": next_observation.astype(float).tolist(),
                "done": bool(done),
                "success": bool(success),
                "episode_id": int(episode_id),
                "step_index": int(step_index),
                "collection_mode": collection_mode,
                "terminal_reason": terminal_reason,
                "scenario": scenario,
                policy_key: metadata,
                "privileged_state": truth_before.to_metadata(),
                "next_privileged_state": truth_after.to_metadata(),
                "env_info": {
                    "sim_time": float(result.sim_time),
                    "marker_visible": bool(marker_visible),
                    "next_marker_visible": bool(fresh),
                    "observation_confidence": float(observation[-1]),
                    "next_observation_confidence": float(next_observation[-1]),
                    **judgement.metrics,
                },
            })
            observation = next_observation
            marker_visible = bool(fresh)
            last_action = action.copy()
            if done:
                break
    except Exception as error:  # 仿真异常时保留已有 transition 并写入 raw，供排查。
        success = False
        terminal_reason = "EXCEPTION"
        if episode:
            episode[-1].update(done=True, success=False, terminal_reason=terminal_reason, error=str(error))
        print(f"回合 {episode_id} 运行异常: {error}")
    finally:
        backend.hold()

    phases = [str((step.get(policy_key) or {}).get("phase")) for step in episode]
    summary = {
        "episode_id": int(episode_id),
        "label": scenario.get("label"),
        "success": bool(success),
        "terminal_reason": terminal_reason,
        "steps": len(episode),
        "search_steps": int(sum(phase == "SEARCH" for phase in phases)),
        "touchdown": final_metrics,
    }
    return episode, summary
