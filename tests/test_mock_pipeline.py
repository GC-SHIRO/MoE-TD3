#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mock 后端全链路：采集 → 校验 →（有 torch 时）Stage 0 训练 → 闭环评估。"""

import importlib.util
import os
import tempfile
import unittest
from dataclasses import replace

import collect
from backends.mock import MockBackend
from core.rollout import run_episode
from core.ship_motion import sample_scenario
from core.validate import read_episodes, validate_dataset

HAS_TORCH = importlib.util.find_spec("torch") is not None


class MockPipelineTests(unittest.TestCase):
    def test_collect_writes_valid_continuous_episodes(self):
        with tempfile.TemporaryDirectory() as folder:
            training = os.path.join(folder, "train.jsonl")
            raw = os.path.join(folder, "raw.jsonl")
            counts = collect.run_worker(0, 2, 6, training, raw, test_mode=True)
            self.assertEqual(sum(counts.values()), 2)
            episodes = read_episodes(training)
            errors, statistics = validate_dataset(episodes)
            self.assertEqual(errors, [])
            for episode in episodes:
                self.assertEqual(episode[-1]["terminal_reason"], "DECK_CONTACT_LANDED")
                self.assertIn("phase", episode[0]["expert"])
                for current, following in zip(episode, episode[1:]):
                    self.assertEqual(current["next_observation"], following["observation"])

    def test_episode_is_reproducible_with_same_seed(self):
        scenario = sample_scenario("line_variable", 2, 11)
        lengths = []
        for _ in range(2):
            expert = collect.GlobalLandingExpert(collect.EXPERT_CONFIG)
            episode, _ = run_episode(MockBackend(0, {"seed": 0}), collect.expert_policy(expert), "expert",
                                     scenario, 11, 600, 1, "test")
            lengths.append((len(episode), episode[-1]["action"]))
        self.assertEqual(lengths[0], lengths[1])

    @unittest.skipUnless(HAS_TORCH, "需要 PyTorch")
    def test_stage0_checkpoint_runs_closed_loop(self):
        from model.moe_td3 import Args
        from scripts.train import stage0_single_head

        import evaluate

        with tempfile.TemporaryDirectory() as folder:
            training = os.path.join(folder, "train.jsonl")
            collect.run_worker(0, 4, 10, training, os.path.join(folder, "raw.jsonl"), test_mode=True)
            args = replace(Args(), data_path=training, training_steps=3, batch_size=8, hidden_dim=32,
                           transformer_ffn_dim=64, router_hidden_dim=16, log_every=1, save_every=3,
                           validation_fraction=0.5)
            final = stage0_single_head.run(args=args, output_dir=os.path.join(folder, "exp", "stage0"), eval_every=3)
            policy, reset = evaluate.moe_policy(str(final), "hard")
            reset()
            _, summary = run_episode(MockBackend(0, {"seed": 1}), policy, "policy",
                                     sample_scenario("static", 1, 5), 5, 30, 1, "evaluation")
            self.assertGreater(summary["steps"], 0)


if __name__ == "__main__":
    unittest.main()
