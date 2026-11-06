#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检测新鲜度、相机可见性、终止判定、早停、类别平衡和 worker 文件合并。"""

import json
import os
import tempfile
import unittest

import numpy as np

from core.camera import marker_in_view, truth_detection
from core.dataset import merge_files, prepare_output_files
from core.episode import EarlyAbort, Truth, choose_label, judge_step
from core.observation import DETECTION_LOST_TIMEOUT, DetectionTracker
from core.ship_motion import DeckMotion, sample_scenario


def make_truth(drone_position, drone_velocity=(0.0, 0.0, 0.0)):
    deck = DeckMotion(sample_scenario("static", 0, 1), (0.0, 0.0)).state(0.0)
    return Truth(np.asarray(drone_position, float), np.asarray(drone_velocity, float), np.eye(3), 0.0, deck)


class DetectionTrackerTests(unittest.TestCase):
    def test_short_dropout_keeps_position_but_zeroes_confidence(self):
        tracker = DetectionTracker()
        tracker.update(True, [1.0, 2.0, 3.0], 0.9, 0.0)
        position, fresh, confidence = tracker.update(False, [0.0, 0.0, 0.0], 0.0, 0.2)
        np.testing.assert_allclose(position, [1.0, 2.0, 3.0])
        self.assertTrue(fresh)
        self.assertEqual(confidence, 0.0)

    def test_dropout_longer_than_timeout_is_lost(self):
        tracker = DetectionTracker()
        tracker.update(True, [1.0, 2.0, 3.0], 0.9, 0.0)
        _, fresh, _ = tracker.update(False, [0.0, 0.0, 0.0], 0.0, DETECTION_LOST_TIMEOUT + 0.1)
        self.assertFalse(fresh)


class CameraTests(unittest.TestCase):
    def test_marker_below_is_visible_and_relative_position_is_drone_minus_marker(self):
        rng = np.random.RandomState(0)
        drone, marker = np.array([0.5, -0.3, 6.0]), np.array([0.0, 0.0, 1.4])
        self.assertIsNotNone(marker_in_view(drone, np.eye(3), marker))
        hits = [truth_detection(drone, np.eye(3), marker, rng) for _ in range(50)]
        visible = [d.position for d in hits if d.visible]
        self.assertGreater(len(visible), 40)
        np.testing.assert_allclose(np.mean(visible, axis=0), drone - marker, atol=0.05)

    def test_marker_above_or_far_aside_is_not_visible(self):
        self.assertIsNone(marker_in_view([0.0, 0.0, 1.0], np.eye(3), [0.0, 0.0, 3.0]))
        self.assertIsNone(marker_in_view([0.0, 0.0, 2.0], np.eye(3), [20.0, 0.0, 1.4]))


class JudgementTests(unittest.TestCase):
    def setUp(self):
        self.marker = make_truth([0, 0, 0]).deck.marker_position

    def test_contact_on_target_is_success(self):
        result = judge_step(make_truth(self.marker), True, True, [0, 0, 0], (0, 0, 0))
        self.assertEqual((result.done, result.success, result.terminal_reason), (True, True, "DECK_CONTACT_LANDED"))

    def test_contact_off_target_fails(self):
        result = judge_step(make_truth(self.marker + [0.5, 0, 0]), True, True, [0, 0, 0], (0, 0, 0))
        self.assertEqual(result.terminal_reason, "DECK_CONTACT_OFF_TARGET")
        self.assertFalse(result.success)

    def test_underwater_and_visual_out_of_bounds(self):
        self.assertEqual(judge_step(make_truth([3, 3, -0.1]), False, True, [0, 0, 0], (0, 0, 0)).terminal_reason,
                         "WORLD_OUT_OF_BOUNDS")
        self.assertEqual(judge_step(make_truth([3, 3, 5]), False, True, [25, 0, 3], (0, 0, 0)).terminal_reason,
                         "VISUAL_OUT_OF_BOUNDS")

    def test_impact_velocity_uses_approach_state(self):
        before = make_truth(self.marker + [0, 0, 0.05], drone_velocity=(0, 0, -0.4))
        after = make_truth(self.marker)
        result = judge_step(after, True, True, [0, 0, 0], (0, 0, 0), approach=before)
        self.assertAlmostEqual(result.metrics["relative_normal_velocity"], -0.4)


class EarlyAbortTests(unittest.TestCase):
    def test_long_search_streak_aborts(self):
        early = EarlyAbort(max_search_streak=3, stall_window_steps=100, min_descent_progress=0.3)
        reasons = [early.update("SEARCH", 5.0) for _ in range(4)]
        self.assertEqual(reasons[-1], "EARLY_ABORT_SEARCH")

    def test_stall_without_descent_aborts(self):
        early = EarlyAbort(max_search_streak=100, stall_window_steps=5, min_descent_progress=0.3)
        reasons = [early.update("TRACK", 5.0) for _ in range(6)]
        self.assertIn("EARLY_ABORT_STALL", reasons)


class BalancingAndFilesTests(unittest.TestCase):
    def test_choose_label_prefers_least_collected(self):
        rng = np.random.RandomState(0)
        self.assertEqual(choose_label({"a/sea0": 3, "b/sea1": 1, "c/sea2": 2}, rng), "b/sea1")

    def test_merge_appends_and_removes_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            sources = [os.path.join(folder, f"w{k}.jsonl") for k in range(2)]
            for k, path in enumerate(sources):
                with open(path, "w", encoding="utf-8") as output:
                    output.write(json.dumps([{"k": k}]) + "\n")
            destination = os.path.join(folder, "all.jsonl")
            self.assertEqual(merge_files(sources, destination), 2)
            self.assertFalse(any(os.path.exists(path) for path in sources))

    def test_training_and_raw_must_differ(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "same.jsonl")
            with self.assertRaises(ValueError):
                prepare_output_files(path, path, True)


if __name__ == "__main__":
    unittest.main()
