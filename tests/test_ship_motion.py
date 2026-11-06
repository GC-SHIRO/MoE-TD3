#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""甲板运动测试：水平运动、摇荡幅度、marker 速度、接触判定和插件命令格式。"""

import math
import unittest

import numpy as np

from core.ship_motion import (
    MAX_HEADING_DEVIATION_DEG,
    MOTION_CLASSES,
    SEA_STATES,
    DeckMotion,
    encode_plugin_command,
    sample_scenario,
)


ORIGIN = (10.0, 5.0)


class ShipMotionTests(unittest.TestCase):
    def test_static_sea0_never_moves(self):
        deck = DeckMotion(sample_scenario("static", 0, 1), ORIGIN)
        start = deck.state(0.0)
        later = deck.state(30.0)
        np.testing.assert_allclose(start.marker_position, later.marker_position)
        np.testing.assert_allclose(later.marker_velocity, 0.0, atol=1e-9)
        self.assertAlmostEqual(later.tilt_deg(), 0.0)

    def test_constant_line_moves_along_heading(self):
        scenario = sample_scenario("line_constant", 0, 2)
        deck = DeckMotion(scenario, ORIGIN)
        speed = scenario["speed_schedule"][0][1]
        heading = scenario["heading"]
        displacement = deck.state(10.0).position - deck.state(0.0).position
        np.testing.assert_allclose(displacement[:2], [10 * speed * math.cos(heading), 10 * speed * math.sin(heading)])

    def test_heading_stays_in_front_of_bow(self):
        limit = math.radians(MAX_HEADING_DEVIATION_DEG)
        for seed in range(200):
            self.assertLessEqual(abs(sample_scenario("line_constant", 0, seed)["heading"]), limit + 1e-12)

    def test_variable_speed_distance_matches_numeric_integration(self):
        deck = DeckMotion(sample_scenario("line_variable", 0, 3), ORIGIN)
        dt = 0.001
        numeric = sum(deck._speed(i * dt) * dt for i in range(int(40.0 / dt)))
        self.assertAlmostEqual(deck._distance(40.0), numeric, places=2)

    def test_wave_amplitude_respects_sea_state(self):
        for sea in SEA_STATES:
            deck = DeckMotion(sample_scenario("static", sea, 4), ORIGIN)
            heave = [deck.pose(t)[0][2] - deck.origin[2] for t in np.arange(0.0, 60.0, 0.05)]
            roll = [deck.pose(t)[1][0] for t in np.arange(0.0, 60.0, 0.05)]
            self.assertLessEqual(max(abs(v) for v in heave), SEA_STATES[sea]["heave"] + 1e-9)
            self.assertLessEqual(max(abs(v) for v in roll), math.radians(SEA_STATES[sea]["roll_deg"]) + 1e-9)

    def test_marker_velocity_includes_rotation(self):
        """摇荡时 marker 速度应与位置的数值导数一致。"""
        deck = DeckMotion(sample_scenario("line_variable", 3, 5), ORIGIN)
        t, h = 12.3, 1e-4
        numeric = (deck.marker_position(t + h) - deck.marker_position(t - h)) / (2 * h)
        np.testing.assert_allclose(deck.state(t).marker_velocity, numeric, atol=1e-3)

    def test_deck_contact_inside_and_outside(self):
        deck = DeckMotion(sample_scenario("static", 0, 6), ORIGIN)
        marker = deck.state(0.0).marker_position
        self.assertTrue(deck.deck_contact(marker - [0.0, 0.0, 0.01], 0.0))
        self.assertFalse(deck.deck_contact(marker + [0.0, 0.0, 0.2], 0.0))
        self.assertFalse(deck.deck_contact(marker + [3.0, 0.0, -0.01], 0.0))

    def test_plugin_command_field_count(self):
        """命令字段数必须与 DeckMotion.cc 的解析顺序一致。"""
        for motion in MOTION_CLASSES:
            scenario = sample_scenario(motion, 2, 7)
            fields = encode_plugin_command(scenario, 3.5, ORIGIN).split()
            n_seg = int(fields[5])
            n_wave = int(fields[6 + 2 * n_seg])
            self.assertEqual(len(fields), 7 + 2 * n_seg + 4 * n_wave)
            self.assertEqual(n_seg, len(scenario["speed_schedule"]))
            self.assertEqual(n_wave, len(scenario["waves"]))

    def test_negative_time_holds_initial_pose(self):
        """复位期间（t<0）保持 t=0 位姿，开始计时时甲板不跳变。"""
        deck = DeckMotion(sample_scenario("line_variable", 3, 8), ORIGIN)
        np.testing.assert_allclose(deck.state(-5.0).marker_position, deck.state(0.0).marker_position)


if __name__ == "__main__":
    unittest.main()
