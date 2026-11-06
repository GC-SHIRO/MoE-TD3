#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成包含 N 个 slot 的 Gazebo 世界。

slot 布局、marker 偏移和甲板尺寸直接读取仓库 core/，保证与 Python 真值模型一致。
仓库根目录取环境变量 LANDING_REPO_ROOT；未设置时按源码树相对路径推断。

用法：python generate_world.py --num-slots 2 --output /tmp/landing.sdf
"""

import argparse
import os
import sys

REPO_ROOT = os.environ.get(
    "LANDING_REPO_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")),
)
sys.path.insert(0, REPO_ROOT)

from core.ship_motion import DECK_HALF_SIZE, MARKER_OFFSET, SHIP_BASE_Z  # noqa: E402
from core.slots import ship_origin_xy, slot_origin  # noqa: E402


# ==================== 世界参数 ====================
WORLD_NAME = "landing"
PHYSICS_STEP = 0.004                 # 与 PX4 Gazebo 默认世界一致
HULL_SIZE = (4.0, 2.4, 1.0)          # 简化船体，只用于外观和遮挡
DECK_THICKNESS = 0.05
MARKER_SIZE = 1.0                    # marker 边长，需与 YOLO 距离估计参数一致
TAKEOFF_PAD_SIZE = 2.0
SHIP_MASS = 5000.0


def ship_sdf(slot: int) -> str:
    x, y = ship_origin_xy(slot)
    mx, my, mz = MARKER_OFFSET
    hx, hy = DECK_HALF_SIZE
    deck_z = mz - DECK_THICKNESS / 2.0
    return f"""
    <model name="ship_{slot}">
      <pose>{x} {y} {SHIP_BASE_Z} 0 0 0</pose>
      <link name="hull">
        <gravity>false</gravity>
        <inertial><mass>{SHIP_MASS}</mass>
          <inertia><ixx>{SHIP_MASS}</ixx><iyy>{SHIP_MASS}</iyy><izz>{SHIP_MASS}</izz></inertia>
        </inertial>
        <visual name="hull_visual">
          <pose>{mx} {my} {mz - DECK_THICKNESS - HULL_SIZE[2] / 2.0} 0 0 0</pose>
          <geometry><box><size>{HULL_SIZE[0]} {HULL_SIZE[1]} {HULL_SIZE[2]}</size></box></geometry>
          <material><ambient>0.6 0.6 0.6 1</ambient><diffuse>0.6 0.6 0.6 1</diffuse></material>
        </visual>
        <collision name="deck_collision">
          <pose>{mx} {my} {deck_z} 0 0 0</pose>
          <geometry><box><size>{2 * hx} {2 * hy} {DECK_THICKNESS}</size></box></geometry>
        </collision>
        <visual name="deck_visual">
          <pose>{mx} {my} {deck_z} 0 0 0</pose>
          <geometry><box><size>{2 * hx} {2 * hy} {DECK_THICKNESS}</size></box></geometry>
          <material><ambient>0.3 0.3 0.3 1</ambient><diffuse>0.3 0.3 0.3 1</diffuse></material>
        </visual>
        <visual name="marker_visual">
          <pose>{mx} {my} {mz + 0.001} 0 0 0</pose>
          <geometry><plane><normal>0 0 1</normal><size>{MARKER_SIZE} {MARKER_SIZE}</size></plane></geometry>
          <material>
            <diffuse>1 1 1 1</diffuse>
            <pbr><metal><albedo_map>model://landing_ship/materials/textures/marker.png</albedo_map></metal></pbr>
          </material>
        </visual>
        <sensor name="deck_contact" type="contact">
          <always_on>true</always_on>
          <update_rate>100</update_rate>
          <topic>/landing/slot{slot}/deck_contact</topic>
          <contact><collision>deck_collision</collision></contact>
        </sensor>
      </link>
      <plugin filename="DeckMotion" name="landing::DeckMotion">
        <command_topic>/landing/slot{slot}/deck_command</command_topic>
      </plugin>
    </model>"""


def takeoff_pad_sdf(slot: int) -> str:
    x, y, _ = slot_origin(slot)
    return f"""
    <model name="takeoff_pad_{slot}">
      <static>true</static>
      <pose>{x} {y} -0.05 0 0 0</pose>
      <link name="pad">
        <collision name="collision"><geometry><box><size>{TAKEOFF_PAD_SIZE} {TAKEOFF_PAD_SIZE} 0.1</size></box></geometry></collision>
        <visual name="visual"><geometry><box><size>{TAKEOFF_PAD_SIZE} {TAKEOFF_PAD_SIZE} 0.1</size></box></geometry></visual>
      </link>
    </model>"""


def world_sdf(num_slots: int, world_name: str = WORLD_NAME) -> str:
    slots = "".join(ship_sdf(k) + takeoff_pad_sdf(k) for k in range(num_slots))
    # 海面只有外观、没有碰撞体：无人机 z<0 即判为落水（与旧场景一致）。
    return f"""<?xml version="1.0"?>
<sdf version="1.9">
  <world name="{world_name}">
    <physics type="ode"><max_step_size>{PHYSICS_STEP}</max_step_size><real_time_factor>1.0</real_time_factor></physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-air-pressure-system" name="gz::sim::systems::AirPressure"/>
    <plugin filename="gz-sim-magnetometer-system" name="gz::sim::systems::Magnetometer"/>
    <plugin filename="gz-sim-navsat-system" name="gz::sim::systems::NavSat"/>
    <plugin filename="gz-sim-apply-link-wrench-system" name="gz::sim::systems::ApplyLinkWrench"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors"><render_engine>ogre2</render_engine></plugin>
    <gravity>0 0 -9.8</gravity>
    <magnetic_field>6e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type="adiabatic"/>
    <scene><grid>false</grid><ambient>0.4 0.4 0.4 1</ambient><background>0.7 0.8 0.9 1</background><shadows>false</shadows></scene>
    <light type="directional" name="sun">
      <cast_shadows>false</cast_shadows><pose>0 0 50 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse><specular>0.2 0.2 0.2 1</specular><direction>-0.5 0.1 -0.9</direction>
    </light>
    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model><world_frame_orientation>ENU</world_frame_orientation>
      <latitude_deg>47.397971057728974</latitude_deg><longitude_deg>8.546163739800146</longitude_deg>
      <elevation>0</elevation>
    </spherical_coordinates>
    <model name="sea">
      <static>true</static>
      <link name="surface">
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>2000 2000</size></plane></geometry>
          <material><ambient>0.1 0.25 0.45 1</ambient><diffuse>0.1 0.25 0.45 1</diffuse></material>
        </visual>
      </link>
    </model>{slots}
  </world>
</sdf>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="生成多 slot 降落世界")
    parser.add_argument("--num-slots", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-name", default=WORLD_NAME)
    args = parser.parse_args()
    if args.num_slots <= 0:
        raise SystemExit("num-slots 必须大于 0")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as output:
        output.write(world_sdf(args.num_slots, args.world_name))
    print(f"已生成世界: {args.output}（{args.num_slots} 个 slot）")


if __name__ == "__main__":
    main()
