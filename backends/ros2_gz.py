#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 Jazzy + Gazebo Harmonic + PX4（uXRCE-DDS）真实后端。

状态：按 doc/interfaces.md 的契约写完，尚未在真实仿真中运行；接入时按
doc/migration_from_ros1.md 的适配清单逐项确认。

一个实例控制一个 slot。PX4 与 Gazebo 锁步运行，step 按 /clock 等待 TIME_DELTA 仿真时间，
不暂停世界。甲板真值来自 core.ship_motion（与 DeckMotion 插件同一公式），
复位采用"相对复位"：无人机飞回起始点，甲板插件把船移回原点，不调用世界重置。
"""

import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from backends.base import Backend, StepResult
from core.camera import Detection, truth_detection
from core.episode import Truth
from core.frames import (
    HOLD_YAW_NED,
    action_to_ned_velocity,
    enu_to_ned,
    ned_to_enu,
    px4_attitude_to_enu_rotation,
    rotation_to_rpy,
)
from core.observation import TIME_DELTA
from core.ship_motion import DeckMotion, encode_plugin_command
from core.slots import sample_start_position, ship_origin_xy, slot_origin


# ==================== 默认参数（可由 collect.py 的 BACKEND_CONFIG 覆盖） ====================
DEFAULTS: Dict[str, Any] = {
    "detection_source": "truth",                 # "truth" 或 "yolo"
    # PX4 多机时实例 k 的话题前缀为 /px4_k；实例 0 默认无前缀。以锁定的 PX4 版本为准。
    "px4_namespace_template": "/px4_{slot}",
    "px4_namespace_slot0": "",
    "status_topic": "fmu/out/vehicle_status",    # 新版 PX4 可能为 vehicle_status_v1
    "odometry_topic": "fmu/out/vehicle_odometry",
    "detection_topic": "/landing/slot{slot}/detection",
    "contact_topic": "/landing/slot{slot}/deck_contact",
    "deck_command_topic": "/landing/slot{slot}/deck_command",
    "heartbeat_hz": 20.0,
    "arrive_tolerance": 0.3,                     # 米
    "arrive_speed": 0.2,                         # m/s
    "fly_to_timeout": 40.0,                      # 仿真秒
    "initial_detection_wait": 5.0,               # 仿真秒
    "settle_seconds": 0.5,
    "liftoff_height": 1.5,                       # 回合结束后先抬升，避免贴着甲板横移
    "startup_timeout": 60.0,                     # 墙钟秒，等待 PX4 话题和 /clock
}
HOLD_START_DELAY = 1.0e6                         # 复位期间把场景开始时间推到很远的将来


class Ros2GzBackend(Backend):
    def __init__(self, slot: int, config: Dict[str, Any]) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleOdometry, VehicleStatus
        from ros_gz_interfaces.msg import Contacts
        from landing_interfaces.msg import MarkerDetection

        self._msgs = {
            "OffboardControlMode": OffboardControlMode, "TrajectorySetpoint": TrajectorySetpoint,
            "VehicleCommand": VehicleCommand, "String": String,
        }
        self.cfg = {**DEFAULTS, **config}
        self.slot = int(slot)
        self.slot_origin = slot_origin(self.slot)
        self.rng = np.random.RandomState(int(self.cfg.get("seed", 0)) + 7919 * self.slot)
        self.quadrant = 0

        if not rclpy.ok():
            rclpy.init()
        self.node = Node(f"landing_backend_slot{self.slot}",
                         parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        namespace = self.cfg["px4_namespace_slot0"] if self.slot == 0 else self.cfg["px4_namespace_template"].format(slot=self.slot)
        px4_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST, depth=1)

        self._lock = threading.Lock()
        self._odometry = None
        self._status = None
        self._contact = False
        self._detection_msg = None
        self._detection_stamp = -1.0

        self.node.create_subscription(VehicleOdometry, f"{namespace}/{self.cfg['odometry_topic']}", self._on_odometry, px4_qos)
        self.node.create_subscription(VehicleStatus, f"{namespace}/{self.cfg['status_topic']}", self._on_status, px4_qos)
        self.node.create_subscription(Contacts, self.cfg["contact_topic"].format(slot=self.slot), self._on_contact, 10)
        self.node.create_subscription(MarkerDetection, self.cfg["detection_topic"].format(slot=self.slot), self._on_detection, 10)
        self._offboard_pub = self.node.create_publisher(OffboardControlMode, f"{namespace}/fmu/in/offboard_control_mode", px4_qos)
        self._setpoint_pub = self.node.create_publisher(TrajectorySetpoint, f"{namespace}/fmu/in/trajectory_setpoint", px4_qos)
        self._command_pub = self.node.create_publisher(VehicleCommand, f"{namespace}/fmu/in/vehicle_command", px4_qos)
        self._deck_pub = self.node.create_publisher(String, self.cfg["deck_command_topic"].format(slot=self.slot), 10)

        # offboard 要求持续心跳；心跳在后台线程中按固定频率发布最近一次 setpoint。
        self._mode = "position"
        self._position_target_ned = np.zeros(3)
        self._velocity_target_ned = np.zeros(3)
        self.node.create_timer(1.0 / float(self.cfg["heartbeat_hz"]), self._heartbeat)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        self.deck: Optional[DeckMotion] = None
        self.episode_start = 0.0
        self._wait_for_startup()
        self._arm_and_offboard()

    # ---------- 回调 ----------
    def _on_odometry(self, msg) -> None:
        with self._lock:
            self._odometry = msg

    def _on_status(self, msg) -> None:
        with self._lock:
            self._status = msg

    def _on_contact(self, msg) -> None:
        if len(msg.contacts) > 0:
            with self._lock:
                self._contact = True

    def _on_detection(self, msg) -> None:
        with self._lock:
            self._detection_msg = msg
            self._detection_stamp = self._now()

    # ---------- 时间 ----------
    def _now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _wait_sim(self, seconds: float) -> None:
        target = self._now() + float(seconds)
        while self._now() < target:
            time.sleep(0.001)

    # ---------- PX4 ----------
    def _px4_timestamp(self) -> int:
        return int(self.node.get_clock().now().nanoseconds / 1000)

    def _heartbeat(self) -> None:
        mode = self._msgs["OffboardControlMode"]()
        mode.timestamp = self._px4_timestamp()
        mode.position = self._mode == "position"
        mode.velocity = self._mode == "velocity"
        self._offboard_pub.publish(mode)

        setpoint = self._msgs["TrajectorySetpoint"]()
        setpoint.timestamp = mode.timestamp
        nan3 = [float("nan")] * 3
        if self._mode == "position":
            setpoint.position = [float(v) for v in self._position_target_ned]
            setpoint.velocity = nan3
        else:
            setpoint.position = nan3
            setpoint.velocity = [float(v) for v in self._velocity_target_ned]
        setpoint.acceleration = nan3
        setpoint.yaw = float(HOLD_YAW_NED)        # 偏航锁定为机头朝世界 +x
        setpoint.yawspeed = float("nan")
        self._setpoint_pub.publish(setpoint)

    def _send_command(self, command: int, param1: float = 0.0, param2: float = 0.0) -> None:
        msg = self._msgs["VehicleCommand"]()
        msg.timestamp = self._px4_timestamp()
        msg.command = int(command)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.target_system = self.slot + 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self._command_pub.publish(msg)

    def _wait_for_startup(self) -> None:
        deadline = time.monotonic() + float(self.cfg["startup_timeout"])
        while time.monotonic() < deadline:
            with self._lock:
                ready = self._odometry is not None and self._status is not None
            if ready and self._now() > 0.0:
                return
            time.sleep(0.1)
        raise RuntimeError(f"slot {self.slot}: 等待 PX4 odometry/status 与 /clock 超时，请检查 MicroXRCEAgent 与话题前缀")

    def _arm_and_offboard(self) -> None:
        """先持续发送 setpoint，再切 OFFBOARD 并解锁（PX4 的 offboard 进入条件）。"""
        self._position_target_ned = self._odometry_position_ned()
        self._mode = "position"
        self._wait_sim(1.0)
        VEHICLE_CMD_DO_SET_MODE, VEHICLE_CMD_COMPONENT_ARM_DISARM = 176, 400
        for _ in range(10):
            self._send_command(VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)     # custom main mode 6 = OFFBOARD
            self._send_command(VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            self._wait_sim(0.5)
            with self._lock:
                status = self._status
            ARMING_STATE_ARMED, NAVIGATION_STATE_OFFBOARD = 2, 14
            if status is not None and status.arming_state == ARMING_STATE_ARMED and status.nav_state == NAVIGATION_STATE_OFFBOARD:
                return
        raise RuntimeError(f"slot {self.slot}: 无法切换 OFFBOARD 并解锁")

    def _odometry_position_ned(self) -> np.ndarray:
        with self._lock:
            odometry = self._odometry
        return np.asarray(odometry.position, dtype=np.float64)

    def _fly_to(self, target_enu: np.ndarray) -> None:
        """位置模式飞到世界 ENU 目标点；PX4 本地原点假定为该 slot 的出生点。"""
        self._position_target_ned = enu_to_ned(np.asarray(target_enu) - self.slot_origin)
        self._mode = "position"
        deadline = self._now() + float(self.cfg["fly_to_timeout"])
        while self._now() < deadline:
            truth_position, truth_velocity = self._drone_state()[:2]
            if (np.linalg.norm(truth_position - target_enu) < self.cfg["arrive_tolerance"]
                    and np.linalg.norm(truth_velocity) < self.cfg["arrive_speed"]):
                self._wait_sim(self.cfg["settle_seconds"])
                return
            self._wait_sim(0.1)
        raise RuntimeError(f"slot {self.slot}: 飞往起始点超时")

    # ---------- 状态 ----------
    def _drone_state(self):
        with self._lock:
            odometry = self._odometry
        position = self.slot_origin + ned_to_enu(odometry.position)
        # 速度坐标系以 velocity_frame 字段为准；这里按 NED 处理，接入时确认。
        velocity = ned_to_enu(odometry.velocity)
        rotation = px4_attitude_to_enu_rotation(odometry.q)
        return position, velocity, rotation

    def _send_deck(self, scenario: Dict[str, Any], start_time: float) -> None:
        msg = self._msgs["String"]()
        msg.data = encode_plugin_command(scenario, start_time, ship_origin_xy(self.slot))
        self._deck_pub.publish(msg)

    # ---------- Backend 接口 ----------
    def reset(self, scenario: Dict[str, Any], seed: int) -> Optional[Detection]:
        self.rng = np.random.RandomState(int(seed))
        # 1. 船回到原点，并保持在本场景 t=0 的位姿（插件在 start_time 之前保持初始位姿），
        #    这样开始计时时甲板位姿连续，不会因摇荡初相位产生跳变。
        self.deck = DeckMotion(scenario, ship_origin_xy(self.slot))
        self._send_deck(scenario, self._now() + HOLD_START_DELAY)
        # 2. 无人机飞到起始点。
        start = sample_start_position(self.slot, self.quadrant, self.rng)
        self.quadrant += 1
        self._fly_to(start)
        # 3. 等待首个有效检测；此时甲板仍在原点，与场景 t=0 的位姿一致。
        deadline = self._now() + float(self.cfg["initial_detection_wait"])
        self.episode_start = self._now() + HOLD_START_DELAY   # 计时前甲板真值取 t=0
        first = None
        while self._now() < deadline:
            detection = self._read_detection(since=self._now() - TIME_DELTA)
            if detection.visible:
                first = detection
                break
            self._wait_sim(TIME_DELTA)
        if first is None:
            return None
        # 4. 从当前仿真时刻开始执行场景运动，并切到速度模式。
        self.episode_start = self._now()
        self._send_deck(scenario, self.episode_start)
        with self._lock:
            self._contact = False
        self._velocity_target_ned = np.zeros(3)
        self._mode = "velocity"
        return first

    def step(self, action: np.ndarray) -> StepResult:
        step_start = self._now()
        self._velocity_target_ned = action_to_ned_velocity(np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0))
        self._mode = "velocity"
        self._wait_sim(TIME_DELTA)
        self._last_step_start = step_start
        with self._lock:
            contact = self._contact
        return StepResult(sim_time=self._now() - self.episode_start, deck_contact=bool(contact))

    def truth(self) -> Truth:
        position, velocity, rotation = self._drone_state()
        _, _, yaw = rotation_to_rpy(rotation)
        return Truth(position, velocity, rotation, yaw, self.deck.state(self._now() - self.episode_start))

    def _read_detection(self, since: float) -> Detection:
        if self.cfg["detection_source"] == "truth":
            position, _, rotation = self._drone_state()
            marker = self.deck.state(self._now() - self.episode_start).marker_position
            return truth_detection(position, rotation, marker, self.rng)
        with self._lock:
            msg, stamp = self._detection_msg, self._detection_stamp
        # 本步没有收到新的检测消息时视为无视觉帧。
        if msg is None or stamp < since or not msg.visible:
            return Detection(False, np.zeros(3), 0.0)
        return Detection(True, np.array([msg.position.x, msg.position.y, msg.position.z]), float(msg.confidence))

    def detection(self) -> Detection:
        return self._read_detection(since=getattr(self, "_last_step_start", self._now() - TIME_DELTA))

    def hold(self) -> None:
        position = self._drone_state()[0]
        self._position_target_ned = enu_to_ned(position + np.array([0.0, 0.0, self.cfg["liftoff_height"]]) - self.slot_origin)
        self._mode = "position"

    def close(self) -> None:
        import rclpy

        self._executor.shutdown()
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
