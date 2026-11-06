#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""启动整套仿真：Gazebo 世界 → MicroXRCEAgent → 各 slot 的 PX4 → ros_gz_bridge →（可选）YOLO。

采集进程 python collect.py 单独在仓库根目录运行。

示例：
  ros2 launch landing_sim landing.launch.py num_slots:=2 headless:=true detection_source:=truth
"""

import os
import sys

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


# ==================== 启动参数默认值 ====================
DEFAULTS = {
    "num_slots": "1",
    "headless": "true",
    "world_name": "landing",
    "px4_dir": "/opt/PX4-Autopilot",
    # 带下视相机的 x500 机型；名称和 autostart 编号以锁定的 PX4 版本为准。
    "px4_model": "gz_x500_mono_cam_down",
    "px4_autostart": "4014",
    "xrce_port": "8888",
    "detection_source": "truth",        # truth 不启动 YOLO；yolo 启动检测节点
    "yolo_weights": "/workspace/weights/marker_yolo.pt",
    # PX4 生成的模型名为 <机型>_<实例号>；相机话题以实际模型 SDF 为准。
    "camera_topic_template": "/world/{world}/model/{model}_{slot}/link/camera_link/sensor/camera/image",
    "camera_info_template": "/world/{world}/model/{model}_{slot}/link/camera_link/sensor/camera/camera_info",
    "px4_start_delay": "5.0",          # Gazebo 启动后再拉起 PX4
    "px4_stagger": "4.0",              # 多个 PX4 实例错开启动
}


def _setup(context):
    get = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    num_slots = int(get("num_slots"))
    world = get("world_name")
    model = get("px4_model")
    model_name = model[3:] if model.startswith("gz_") else model
    share = get_package_share_directory("landing_sim")

    sys.path.insert(0, os.path.join(share, "scripts"))
    import generate_world  # noqa: E402
    from core.slots import slot_origin  # generate_world 已把仓库根目录加入 sys.path

    world_path = f"/tmp/{world}_{num_slots}slots.sdf"
    with open(world_path, "w", encoding="utf-8") as output:
        output.write(generate_world.world_sdf(num_slots, world))

    actions = [
        SetEnvironmentVariable("GZ_SIM_SYSTEM_PLUGIN_PATH", ":".join(filter(None, [
            os.path.join(get_package_prefix("landing_sim"), "lib", "landing_sim"),
            os.environ.get("GZ_SIM_SYSTEM_PLUGIN_PATH", "")]))),
        SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", ":".join(filter(None, [
            os.path.join(share, "models"),
            os.path.join(get("px4_dir"), "Tools", "simulation", "gz", "models"),
            os.environ.get("GZ_SIM_RESOURCE_PATH", "")]))),
        ExecuteProcess(cmd=["gz", "sim", "-r"] + (["-s"] if get("headless") == "true" else []) + [world_path],
                       output="screen"),
        ExecuteProcess(cmd=["MicroXRCEAgent", "udp4", "-p", get("xrce_port")], output="screen"),
    ]

    px4_build = os.path.join(get("px4_dir"), "build", "px4_sitl_default")
    for slot in range(num_slots):
        x, y, _ = slot_origin(slot)
        env = {
            **os.environ,
            "PX4_GZ_STANDALONE": "1",
            "PX4_GZ_WORLD": world,
            "PX4_SIM_MODEL": model,
            "PX4_SYS_AUTOSTART": get("px4_autostart"),
            # 出生偏航为 0（机头朝世界 +x），与动作坐标约定一致。
            "PX4_GZ_MODEL_POSE": f"{x},{y},0.1,0,0,0",
        }
        actions.append(TimerAction(
            period=float(get("px4_start_delay")) + slot * float(get("px4_stagger")),
            actions=[ExecuteProcess(cmd=[os.path.join(px4_build, "bin", "px4"), "-i", str(slot)],
                                    cwd=px4_build, additional_env=env, output="screen")],
        ))

    bridge_args = ["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"]
    remaps = []
    for slot in range(num_slots):
        bridge_args += [
            f"/landing/slot{slot}/deck_contact@ros_gz_interfaces/msg/Contacts[gz.msgs.Contacts",
            f"/landing/slot{slot}/deck_command@std_msgs/msg/String]gz.msgs.StringMsg",
        ]
        if get("detection_source") == "yolo":
            image = get("camera_topic_template").format(world=world, model=model_name, slot=slot)
            info = get("camera_info_template").format(world=world, model=model_name, slot=slot)
            bridge_args += [f"{image}@sensor_msgs/msg/Image[gz.msgs.Image",
                            f"{info}@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo"]
            remaps += [(image, f"/landing/slot{slot}/camera/image"),
                       (info, f"/landing/slot{slot}/camera/camera_info")]
    actions.append(Node(package="ros_gz_bridge", executable="parameter_bridge", arguments=bridge_args,
                        remappings=remaps, parameters=[{"use_sim_time": True}], output="screen"))

    if get("detection_source") == "yolo":
        for slot in range(num_slots):
            actions.append(Node(
                package="landing_perception", executable="yolo_node", name=f"yolo_slot{slot}", output="screen",
                parameters=[{"use_sim_time": True, "slot": slot, "weights": get("yolo_weights")}],
            ))
    return actions


def generate_launch_description():
    arguments = [DeclareLaunchArgument(name, default_value=value) for name, value in DEFAULTS.items()]
    return LaunchDescription(arguments + [OpaqueFunction(function=_setup)])
