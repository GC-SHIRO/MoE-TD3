#!/usr/bin/env bash
# 加载 ROS2、px4_msgs 和本仓库工作空间；首次进入时自动编译 ros2_ws。
set -e
source /opt/ros/jazzy/setup.bash
source /opt/px4_ws/install/setup.bash
if [ ! -f /workspace/ros2_ws/install/setup.bash ]; then
  echo "首次编译 ros2_ws ..."
  (cd /workspace/ros2_ws && colcon build --symlink-install)
fi
source /workspace/ros2_ws/install/setup.bash
exec "$@"
