# 仿真镜像：Ubuntu 24.04 + ROS2 Jazzy + Gazebo Harmonic + PX4 SITL + MicroXRCEAgent。
# 版本号集中在下方 ARG；验证通过后把实际组合写进 doc/environment.md。
FROM osrf/ros:jazzy-desktop

ARG PX4_TAG=v1.16.0
ARG PX4_MSGS_BRANCH=release/1.16
ARG XRCE_AGENT_TAG=v2.4.3

ENV DEBIAN_FRONTEND=noninteractive \
    LANDING_REPO_ROOT=/workspace \
    PX4_DIR=/opt/PX4-Autopilot

RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3-pip python3-venv wget lsb-release gnupg \
      ros-jazzy-ros-gz ros-jazzy-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# PX4：只编译 SITL，不编译 NuttX 固件。
RUN git clone --branch ${PX4_TAG} --recursive --depth 1 https://github.com/PX4/PX4-Autopilot.git ${PX4_DIR} \
    && bash ${PX4_DIR}/Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools \
    && cd ${PX4_DIR} && make px4_sitl \
    && rm -rf /var/lib/apt/lists/*

# MicroXRCEAgent：PX4 uXRCE-DDS 的 ROS2 端。
RUN git clone --branch ${XRCE_AGENT_TAG} --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git /opt/xrce \
    && cmake -S /opt/xrce -B /opt/xrce/build && cmake --build /opt/xrce/build -j"$(nproc)" \
    && cmake --install /opt/xrce/build && ldconfig

# px4_msgs 分支必须与 PX4 版本一致，放在镜像内单独的工作空间。
RUN mkdir -p /opt/px4_ws/src \
    && git clone --branch ${PX4_MSGS_BRANCH} --depth 1 https://github.com/PX4/px4_msgs.git /opt/px4_ws/src/px4_msgs \
    && . /opt/ros/jazzy/setup.sh && cd /opt/px4_ws && colcon build --symlink-install

# Python 依赖：训练依赖与 YOLO。Ubuntu 24.04 的系统 Python 受 PEP 668 保护，使用 --break-system-packages。
COPY requirements.txt requirements-sim.txt /tmp/
RUN pip3 install --break-system-packages -r /tmp/requirements-sim.txt

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
WORKDIR /workspace
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
