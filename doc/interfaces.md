# 接口契约

## 1. 数据语义（与旧数据兼容的前提）

| 项 | 约定 | 代码位置 |
| --- | --- | --- |
| 控制周期 | `TIME_DELTA=0.1s`，按仿真时间 | `core/observation.py` |
| 动作 | 3 维，`[-1,1]`，偏航固定为 0 时的机体系速度，等同世界 ENU `(vx, vy, vz)`，z 向上为正 | `core/frames.py` |
| observation | 检测位置 3 + 差分速度 3 + 差分加速度 3 + 置信度 1；速度限幅 10、加速度限幅 100 | `core/observation.py` |
| 检测位置 | YOLO 预测的目标相对距离：无人机相对 marker 的三维位置，米，轴向与世界 ENU 对齐（偏航固定为 0 时即机体轴） | `core/camera.py` |
| 失检 | 0.5 s 内保持；超时后位置保持、导数与置信度清零；空检测帧置信度立即为 0 | `core/observation.py` |
| 终止 | 接触甲板即结束，水平误差 ≤ 0.30 m 为成功；落水、越界、视觉越界失败 | `core/episode.py` |
| 奖励 | 成功 +300，失败 -200，过程 `-0.1 × L3 距离` | `core/observation.py` |
| 场景标签 | `运动类别/sea海况`，如 `line_constant/sea3`；按标签平衡成功数 | `core/ship_motion.py` |

每步除五个核心字段外还记录：`expert`（含 `phase`）、`scenario`、`privileged_state`、
`next_privileged_state`、`env_info`（含 `relative_normal_velocity`、`deck_tilt_deg` 等触地指标）。

## 2. 船甲板

- 船体偏航固定为 0（船头朝世界 +x）；直线航向在 ±120° 内随机，只决定速度方向。
- 姿态：升沉、横摇、纵摇，每轴两个正弦分量；海况 0–4 的幅度上限见 `SEA_STATES`。
- marker 位置 = 船体原点 + 旋转后的 `MARKER_OFFSET`；marker 速度包含姿态角速度带来的切向分量。
- 复位期间甲板保持场景 t=0 的位姿，开始计时时不跳变。
- Gazebo 中由 `DeckMotion` 插件按仿真时间设置船体位姿；命令格式见 `encode_plugin_command`。

## 3. ROS2 话题（slot k）

| 话题 | 类型 | 方向 |
| --- | --- | --- |
| `{ns}/fmu/in/offboard_control_mode` | `px4_msgs/OffboardControlMode` | 后端 → PX4，20 Hz 心跳 |
| `{ns}/fmu/in/trajectory_setpoint` | `px4_msgs/TrajectorySetpoint` | 后端 → PX4，NED 速度或位置，`yaw=π/2` |
| `{ns}/fmu/in/vehicle_command` | `px4_msgs/VehicleCommand` | 后端 → PX4，解锁与 OFFBOARD |
| `{ns}/fmu/out/vehicle_odometry` | `px4_msgs/VehicleOdometry` | PX4 → 后端 |
| `{ns}/fmu/out/vehicle_status` | `px4_msgs/VehicleStatus` | PX4 → 后端 |
| `/landing/slot{k}/deck_command` | `std_msgs/String` → `gz.msgs.StringMsg` | 后端 → 甲板插件 |
| `/landing/slot{k}/deck_contact` | `gz.msgs.Contacts` → `ros_gz_interfaces/Contacts` | 甲板接触传感器 → 后端 |
| `/landing/slot{k}/camera/image`、`camera_info` | `sensor_msgs` | 相机 → YOLO 节点 |
| `/landing/slot{k}/detection` | `landing_interfaces/MarkerDetection` | YOLO 节点 → 后端 |
| `/clock` | `rosgraph_msgs/Clock` | Gazebo → 全部节点（`use_sim_time`） |

`{ns}`：实例 0 为空，实例 k 为 `/px4_k`（以锁定 PX4 版本为准，可在 `BACKEND_CONFIG` 覆盖）。

## 4. 已确认与待确认事项

- **已确认（2026-11-06）**：旧 YOLO 节点输出的是预测目标的相对距离（米制三维相对位置），
  不是像素坐标。旧数据 `expert_data_lstm.json` 中前三维从约 `(5.5, 3.4, 8.9)` 收敛到
  `(0.03, 0.11, 0.5)`，与此一致。
- 符号与轴向（无人机减 marker、x/y 与世界 ENU 对齐）是从旧数据推断的：观测 x 为正时专家动作
  x 为负、高度维随下降减小。接入 YOLO 后用 `detection_source:=truth` 与 `yolo` 的同一时刻输出对比确认。
- 旧相机的分辨率、视场角和安装方向（`core/camera.py` 顶部与 `yolo_node.py` 的
  `CAMERA_TO_BODY`）。
