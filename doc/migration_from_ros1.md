# 从 ROS1 工具链迁移

来源：旧仓库 `Landing_New`（main 分支，迁移方案见其 `doc/ros2_migration_plan.md`）。

## 1. 迁移对照

| 旧文件 | 新位置 | 处理 |
| --- | --- | --- |
| `Sampling/global_expert.py` | `core/expert.py` | 原样 |
| `Sampling/validate_expert_data.py` | `core/validate.py` | 只改默认数据路径 |
| `Sampling/collect_global_expert.py` | `core/observation.py`、`core/episode.py`、`core/dataset.py`、`core/rollout.py`、`collect.py` | 拆分；逻辑不变 |
| `Sampling/collect_parallel.py` | `collect.py`（`NUM_SLOTS`）、`core/episode.py`（`EarlyAbort`） | 多实例改为单世界多 slot |
| `Simulation/ship_motion.py` | `core/ship_motion.py` + `DeckMotion.cc` | 重写；新增升沉/横摇/纵摇 |
| `Simulation/env/env_base.py` | `backends/ros2_gz.py` | 按接口契约重写 |
| `model/*`、`scripts/train/*`、`scripts/train_offline.py`、`scripts/check_data_dist.py` | 同名路径 | 原样，只改默认数据路径和实验目录 |
| `Sampling/tests/*`、`model/tests/*` | `tests/`、`model/tests/` | 改导入路径 |
| 其余旧环境、旧评估、`.bak`、旧数据 | — | 不迁移 |

与迁移方案的差异：

- 训练脚本保留在 `scripts/train/`，不移到 `train/`，避免修改导入；
- 没有单独的 `landing_control` 包和真值检测节点：offboard 控制写在 `backends/ros2_gz.py`，
  真值伪检测由后端直接调用 `core/camera.py`；
- 没有 `DeckState` 消息：甲板真值由 Python 模型按仿真时间计算，与插件同一公式。

## 2. 新增内容

- 场景标签 = 运动类别 × 海况（0–4）；
- 触地指标：相对法向速度（取接触前一步）、甲板倾角；
- mock 后端与 mock 全链路测试；
- 闭环评估入口 `evaluate.py`（特权专家 / MoE 存档）。

mock 上的特权专家评估已经显示：海况 4 下触地法向速度从约 0.08 m/s 升到 0.2–0.66 m/s，
专家不补偿升沉，这可以作为论文的动机实验之一（需在真实仿真中复现）。

## 3. 适配清单（接入真实仿真时逐项确认）

**PX4 与控制**

- [ ] `px4_msgs` 分支与 PX4 版本一致；`fmu/out` 话题名（是否带 `_v1` 等后缀）
- [ ] 实例 0 与实例 k 的话题前缀（`px4_namespace_*`）
- [ ] `VehicleCommand` 的 `target_system` 是否为实例号 + 1
- [ ] OFFBOARD 进入与解锁成功；心跳不中断
- [ ] 动作 `(1,0,0)` 沿 Gazebo 世界 +x 飞行，偏航保持朝 +x
- [ ] `VehicleOdometry.velocity_frame` 为 NED；PX4 本地原点等于出生点
- [ ] 0.1 s 仿真时间步长的实际抖动；real-time factor > 1 时是否稳定

**甲板与接触**

- [ ] `DeckMotion` 插件加载成功，位姿与 `core/ship_motion.py` 在同一时刻一致（差 < 5 cm）
- [ ] `SetWorldPoseCmd` 驱动的船体与无人机接触是否稳定，触地有无穿透或弹飞
- [ ] 接触传感器话题经 bridge 到达后端；只有无人机触地时才有接触
- [ ] 复位时船体回到原点、甲板保持 t=0 位姿，无人机能从甲板上抬升离开

**相机与检测**

- [ ] PX4 机型名 `gz_x500_mono_cam_down` 与 `PX4_SYS_AUTOSTART` 编号
- [ ] 相机 gz 话题名（`camera_topic_template`）
- [ ] 相机分辨率、视场角、安装方向与旧相机一致，同步 `core/camera.py` 与 `CAMERA_TO_BODY`
- [x] 检测位置语义：YOLO 输出预测目标的相对距离（米制），已于 2026-11-06 确认
- [ ] 检测位置的符号与轴向：同一时刻 truth 与 yolo 输出一致（见 `doc/interfaces.md` 第 4 节）
- [ ] marker 贴图与旧场景一致；YOLO 在新渲染画面下的检测率
- [ ] 距离估计方法与旧节点一致（当前按检测框尺寸估计）

**数据**

- [ ] 新旧数据分布对比（动作、回合长度、失检率、SEARCH 比例），决定是否混用
- [ ] 用旧 YOLO 日志标定 `core/camera.py` 的噪声与漏检参数
