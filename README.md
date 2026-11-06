# MoE-TD3：动态船载降落

无人机在带升沉、横摇、纵摇的船甲板上视觉降落。全局真值专家采集数据，
阶段路由 MoE-TD3（Causal Transformer + 五个阶段专家）离线训练，在仿真中闭环评估。

仿真栈：ROS2 Jazzy + Gazebo Harmonic + PX4 SITL（uXRCE-DDS）。
本仓库由旧仓库 `Landing_New`（ROS1 + Gazebo Classic + MAVROS + XTDrone）迁移而来，
迁移说明和适配清单见 [doc/migration_from_ros1.md](doc/migration_from_ros1.md)。

## 当前状态

| 部分 | 状态 |
| --- | --- |
| `core/`、`backends/mock.py`、采集、评估、训练 | 完成，离线测试全部通过 |
| mock 全链路（采集 → 校验 → 训练 → 闭环评估） | 完成，有自动测试 |
| `backends/ros2_gz.py`、`ros2_ws/`、`docker/` | 按接口契约写完，**尚未在真实仿真中运行** |

## 目录结构

```text
core/            纯 Python 核心，不依赖 ROS：专家、observation、坐标换算、甲板运动、相机、判定、数据
backends/        仿真后端：mock（运动学）与 ros2_gz（真实仿真）
model/           TD3-BC/LSTM 与 MoE-TD3 模型（离线数据接口标准）
scripts/train/   MoE 三阶段离线训练
collect.py       专家采集入口
evaluate.py      闭环评估入口
ros2_ws/src/     landing_interfaces（消息）、landing_sim（世界/插件/启动）、landing_perception（YOLO）
docker/          仿真镜像
tests/           离线测试
doc/             接口契约、部署、环境依赖、迁移说明
```

## 快速开始（无需仿真）

```bash
pip install -r requirements.txt
python collect.py --test                       # mock 后端冒烟测试
python -m unittest discover -s tests -t . -v
python -m unittest discover -s model/tests -v
```

`collect.py` 顶部 `BACKEND = "mock"` 时数据写入 `data/expert/mock/`，只用于链路测试，
不能作为正式训练数据。

## 正式流程

1. 按 [doc/deployment.md](doc/deployment.md) 构建镜像并启动仿真；
2. `collect.py` 顶部改 `BACKEND = "ros2_gz"`，先 `--test`，再正式采集；
3. `python -m core.validate` 前修改 `core/validate.py` 顶部 `DATA_PATH` 并检查数据；
4. 依次运行 `scripts/train/` 的三个阶段（见 [训练说明](scripts/train/README.md)）；
5. `evaluate.py` 顶部设置 `POLICY` 和 `CHECKPOINT`，运行闭环评估。

## 依赖

见 [doc/environment.md](doc/environment.md)。
