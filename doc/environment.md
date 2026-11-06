# 环境依赖

## 1. 离线训练、测试与 mock 后端

```bash
pip install -r requirements.txt
```

| 依赖 | 版本约束 | 原因 |
| --- | --- | --- |
| Python | ≥ 3.8 | 代码未使用 3.9 以后的运行时语法 |
| PyTorch | ≥ 1.13 | `scripts/train/common.py` 使用 `torch.load(..., weights_only=False)` |
| NumPy | ≥ 1.20 | 1.x 与 2.x 均可 |
| TensorBoard | 任意 | 仅 LSTM 版 `scripts/train_offline.py` 使用 |

已验证（2026-11-06，Windows）：

- Python 3.8.20 + PyTorch 1.13.1+cu117：`tests/` 46 个测试与 `model/tests/` 18 个测试全部通过；
- Python 3.13.5 + NumPy 2.1.3（无 PyTorch）：`tests/` 45 个通过，1 个需要 PyTorch 的测试跳过。

## 2. 仿真（Docker 镜像，见 `docker/Dockerfile`）

| 组件 | 目标版本 | 状态 |
| --- | --- | --- |
| Ubuntu | 24.04 | 未验证 |
| ROS2 | Jazzy | 未验证 |
| Gazebo | Harmonic（gz-sim8） | 未验证 |
| PX4 | `v1.16.0`（`PX4_TAG`） | 未验证 |
| px4_msgs | `release/1.16`（必须与 PX4 一致） | 未验证 |
| Micro-XRCE-DDS-Agent | `v2.4.3` | 未验证 |
| Ultralytics YOLO | 最新 | 未验证 |

Ubuntu 24.04 的系统 Python 为 3.12，PyTorch 版本不再受 ROS1 Noetic 的 Python 3.8 限制。
