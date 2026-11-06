# 项目开发约定

本文件适用于整个仓库。

## 总体原则

- 优先使用简单、直接、容易现场修改的实现，不为暂时不存在的需求增加复杂抽象。
- 唯一的抽象层是 `backends/base.py`（mock 与真实后端可互换）；不再引入其他抽象。
- 修改应聚焦当前任务；旧实现已经被替代时可以删除，需要追溯时使用 Git 历史。
- 除非用户明确要求，不修改与当前任务无关的训练、评估或仿真代码。

## 参数与命令行

- 经常调整的参数集中放在脚本顶部，并按用途分组。
- 默认通过直接修改脚本顶部参数完成配置；`collect.py` 只保留 `--test` 一个开关。
- `--test` 的数据必须写入独立文件，不得污染正式数据。

## 注释与输出

- 新增或修改的代码注释、docstring、运行提示和异常信息使用中文。
- 变量、函数和类名使用语义清晰的英文。
- 注释解释设计原因、坐标语义和数据约束，避免重复描述代码表面行为。

## 数据约定

- `model/td3_offline.py` 与 `model/moe_td3.py` 是离线训练数据接口的标准；除非用户明确要求，否则不要修改。
- 每步必须包含 `observation`、`action`、`reward`、`next_observation`、`done`；MoE 还需要 `expert.phase`。
- `state_dim=10`、`action_dim=3`、`max_action=1.0`；LSTM 版 `seq_len=8`，MoE 版 `seq_len=32`。
- 采集写原始 observation，不提前归一化；归一化由训练入口完成。
- 每行保存一个完整 episode，不删除中间 transition，必须满足 `step[i].next_observation == step[i+1].observation`。
- 只把完整成功 episode 写入训练文件；所有尝试写入 raw 文件。
- 失检时位置保持上一有效检测，速度、加速度和置信度置零；不使用全局真值冒充策略输入。
- 非近地失检进入 `SEARCH`：保持上一实际 action 的水平分量，z 为正（上升）。
- 近地失检不进入 `SEARCH`；重新检测到 marker 后立即退出 `SEARCH`。
- 坐标换算只放在 `core/frames.py`；甲板运动公式在 `core/ship_motion.py` 与
  `ros2_ws/src/landing_sim/src/DeckMotion.cc` 中必须同步修改。

## 测试约定

- 默认只运行不依赖 ROS、Gazebo、PX4 和 YOLO 的离线测试。
- 不要由代理自动启动仿真、PX4、Gazebo、ROS 或 YOLO；仿真测试由用户手动执行。
- 完成修改后运行 `tests/` 与 `model/tests/` 的离线测试、`python -m compileall` 和 `git diff --check`。

## 文档同步

- 采集入口、参数、输出文件或数据语义变化时，同步更新 `README.md` 与 `doc/interfaces.md`。
- 依赖或版本变化时，同步更新 `requirements*.txt`、`docker/Dockerfile` 与 `doc/environment.md`。
- 接入真实仿真时确认的事项，在 `doc/migration_from_ros1.md` 的适配清单中勾选并记录结论。
