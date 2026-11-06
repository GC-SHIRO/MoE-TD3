# 部署说明

> 本流程按目标版本编写，尚未在真实机器上完整跑通；首次部署时把实际结果补充到这里。

## 1. 宿主机

- Ubuntu 24.04（推荐）或 Windows + WSL2（Ubuntu 24.04）；
- NVIDIA 驱动、Docker、NVIDIA Container Toolkit；
- 显存需容纳 Gazebo 相机渲染和 `NUM_SLOTS` 份 YOLO。

## 2. 构建镜像

在仓库根目录：

```bash
docker compose -f docker/compose.yaml build
```

版本号集中在 `docker/Dockerfile` 顶部：`PX4_TAG`、`PX4_MSGS_BRANCH`（必须与 PX4 版本一致）、
`XRCE_AGENT_TAG`。

## 3. 准备仓库外资源

| 资源 | 放置位置 |
| --- | --- |
| marker 贴图 `marker.png`（与旧场景相同的图案） | `ros2_ws/src/landing_sim/models/landing_ship/materials/textures/` |
| YOLO 权重 | `weights/marker_yolo.pt`（已加入 `.gitignore`），或启动参数 `yolo_weights` |

只用 `detection_source:=truth` 时两者都不需要。

## 4. 启动仿真

```bash
docker compose -f docker/compose.yaml run --rm sim
```

容器内（首次进入会自动 `colcon build`）：

```bash
ros2 launch landing_sim landing.launch.py num_slots:=1 headless:=true detection_source:=truth
```

另开一个终端进入同一容器，在 `/workspace` 运行采集。先把 `collect.py` 顶部改为
`BACKEND = "ros2_gz"`，`NUM_SLOTS` 与启动参数 `num_slots` 一致：

```bash
python3 collect.py --test
```

## 5. 建议的接入顺序

1. `detection_source:=truth`，单 slot：确认 PX4 解锁、OFFBOARD、动作方向、甲板运动和接触；
2. 换成 `detection_source:=yolo`：确认相机话题、检测率和检测位置语义；
3. 多 slot 并行；
4. 每一步的结论记录到 `doc/migration_from_ros1.md` 的适配清单。

## 6. 版本记录

验证通过后，把 PX4 tag、`px4_msgs` 提交号、镜像 tag、YOLO 权重哈希写进 `doc/environment.md`。
