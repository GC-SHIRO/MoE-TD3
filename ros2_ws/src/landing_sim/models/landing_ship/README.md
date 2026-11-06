# landing_ship 资源

船模型本身由 `scripts/generate_world.py` 内联生成，这个目录只放贴图：

```text
materials/textures/marker.png
```

`marker.png` 需要使用与旧场景相同的 marker 图案（YOLO 权重是在这个图案上训练的），
从旧工具链的 Gazebo 模型目录拷贝过来。缺少贴图时 marker 显示为白色平面，YOLO 无法检测，
但 `detection_source="truth"` 的伪检测不受影响。

启动文件会把本包 `share/landing_sim/models` 加入 `GZ_SIM_RESOURCE_PATH`。
