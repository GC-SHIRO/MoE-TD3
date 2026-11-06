#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO marker 检测节点：图像 → MarkerDetection。

输出语义（已确认，与 core/camera.py 和 doc/interfaces.md 一致）：预测目标的相对距离，
即无人机相对 marker 的三维位置，米，轴向与机体/世界 ENU 对齐（偏航固定为 0）。
距离由检测框尺寸和 marker 实际边长按针孔模型估计；不做姿态补偿。
距离估计方法与 CAMERA_TO_BODY 轴向仍需与旧 yolov11_ros 节点核对（见适配清单）。
"""

import math

import numpy as np
import rclpy
from cv_bridge import CvBridge
from landing_interfaces.msg import MarkerDetection
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


# ==================== 检测参数 ====================
CONFIDENCE_THRESHOLD = 0.25
MARKER_CLASS_ID = 0
MARKER_SIZE = 1.0                    # 米，需与 landing_sim/scripts/generate_world.py 一致
# 下视相机光学系（x 右、y 下、z 沿光轴）→ 机体 FLU。默认假设图像上方朝机头。
CAMERA_TO_BODY = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


class YoloMarkerNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_marker_node")
        slot = int(self.declare_parameter("slot", 0).value)
        weights = str(self.declare_parameter("weights", "").value)
        device = str(self.declare_parameter("device", "").value)
        if not weights:
            raise RuntimeError("必须通过参数 weights 指定 YOLO 权重路径")
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.device = device or None
        self.bridge = CvBridge()
        self.intrinsics = None
        self.create_subscription(CameraInfo, f"/landing/slot{slot}/camera/camera_info", self._on_info, 10)
        self.create_subscription(Image, f"/landing/slot{slot}/camera/image", self._on_image, 1)
        self.publisher = self.create_publisher(MarkerDetection, f"/landing/slot{slot}/detection", 10)
        self.get_logger().info(f"YOLO 检测节点已启动: slot={slot}, weights={weights}")

    def _on_info(self, msg: CameraInfo) -> None:
        k = msg.k
        self.intrinsics = (float(k[0]), float(k[4]), float(k[2]), float(k[5]))

    def _on_image(self, msg: Image) -> None:
        output = MarkerDetection()
        output.header = msg.header
        output.visible = False
        output.confidence = 0.0
        if self.intrinsics is None:
            self.publisher.publish(output)
            return
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        result = self.model.predict(image, conf=CONFIDENCE_THRESHOLD, device=self.device, verbose=False)[0]
        boxes = [box for box in result.boxes if int(box.cls.item()) == MARKER_CLASS_ID]
        if boxes:
            best = max(boxes, key=lambda box: float(box.conf.item()))
            x1, y1, x2, y2 = (float(v) for v in best.xyxy[0].tolist())
            fx, fy, cx, cy = self.intrinsics
            size_px = max(x2 - x1, y2 - y1)
            if size_px > 1.0:
                distance = fx * MARKER_SIZE / size_px
                u, v = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
                ray = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
                marker_camera = ray / np.linalg.norm(ray) * distance
                relative = -(CAMERA_TO_BODY @ marker_camera)       # 无人机相对 marker
                if np.all(np.isfinite(relative)) and math.isfinite(distance):
                    output.visible = True
                    output.position.x, output.position.y, output.position.z = (float(v) for v in relative)
                    output.confidence = float(best.conf.item())
        # 空检测也要发布，采集端据此区分“看不到 marker”和“没有视觉帧”。
        self.publisher.publish(output)


def main() -> None:
    rclpy.init()
    node = YoloMarkerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
