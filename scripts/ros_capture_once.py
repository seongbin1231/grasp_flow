"""One-shot grab of /camera/rgb + /camera/depth at DOMAIN_ID=13.

Run with the ROS-bound interpreter (system python3 after sourcing
/opt/ros/humble/setup.bash). Saves PNGs to OUT_DIR and exits.
"""
import os, sys, time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image

OUT_DIR = Path("/home/robotics/Competition/YOLO_Grasp/img_dataset/live_capture")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_image(msg: Image) -> np.ndarray:
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(h, w, 3)
        if enc == "rgb8":
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
    if enc in ("mono16", "16uc1"):
        return buf.view(np.uint16).reshape(h, w)
    if enc in ("32fc1",):
        return buf.view(np.float32).reshape(h, w)
    raise ValueError(f"unsupported encoding: {msg.encoding}")


class OneShot(Node):
    def __init__(self):
        super().__init__("yolograsp_oneshot")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.rgb = None
        self.depth = None
        self.create_subscription(Image, "/camera/rgb",
                                 self._on_rgb, qos)
        self.create_subscription(Image, "/camera/depth",
                                 self._on_depth, qos)

    def _on_rgb(self, msg):
        if self.rgb is None:
            self.rgb = decode_image(msg)
            self.get_logger().info(f"rgb: {self.rgb.shape} {self.rgb.dtype} enc={msg.encoding}")

    def _on_depth(self, msg):
        if self.depth is None:
            self.depth = decode_image(msg)
            self.get_logger().info(f"depth: {self.depth.shape} {self.depth.dtype} enc={msg.encoding}  min={self.depth.min()} max={self.depth.max()}")


def main():
    os.environ.setdefault("ROS_DOMAIN_ID", "13")
    rclpy.init()
    node = OneShot()
    t0 = time.time()
    while rclpy.ok() and (node.rgb is None or node.depth is None):
        rclpy.spin_once(node, timeout_sec=0.2)
        if time.time() - t0 > 10.0:
            node.get_logger().error("timeout waiting for topics")
            break
    if node.rgb is not None:
        cv2.imwrite(str(OUT_DIR / "rgb.png"), node.rgb)
        print(f"[save] {OUT_DIR/'rgb.png'}")
    if node.depth is not None:
        d = node.depth
        if d.dtype == np.float32:
            d = (d * 1000.0).astype(np.uint16)   # save as mm
            enc = "float32→uint16 mm"
        else:
            enc = str(d.dtype)
        cv2.imwrite(str(OUT_DIR / "depth.png"), d)
        print(f"[save] {OUT_DIR/'depth.png'}  ({enc})")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
