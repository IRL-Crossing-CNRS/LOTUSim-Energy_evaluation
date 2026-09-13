#!/usr/bin/env python3
"""Publish a local image as a ROS 2 sensor_msgs/CompressedImage message."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


DEFAULT_TOPIC = "/energy/patrol_then_inspect0/inspection/image"
DEFAULT_IMAGE = Path(__file__).resolve().parent / "test_image_2.png"


class CompressedImagePublisher(Node):
    def __init__(
        self,
        image_path: Path,
        topic: str,
        rate_hz: float,
        jpeg_quality: int,
        publish_once: bool,
    ) -> None:
        super().__init__("custom_compressed_image_publisher")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")

        success, encoded = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not success:
            raise RuntimeError(f"Could not JPEG-encode image: {image_path}")

        self._jpeg_data = encoded.tobytes()
        self._image_path = image_path
        self._topic = topic

        qos = QoSProfile(
            
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        self._publisher = self.create_publisher(CompressedImage, topic, qos)

        if not publish_once:
            self._timer = self.create_timer(1.0 / rate_hz, self.publish_image)
            self.get_logger().info(
                f"Publishing {image_path} to {topic} at {rate_hz:g} Hz"
            )



    def publish_image(self) -> None:
        message = CompressedImage()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "custom_camera"
        message.format = "jpeg"
        message.data = self._jpeg_data
        self._publisher.publish(message)
        self.get_logger().info(
            f"Published {self._image_path.name} ({len(self._jpeg_data)} JPEG bytes)"
        )


  

    def wait_for_subscriber(self, timeout_seconds: float = 5.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while self._publisher.get_subscription_count() == 0:
            if time.monotonic() >= deadline:
                self.get_logger().warning(
                    f"No subscriber discovered on {self._topic}; publishing anyway"
                )
                return
            rclpy.spin_once(self, timeout_sec=0.1)


                
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish an image as a ROS 2 CompressedImage."
    )
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE,
        help=f"image to publish (default: {DEFAULT_IMAGE})",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ROS 2 topic")
    parser.add_argument(
        "--rate", type=float, default=1.0, help="publish rate in Hz (default: 1)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="1-100",
        help="JPEG quality (default: 95)",
    )
    parser.add_argument(
        "--once", action="store_true", help="publish one message and exit"
    )
    args = parser.parse_args()

    if args.rate <= 0:
        parser.error("--rate must be greater than zero")

    args.image = args.image.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = None

    try:
        node = CompressedImagePublisher(
            image_path=args.image,
            topic=args.topic,
            rate_hz=args.rate,
            
            jpeg_quality=args.quality,
            publish_once=args.once,
            
        )

        if args.once:
            node.wait_for_subscriber()
            node.publish_image()
            rclpy.spin_once(node, timeout_sec=0.5)
        else:
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
