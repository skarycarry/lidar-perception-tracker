#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Header

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
import numpy as np
from lidar_tracker.core.detection.factory import create_detector
from lidar_perception_tracker.msg import DetectionArray, Detection as DetectionMsg

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        config_path = Path(get_package_share_directory('lidar_perception_tracker')) / 'config' / 'default.yaml'
        self.config = yaml.safe_load(config_path.read_text())
        self.publish_topic = self.config['detection']['output_topic']
        self.detector = create_detector(config_path)
        # PointPillars handles its own preprocessing internally; subscribe to raw cloud
        if self.detector.needs_external_preprocessing:
            self.subscribe_topic = self.config['preprocessing']['output_topic']
        else:
            self.subscribe_topic = self.config['data']['output_topic']
        self.publisher_ = self.create_publisher(DetectionArray, self.publish_topic, 10)
        self.subscription = self.create_subscription(PointCloud2, self.subscribe_topic, self.point_cloud_callback, 1)
        self.get_logger().info(f'DetectionNode initialized. Subscribing to {self.subscribe_topic} and publishing to {self.publish_topic}')

    def point_cloud_callback(self, msg: PointCloud2):
        if msg.width * msg.height == 0:
            return
        points = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        
        detections = self.detector.detect(points)
        detections = [
            DetectionMsg(
                x=d.x, y=d.y, z=d.z,
                height=d.height, width=d.width, length=d.length,
                rotation_y=d.rotation_y,
                confidence=d.confidence if d.confidence is not None else -1.0,
                object_type=d.object_type if d.object_type is not None else '',
            )
            for d in detections
        ]
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id
        out_msg = DetectionArray()
        out_msg.header = header
        out_msg.detections = detections
        self.publisher_.publish(out_msg)
        self.get_logger().info(f'Published {len(detections)} detections')

def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()