#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
import numpy as np
from lidar_tracker.core.preprocessing.filters import range_crop, voxel_downsample
from lidar_tracker.core.preprocessing.ground_filter import remove_ground

class PreprocessingNode(Node):
    def __init__(self):
        super().__init__('preprocessing_node')
        config_path = Path(get_package_share_directory('lidar_perception_tracker')) / 'config' / 'default.yaml'
        self.config = yaml.safe_load(config_path.read_text())
        self.subscribe_topic = self.config['data']['output_topic']
        self.publish_topic = self.config['preprocessing']['output_topic']
        self.min_distance = self.config['preprocessing']['min_distance']
        self.max_distance = self.config['preprocessing']['max_distance']
        self.ground_threshold = self.config['preprocessing']['ground_threshold']
        self.voxel_size = self.config['preprocessing']['voxel_size']
        self.publisher_ = self.create_publisher(PointCloud2, self.publish_topic, 10)
        self.subscription = self.create_subscription(PointCloud2, self.subscribe_topic, self.point_cloud_callback, 10)
        self.get_logger().info(f'PreprocessingNode initialized. Subscribing to {self.subscribe_topic} and publishing to {self.publish_topic}')

    def point_cloud_callback(self, msg: PointCloud2):
        pts_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if not pts_list:
            return
        points = np.array([[p[0], p[1], p[2], p[3]] for p in pts_list], dtype=np.float32)
        
        points = range_crop(points, self.min_distance, self.max_distance)
        points = remove_ground(points, self.ground_threshold)
        points = voxel_downsample(points, self.voxel_size)
        
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id
        out_msg = point_cloud2.create_cloud(header, fields=[
            point_cloud2.PointField(name='x', offset=0, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='y', offset=4, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='z', offset=8, datatype=point_cloud2.PointField.FLOAT32, count=1),
            point_cloud2.PointField(name='intensity', offset=12, datatype=point_cloud2.PointField.FLOAT32, count=1),
        ], points=points)
        
        self.publisher_.publish(out_msg)
        self.get_logger().info(f'Published processed point cloud with {points.shape[0]} points')

def main(args=None):
    rclpy.init(args=args)
    node = PreprocessingNode()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()