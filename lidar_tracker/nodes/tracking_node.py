#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.detection.base import Detection
from lidar_perception_tracker.msg import TrackArray, Track as TrackMsg, DetectionArray

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')

        config_path = Path(get_package_share_directory('lidar_perception_tracker')) / 'config' / 'default.yaml'
        self.config = yaml.safe_load(config_path.read_text())
        self.subscribe_topic = self.config['detection']['output_topic']
        self.publish_topic = self.config['tracking']['output_topic']
        self.tracker = Sort3D(
            max_age=self.config['tracking']['max_age'],
            min_hits=self.config['tracking']['min_hits'],
            match_distance=self.config['tracking']['match_distance'],
        )
        self.last_timestamp = None
        self.prev_track_count = 0
        self.drop_threshold = 3
        self.publisher_ = self.create_publisher(TrackArray, self.publish_topic, 10)
        self.subscription = self.create_subscription(DetectionArray, self.subscribe_topic, self.detection_callback, 10)
        self.get_logger().info(f'TrackingNode initialized. Subscribing to {self.subscribe_topic} and publishing to {self.publish_topic}')

    def detection_callback(self, msg: DetectionArray):
        current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_timestamp is not None:
            dt = current_time - self.last_timestamp
        else:
            dt = 0.1
        self.last_timestamp = current_time

        detections = []
        for det_msg in msg.detections:
            det = Detection(
                x=det_msg.x,
                y=det_msg.y,
                z=det_msg.z,
                width=det_msg.width,
                length=det_msg.length,
                height=det_msg.height,
                rotation_y=det_msg.rotation_y,
                confidence=det_msg.confidence if det_msg.confidence != -1.0 else None,
                object_type=det_msg.object_type if det_msg.object_type else None
            )
            detections.append(det)

        tracks = self.tracker.update(detections, dt=dt)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = msg.header.frame_id
        track_array_msg = TrackArray()
        track_array_msg.header = header
        for track in tracks:
            track_msg = TrackMsg()
            track_msg.x = track.state[0]
            track_msg.y = track.state[1]
            track_msg.z = track.state[2]
            track_msg.vx = track.state[3]
            track_msg.vy = track.state[4]
            track_msg.vz = track.state[5]
            track_msg.width = track.last_detection.width
            track_msg.length = track.last_detection.length
            track_msg.height = track.last_detection.height
            track_msg.rotation_y = track.last_detection.rotation_y
            track_msg.track_id = track.track_id
            track_array_msg.tracks.append(track_msg)

        self.publisher_.publish(track_array_msg)

        current_count = len(tracks)
        dropped = self.prev_track_count - current_count
        if dropped >= self.drop_threshold:
            self.get_logger().warn(
                f'Large track drop: {self.prev_track_count} -> {current_count} '
                f'(-{dropped}) at t={current_time:.3f}, dt={dt:.3f}s'
            )
        self.prev_track_count = current_count
        self.get_logger().info(f'Published {current_count} tracks')

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    rclpy.spin(node)
    node.destroy_node()

if __name__ == '__main__':
    main()