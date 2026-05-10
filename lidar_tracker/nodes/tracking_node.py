#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import message_filters

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from lidar_tracker.core.tracking.sort3d import Sort3D
from lidar_tracker.core.detection.base import Detection
from lidar_tracker.core.preprocessing.ego_motion import EgoMotionEstimator
from lidar_perception_tracker.msg import TrackArray, Track as TrackMsg, DetectionArray


class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')

        config_path = Path(get_package_share_directory('lidar_perception_tracker')) / 'config' / 'default.yaml'
        self.config = yaml.safe_load(config_path.read_text())
        self.publish_topic = self.config['tracking']['output_topic']

        mode = self.config['detection']['mode']
        trk_cfg = self.config['tracking'][mode]
        self.tracker = Sort3D(
            max_age=trk_cfg['max_age'],
            min_hits=trk_cfg['min_hits'],
            match_distance=trk_cfg['match_distance'],
        )
        self.ego_estimator = EgoMotionEstimator()
        self.last_timestamp = None
        self.prev_track_count = 0
        self.drop_threshold = 3

        self.publisher_ = self.create_publisher(TrackArray, self.publish_topic, 10)

        # Subscribe to raw point cloud (used for ego-motion only) and detections,
        # synchronised so we always pair the cloud that produced the detections.
        raw_cloud_topic = self.config['data']['output_topic']
        det_topic       = self.config['detection']['output_topic']

        det_sub   = message_filters.Subscriber(self, DetectionArray, det_topic)
        cloud_sub = message_filters.Subscriber(self, PointCloud2, raw_cloud_topic)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [det_sub, cloud_sub], queue_size=2, slop=0.1,
        )
        self._sync.registerCallback(self._callback)

        self.get_logger().info(
            f'TrackingNode initialized. Subscribing to {det_topic} + {raw_cloud_topic}, '
            f'publishing to {self.publish_topic}'
        )

    def _callback(self, det_msg: DetectionArray, cloud_msg: PointCloud2):
        current_time = det_msg.header.stamp.sec + det_msg.header.stamp.nanosec * 1e-9
        dt = (current_time - self.last_timestamp) if self.last_timestamp is not None else 0.1
        self.last_timestamp = current_time

        # Update cumulative ego-motion from raw point cloud (fast path, no Python loop)
        n_pts = cloud_msg.width * cloud_msg.height
        if n_pts > 0:
            pts = np.frombuffer(cloud_msg.data, dtype=np.float32).reshape(-1, 4)
            self.ego_estimator.update(pts)

        # Sensor-frame detections → world frame before handing to tracker
        sensor_dets = [
            Detection(
                x=d.x, y=d.y, z=d.z,
                width=d.width, length=d.length, height=d.height,
                rotation_y=d.rotation_y,
                confidence=d.confidence if d.confidence != -1.0 else None,
                object_type=d.object_type if d.object_type else None,
            )
            for d in det_msg.detections
        ]
        world_dets = []
        for det in sensor_dets:
            pos_w = self.ego_estimator.sensor_to_world(det.position)
            world_dets.append(Detection(
                x=float(pos_w[0]), y=float(pos_w[1]), z=float(pos_w[2]),
                width=det.width, length=det.length, height=det.height,
                rotation_y=det.rotation_y, confidence=det.confidence,
                object_type=det.object_type,
            ))

        tracks = self.tracker.update(world_dets, dt=dt)

        # Convert world-frame track positions back to current sensor frame for publishing
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = det_msg.header.frame_id
        track_array_msg = TrackArray()
        track_array_msg.header = header
        for track in tracks:
            pos_s = self.ego_estimator.world_to_sensor(track.state[:3])
            vel_s = self.ego_estimator.R_ws.T @ track.state[3:6]
            tm = TrackMsg()
            tm.x          = float(pos_s[0])
            tm.y          = float(pos_s[1])
            tm.z          = float(pos_s[2])
            tm.vx         = float(vel_s[0])
            tm.vy         = float(vel_s[1])
            tm.vz         = float(vel_s[2])
            tm.width      = track.last_detection.width
            tm.length     = track.last_detection.length
            tm.height     = track.last_detection.height
            tm.rotation_y = track.last_detection.rotation_y
            tm.track_id   = track.track_id
            track_array_msg.tracks.append(tm)

        self.publisher_.publish(track_array_msg)

        current_count = len(tracks)
        dropped = self.prev_track_count - current_count
        if dropped >= self.drop_threshold:
            self.get_logger().warn(
                f'Large track drop: {self.prev_track_count} → {current_count} '
                f'(-{dropped}) at t={current_time:.3f}'
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
