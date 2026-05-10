#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import numpy as np

from pathlib import Path
import yaml
from ament_index_python.packages import get_package_share_directory
from lidar_tracker.core.data.kitti_loader import load_lidar_frames


TRANSITION_SECS = 3  # seconds to show the "moving to next scenario" banner


class DataNode(Node):
    def __init__(self):
        super().__init__('data_node')
        config_path = Path(get_package_share_directory('lidar_perception_tracker')) / 'config' / 'default.yaml'
        self.config = yaml.safe_load(config_path.read_text())
        self.publish_rate = self.config['data']['publish_rate']

        self.declare_parameter('dataset_path', self.config['data']['source'])
        dataset_path = self.get_parameter('dataset_path').get_parameter_value().string_value
        velodyne_dir = Path(dataset_path).expanduser() / 'training' / 'velodyne'

        self.sequences = sorted([d for d in velodyne_dir.iterdir() if d.is_dir()])
        if not self.sequences:
            self.get_logger().error(f'No sequence directories found in {velodyne_dir}')
            return

        self.seq_idx = 0
        self.frames = None
        self.frame_count = 0
        self.transition_ticks = 0  # timer ticks remaining in transition pause

        self.cloud_pub  = self.create_publisher(PointCloud2,  self.config['data']['output_topic'], 1)
        self.status_pub = self.create_publisher(MarkerArray, '/lidar_perception_tracker/viz/status', 1)

        self._start_next_sequence()
        self.timer = self.create_timer(1.0 / self.publish_rate, self._tick)

    # ── Sequence management ───────────────────────────────────────────────────

    def _start_next_sequence(self):
        seq_dir = self.sequences[self.seq_idx]
        self.frames = load_lidar_frames(seq_dir)
        self.frame_count = 0
        self.get_logger().info(f'Starting sequence {seq_dir.name}')

    def _begin_transition(self):
        self.get_logger().info(
            f'Sequence {self.sequences[self.seq_idx].name} complete '
            f'— published {self.frame_count} frames.'
        )
        self.seq_idx += 1
        self.transition_ticks = int(self.publish_rate * TRANSITION_SECS)

    # ── Timer callback ────────────────────────────────────────────────────────

    def _tick(self):
        if self.transition_ticks > 0:
            self._publish_banner()
            self.transition_ticks -= 1
            if self.transition_ticks == 0:
                if self.seq_idx >= len(self.sequences):
                    self.get_logger().info('All sequences complete.')
                    self.timer.cancel()
                    # Brief delay so the last banner finishes rendering, then exit
                    self.create_timer(1.0, rclpy.shutdown)
                else:
                    self._start_next_sequence()
            return

        try:
            frame = next(self.frames)
            self._publish_cloud(frame)
            self.frame_count += 1
        except StopIteration:
            self._begin_transition()

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_cloud(self, frame: np.ndarray):
        n = len(frame)
        msg = PointCloud2()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lidar_frame'
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField(name='x',         offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y',         offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z',         offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = 16 * n
        msg.data = frame.astype(np.float32).tobytes()
        msg.is_dense = True
        self.cloud_pub.publish(msg)

    def _publish_banner(self):
        next_name = (
            self.sequences[self.seq_idx].name
            if self.seq_idx < len(self.sequences)
            else 'done'
        )
        current_name = self.sequences[self.seq_idx - 1].name

        m = Marker()
        m.header.frame_id = 'lidar_frame'
        m.header.stamp = self.get_clock().now().to_msg()
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = 15.0
        m.pose.position.y = 0.0
        m.pose.position.z = 3.0
        m.pose.orientation.w = 1.0
        m.scale.z = 1.2
        m.color.r = 1.0
        m.color.g = 0.85
        m.color.b = 0.0
        m.color.a = 1.0
        m.text = f'Sequence {current_name} complete\nMoving to scenario {next_name}...'
        m.lifetime.nanosec = int(1.5e9 / self.publish_rate)

        markers = MarkerArray()
        markers.markers = [m]
        self.status_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = DataNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
