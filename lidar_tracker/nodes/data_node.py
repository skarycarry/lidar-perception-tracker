import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from pathlib import Path
import yaml
from lidar_tracker.core.data.kitti_loader import load_velodyne_frames

class DataNode(Node):
    def __init__(self):
        super().__init__('data_node')
        self.config = yaml.safe_load((Path(__file__).parent.parent.parent / 'config' / 'default.yaml').read_text())
        self.publish_rate = self.config['data']['publish_rate']
        self.topic = self.config['data']['output_topic']
        self.publisher_ = self.create_publisher(PointCloud2, self.topic, 10)
        self.timer = self.create_timer(1.0 / self.publish_rate, self.publish_point_cloud)
        self.frames = load_velodyne_frames(Path(self.config['data']['source']).expanduser() / 'training' / 'velodyne')

    def publish_point_cloud(self):
        try:
            frame = next(self.frames)
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = 'lidar_frame'
            points = [(point[0], point[1], point[2], point[3]) for point in frame]
            pc2_msg = point_cloud2.create_cloud(header, fields=[
                point_cloud2.PointField('x', 0, point_cloud2.PointField.FLOAT32, 1),
                point_cloud2.PointField('y', 4, point_cloud2.PointField.FLOAT32, 1),
                point_cloud2.PointField('z', 8, point_cloud2.PointField.FLOAT32, 1),
                point_cloud2.PointField('intensity', 12, point_cloud2.PointField.FLOAT32, 1),
            ], points=points)
            self.publisher_.publish(pc2_msg)
        except StopIteration:
            self.get_logger().info('No more frames to publish.')
            self.timer.cancel()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    data_node = DataNode()
    rclpy.spin(data_node)
    data_node.destroy_node()

if __name__ == '__main__':
    main()