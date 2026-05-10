#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
import math

from lidar_perception_tracker.msg import DetectionArray, TrackArray


class VisualizationNode(Node):
    def __init__(self):
        super().__init__('visualization_node')

        self.create_subscription(DetectionArray, '/lidar_perception_tracker/streams/detections', self.detection_callback, 1)
        self.create_subscription(TrackArray, '/lidar_perception_tracker/streams/tracks', self.track_callback, 1)

        self.detection_pub = self.create_publisher(MarkerArray, '/lidar_perception_tracker/viz/detections', 10)
        self.track_pub = self.create_publisher(MarkerArray, '/lidar_perception_tracker/viz/tracks', 10)
        self.origin_pub = self.create_publisher(MarkerArray, '/lidar_perception_tracker/viz/origin', 10)
        self.create_timer(1.0, self._publish_origin)

        self.get_logger().info('VisualizationNode initialized.')

    def _box_marker(self, marker_id, frame_id, stamp, x, y, z, w, l, h, rotation_y, r, g, b):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + h / 2.0
        m.pose.orientation.z = math.sin(rotation_y / 2.0)
        m.pose.orientation.w = math.cos(rotation_y / 2.0)
        m.scale.x = l
        m.scale.y = w
        m.scale.z = h
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = 0.4
        m.lifetime = Duration(sec=0, nanosec=500_000_000)
        return m

    def _text_marker(self, marker_id, frame_id, stamp, x, y, z, h, text):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z + h + 0.3
        m.pose.orientation.w = 1.0
        m.scale.z = 0.5
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0
        m.text = text
        m.lifetime = Duration(sec=0, nanosec=500_000_000)
        return m

    def _publish_origin(self):
        stamp = self.get_clock().now().to_msg()

        sphere = Marker()
        sphere.header.frame_id = 'base_link'
        sphere.header.stamp = stamp
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 1.0
        sphere.scale.y = 1.0
        sphere.scale.z = 0.5
        sphere.color.r = 1.0
        sphere.color.g = 0.5
        sphere.color.b = 0.0
        sphere.color.a = 1.0

        label = Marker()
        label.header.frame_id = 'base_link'
        label.header.stamp = stamp
        label.id = 1
        label.type = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.z = 1.5
        label.pose.orientation.w = 1.0
        label.scale.z = 0.6
        label.color.r = 1.0
        label.color.g = 1.0
        label.color.b = 1.0
        label.color.a = 1.0
        label.text = 'sensor'

        markers = MarkerArray()
        markers.markers = [sphere, label]
        self.origin_pub.publish(markers)

    def detection_callback(self, msg: DetectionArray):
        markers = MarkerArray()
        for i, det in enumerate(msg.detections):
            markers.markers.append(self._box_marker(
                i, msg.header.frame_id, msg.header.stamp,
                det.x, det.y, det.z, det.width, det.length, det.height, det.rotation_y,
                0.0, 1.0, 0.0,
            ))
        self.detection_pub.publish(markers)

    def track_callback(self, msg: TrackArray):
        markers = MarkerArray()
        for i, track in enumerate(msg.tracks):
            markers.markers.append(self._box_marker(
                i * 2, msg.header.frame_id, msg.header.stamp,
                track.x, track.y, track.z, track.width, track.length, track.height, track.rotation_y,
                0.0, 0.5, 1.0,
            ))
            markers.markers.append(self._text_marker(
                i * 2 + 1, msg.header.frame_id, msg.header.stamp,
                track.x, track.y, track.z, track.height,
                str(track.track_id),
            ))
        self.track_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = VisualizationNode()
    rclpy.spin(node)
    node.destroy_node()


if __name__ == '__main__':
    main()
