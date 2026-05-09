from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    rviz_config = PathJoinSubstitution([
        FindPackageShare('lidar_perception_tracker'), 'rviz', 'pipeline.rviz'
    ])

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_map_base',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'base_link'],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_tf_base_lidar',
            arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'lidar_frame'],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config],
            output='screen',
        ),
        Node(
            package='lidar_perception_tracker',
            executable='data_node',
            name='data_node',
            output='screen',
        ),
        Node(
            package='lidar_perception_tracker',
            executable='preprocessing_node',
            name='preprocessing_node',
            output='screen',
        ),
        Node(
            package='lidar_perception_tracker',
            executable='detection_node',
            name='detection_node',
            output='screen',
        ),
        Node(
            package='lidar_perception_tracker',
            executable='tracking_node',
            name='tracking_node',
            output='screen',
        ),
        Node(
            package='lidar_perception_tracker',
            executable='visualization_node',
            name='visualization_node',
            output='screen',
        ),
    ])
