cmake_minimum_required(VERSION 3.8)
project(lidar_perception_tracker)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection.msg"
  "msg/Track.msg"
  "msg/DetectionArray.msg"
  "msg/TrackArray.msg"
  DEPENDENCIES std_msgs
)

ament_python_install_package(lidar_tracker)

install(PROGRAMS lidar_tracker/nodes/data_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME data_node)
install(PROGRAMS lidar_tracker/nodes/preprocessing_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME preprocessing_node)
install(PROGRAMS lidar_tracker/nodes/detection_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME detection_node)
install(PROGRAMS lidar_tracker/nodes/tracking_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME tracking_node)
install(PROGRAMS lidar_tracker/nodes/visualization_node.py
  DESTINATION lib/${PROJECT_NAME} RENAME visualization_node)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
