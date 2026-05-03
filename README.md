# Lidar Tracker
This repo is made to track objects over time using lidar frames. Vision may come into this at some point but initially this focuses on lidar. The challenge is tracking objects frame to frame with the correct transforms. This is tested on the KITTI dataset

## Implementation
This repo has 3 major components: data ingestion, object detection, and object tracking. Data ingestion takes our sensor inputs (from real sensors or from replayed data) and feeds it to our object detection. Our object detecion then outputs the location and name of the objects. And our tracker takes that in and correlates objects across frames.

## Dependencies
Python 3.9+
ROS2
Open3D
SciPy
NumPy
PyTorch

## How to Run
For full system functionality, run from the given launchfile pointed to the dataset location

## Eval Metrics
Using  MOTA and MOTP, we track the multiple objects over time (need more detail once these are implemented and I understand these better)

## Future Extensions
- Utilize an IoU-based loss function (CIoU) when training the PointPillars detection head, replacing the standard L1 regression loss. This directly optimizes for box overlap quality rather than coordinate error, which should improve MOTA scores.
- Add semantic segmentation as a preprocessing step to remove non-object points (road, sidewalk, buildings) before detection, rather than only removing the ground plane. KITTI does not provide point cloud segmentation labels so this would require a pretrained model.