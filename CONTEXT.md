# Project Context for Claude

## What this project is
A full-stack LiDAR perception and multi-object tracking pipeline built as a robotics portfolio project. Kevin is applying to perception engineering roles but wants full-stack fluency (not just algorithms).

## Stack decisions made
- **Language:** Python
- **ROS2:** Humble — nodes will wrap core algorithms. Not installed on dev machine yet, will be set up on main machine.
- **Detection:** Two modes — euclidean clustering (classical, no GPU) and PointPillars (deep learning, PyTorch). Switchable via config.
- **Tracking:** SORT-3D — Kalman filter + Hungarian algorithm on 3D axis-aligned bounding boxes
- **Visualization:** Open3D
- **Dataset:** KITTI Tracking benchmark (~15GB, lives outside the repo at a local path)
- **Evaluation:** MOTA and MOTP metrics, IoU threshold matching (0.5 default)

## Repo structure
```
lidar_tracker/
    core/          # pure Python algorithms, zero ROS imports — fully unit testable
        data/
        preprocessing/
        detection/
        tracking/
        evaluation/
    nodes/         # thin ROS2 wrappers around core/
config/
launch/
tests/
```

## What's done
- Directory structure and all __init__.py files
- .gitignore
- README.md (sparse, to be fleshed out post-implementation)
- config/default.yaml (in progress — data and tracking sections done, detection needs restructuring for dual-mode support)

## What's next
- Finish config/default.yaml — detection section needs euclidean and point_pillars subsections
- requirements.txt
- core/data/kitti_loader.py — KITTI calib parsing, label parsing, velodyne .bin loading
- core/data/synthetic.py — synthetic scene generator for testing without KITTI
- core/preprocessing/filters.py — range crop, voxel downsample
- core/preprocessing/ground_removal.py — RANSAC via Open3D
- core/detection/base.py — Detection dataclass + Detector ABC
- core/detection/euclidean.py
- core/detection/point_pillars.py
- core/detection/factory.py
- core/tracking/kalman.py
- core/tracking/track.py
- core/tracking/sort3d.py
- core/evaluation/iou3d.py
- core/evaluation/metrics.py
- lidar_tracker/nodes/ (5 ROS2 nodes)
- launch/pipeline.launch.py
- tests/

## How Kevin wants to work
Kevin writes all the code himself. Claude guides with questions and explanations, does not write code. Explain concepts, ask what the next piece should be, review what Kevin writes. Only provide hints when genuinely stuck — not solutions.

## Key concepts Kevin has learned this session
- Why core/ is separated from nodes/ (ROS2 dependency isolation)
- DBSCAN parameters: eps (neighborhood radius) and min_points
- PointPillars architecture: pillars → PointNet → pseudo-image → 2D CNN → detection head
- IoU as the matching primitive for evaluation and tracking association
- MOTA = 1 - (FN + FP + IDSW) / GT, MOTP = mean localization error of true positives
- Future extensions to discuss in interviews: CIoU loss for PointPillars training, semantic segmentation as preprocessing

## KITTI dataset notes
- Labels are in camera coordinates — need calibration matrices to convert to velodyne frame
- Velodyne .bin files: float32 Nx4 [x, y, z, intensity]
- Label format: frame, track_id, type, truncated, occluded, alpha, bbox(4), dims(h,w,l), location(x,y,z), rotation_y
- Calibration: Tr_velo_to_cam (3x4), R0_rect (3x3), P2 (3x4)
