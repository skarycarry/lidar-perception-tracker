# LiDAR Perception Tracker

A full-stack 3D multi-object tracking pipeline for autonomous driving, evaluated on the KITTI Tracking benchmark. Implements three detection backends (Euclidean clustering, PointPillars, PV-RCNN), a Kalman-filter SORT-3D tracker with ego-motion compensation, and a comprehensive evaluation suite (HOTA, MOTA, IDF1, MT/ML, per-class, range-stratified).

---

## Pipeline Architecture

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  Input: Velodyne .bin frames (KITTI format)                      │
  └────────────────────────┬─────────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  Ego-Motion Estimator   │  ICP-based scan-to-scan
              │  (EgoMotionEstimator)   │  registration → R_ws, t_ws
              └────────────┬────────────┘
                           │  sensor → world transform
          ┌────────────────▼──────────────────────┐
          │            Object Detection            │
          │  ┌──────────────────────────────────┐  │
          │  │  mode: euclidean                 │  │  DBSCAN clustering on
          │  │  (EuclideanDetector)             │  │  ground-removed points
          │  ├──────────────────────────────────┤  │
          │  │  mode: point_pillars             │  │  Pillar VFE → BEV backbone
          │  │  (PointPillarsDetector)          │  │  → anchor-based head
          │  ├──────────────────────────────────┤  │
          │  │  mode: pvrcnn                    │  │  Voxel feature extraction
          │  │  (OpenPCDetDetector)             │  │  → set abstraction → RCNN
          │  ├──────────────────────────────────┤  │
          │  │  mode: fusion                    │  │  PP + PV-RCNN with
          │  │  (FusionDetector)                │  │  per-class NMS fusion
          │  └──────────────────────────────────┘  │
          └────────────────┬──────────────────────┘
                           │  [Detection] in world frame
          ┌────────────────▼──────────────────────┐
          │       SORT-3D Tracker                 │
          │  predict  →  associate  →  update     │
          │  (Kalman filter + Hungarian algorithm) │
          │  3D centre-distance matching           │
          └────────────────┬──────────────────────┘
                           │  world → sensor transform
          ┌────────────────▼──────────────────────┐
          │       Evaluation / ROS2 output        │
          │  HOTA  MOTA  IDF1  MT/ML  per-class   │
          │  range-stratified  |  RViz tracks      │
          └───────────────────────────────────────┘
```

The `lidar_tracker/core/` layer is pure Python with zero ROS imports and is fully unit-testable. ROS2 nodes in `nodes/` are thin wrappers.

---

## Benchmark Results

Averaged across all 21 KITTI training sequences (Car + Pedestrian + Cyclist).

| Detector | HOTA | DetA | AssA | IDF1 | MOTA | MT% | ML% | IDS | FP | FN | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Euclidean | 0.2019 | 0.0668 | 0.6709 | 0.0992 | −5.990 | 17.6% | 43.5% | 48 | 9634 | 1070 | 113.8 |
| PointPillars | 0.4591 | 0.3143 | 0.7002 | 0.3734 | +0.077 | 15.5% | 35.2% | 33 | 578 | 1199 | **34.8** |
| PV-RCNN | **0.5732** | **0.4020** | **0.8457** | **0.5085** | +0.047 | **32.1%** | **30.7%** | 73 | 1050 | **770** | 139.4 |
| Fusion (PP+PV, NMS 1.0m) | 0.5564 | 0.3927 | 0.8203 | 0.4871 | +0.003 | 32.2% | 30.9% | 73 | 1125 | 767 | 174.2 |

**Key observations:**

- **Euclidean clustering** has near-zero DetA (0.07) because DBSCAN produces thousands of false positives from noise and building edges — its strong AssA (0.67) shows the tracker itself works well when given clean detections; the detector is the bottleneck. Surprisingly slow at 113.8 ms/frame due to Open3D RANSAC ground removal dominating runtime.
- **PointPillars** is competitive (HOTA 0.46) at only 34.8 ms/frame — 4× faster than PV-RCNN. Precision is excellent (only 578 FP) but it misses ~430 more objects than PV-RCNN, especially at range.
- **PV-RCNN** achieves the best HOTA and AssA. The slightly positive MOTA (+0.047) is notable: GT labels only cover objects meeting strict visibility/size/range criteria, but PV-RCNN correctly detects real objects outside those criteria (counted as FP against GT), which would push MOTA negative — the positive score means HOTA's approach to handling detection quality is more meaningful here.
- **Fusion (PP+PV-RCNN with NMS)** does not improve over PV-RCNN alone — the two models fail on largely the same objects (FN overlap ~95%), so fusion only adds PP's false positives without recovering meaningful misses. Ensembling helps only when failure modes are complementary; these two models are not.

### Range-stratified breakdown

| Detector | Range | HOTA | DetA | FN | FP | GT Dets |
|---|---|---|---|---|---|---|
| Euclidean | 0–20 m | 0.3209 | 0.1539 | 416 | 1914 | 828 |
| Euclidean | 20–40 m | 0.2398 | 0.0803 | 342 | 3935 | 739 |
| Euclidean | 40 m+ | 0.1451 | 0.0420 | 312 | 1151 | 370 |
| PointPillars | 0–20 m | 0.6115 | 0.4804 | 396 | 269 | 828 |
| PointPillars | 20–40 m | 0.4912 | 0.3574 | 417 | 172 | 739 |
| PointPillars | 40 m+ | 0.2124 | 0.1042 | 332 | 9 | 370 |
| PV-RCNN | 0–20 m | 0.6318 | 0.4733 | 280 | 427 | 828 |
| PV-RCNN | 20–40 m | 0.6264 | 0.4679 | 224 | 379 | 739 |
| PV-RCNN | 40 m+ | **0.5088** | **0.3718** | 214 | 46 | 370 |

**Key findings:**

- **Close range (0–20 m):** PointPillars and PV-RCNN are nearly equivalent in HOTA (0.61 vs 0.63). PP actually has *fewer* FPs (269 vs 427), making it the better precision choice at short range.
- **Medium range (20–40 m):** PV-RCNN pulls ahead significantly (0.63 vs 0.49). PP's DetA drops from 0.48 to 0.36.
- **Long range (40 m+):** The gap is decisive — PV-RCNN maintains 0.51 HOTA while PP collapses to 0.21. PV-RCNN's voxel feature extraction and RCNN refinement stage preserve enough signal from sparse distant point clouds; PP's pillar-based encoding degrades sharply. PP's FP count at 40 m+ drops to nearly zero (9), meaning it simply stops detecting rather than hallucinating.
- **Euclidean** degrades consistently at all ranges due to uncontrolled FPs from building edges and ground-removal artifacts, though FN counts are similar to the learned methods.

> Tracker hyperparameters (match\_distance, max\_age, min\_hits) were tuned per-detector using Bayesian optimisation (Optuna, 1000 trials) with HOTA as the objective.

---

## Setup

### Requirements

- Ubuntu 24.04, Python 3.12
- CUDA 12+ (for PointPillars and PV-RCNN)
- ROS2 Jazzy (for live pipeline only — not required for offline eval/benchmark)

### Install

```bash
git clone <this-repo>
cd lidar-perception-tracker

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

#### PV-RCNN / SECOND / Voxel R-CNN (OpenPCDet models)

```bash
pip install spconv-cu124
git clone https://github.com/open-mmlab/OpenPCDet ~/OpenPCDet
cd ~/OpenPCDet
pip install -r requirements.txt
TORCH_DONT_CHECK_COMPILER_ABI=1 pip install -e . --no-build-isolation
```

### KITTI Dataset

Download the [KITTI Tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php):
- Velodyne point clouds
- Camera calibration files
- Training labels

```
~/kitti_dataset/
    training/
        calib/          # per-sequence calibration .txt
        label_02/       # per-sequence label .txt
        velodyne/       # per-sequence directory of .bin frames
```

### Model Weights

Download pretrained KITTI checkpoints from the [OpenPCDet model zoo](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/MODEL_ZOO.md) and place in `models/`:

```
models/
    pointpillar_7728.pth    # PointPillars KITTI checkpoint
    pvrcnn.pth              # PV-RCNN KITTI checkpoint
    pvrcnn.yaml             # PV-RCNN model config
```

`PointPillarsDetector` is a self-contained PyTorch reimplementation that loads OpenPCDet `.pth` checkpoints directly — no OpenPCDet install required for PointPillars.

---

## Usage

### Offline evaluation (single sequence)

```bash
PYTHONPATH=. python scripts/evaluate.py --sequence 0000
```

Reports HOTA, MOTA, IDF1, MT/ML, per-class breakdown, and range-stratified metrics (0–20 m, 20–40 m, 40+ m).

### Multi-detector benchmark (all sequences)

```bash
# All detectors with NMS sweep (default)
PYTHONPATH=. python scripts/benchmark.py

# Specific detectors
PYTHONPATH=. python scripts/benchmark.py --detectors point_pillars pvrcnn

# Custom NMS distances for fusion sweep
PYTHONPATH=. python scripts/benchmark.py --nms 0.5 1.0 1.5 2.0

# Skip fusion (run each detector independently)
PYTHONPATH=. python scripts/benchmark.py --no-fusion
```

When both `point_pillars` and `pvrcnn` are requested, each model is loaded once and detections are cached per frame — fusion NMS distances are swept at no extra GPU cost.

### Hyperparameter tuning

```bash
# Tune tracking params for PV-RCNN (Bayesian, 100 trials)
PYTHONPATH=. python scripts/tune_tracker.py --detector pvrcnn --trials 100
```

### Live ROS2 pipeline

```bash
source /opt/ros/jazzy/setup.bash
ros2 launch launch/pipeline.launch.py
ros2 launch launch/pipeline.launch.py sequence:=0001 dataset_path:=~/kitti_dataset
```

Detection mode is set via `detection.mode` in `config/default.yaml`.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **HOTA** | Higher Order Tracking Accuracy — geometric mean of DetA and AssA, averaged over 15 distance thresholds (0.5–4.0 m). Balances detection quality and identity continuity equally. |
| **DetA** | Detection Jaccard — TP / (TP + FP + FN) over all frames. |
| **AssA** | Association Jaccard — measures how consistently each detected object keeps the same track ID. Penalises ID switches. |
| **IDF1** | ID F1 — 2·IDTP / (total GT detections + total predicted detections). Bipartite-matches GT trajectories to predicted trajectories by maximum overlap. |
| **MOTA** | 1 − (FP + FN + IDS) / GT. Traditional metric; can go negative and is dominated by FP/FN counts. |
| **MT / ML** | Mostly Tracked (≥80% of lifetime matched) / Mostly Lost (≤20%). |

3D centre-distance matching is used throughout (not IoU), which is the natural metric for point cloud data. Default threshold: 2.0 m.

---

## Tests

```bash
pytest tests/ -v
```

54 unit tests covering:
- `tests/test_metrics.py` — MOTA, HOTA, MT/ML, IDF1, per-class (24 tests)
- `tests/test_tracker.py` — Sort3D lifecycle: confirmation, deletion, multi-track, duplicate suppression, Kalman prediction (14 tests)
- `tests/test_preprocessing.py` — range\_crop, voxel\_downsample, remove\_ground (16 tests)

---

## Project Structure

```
lidar_tracker/core/
    data/           kitti_loader — KITTI calibration + label parsing
    detection/      euclidean, point_pillars, openpcdet_detector, fusion, factory
    tracking/       sort3d, track, kalman
    preprocessing/  filters (range_crop, voxel_downsample), ground_filter, ego_motion
    evaluation/     metrics (HOTA, MOTA, IDF1, MT/ML, per-class)
scripts/
    evaluate.py     single-sequence offline eval with range breakdown
    benchmark.py    multi-detector comparison across all sequences
    tune_tracker.py Bayesian hyperparameter optimisation (Optuna)
tests/
config/default.yaml
```
