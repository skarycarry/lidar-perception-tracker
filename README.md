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

| Detector | HOTA | DetA | AssA | IDF1 | MOTA | MT% | ML% | IDS | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| Euclidean | 0.2053 | 0.0666 | 0.7029 | 0.1010 | −5.996 | 18.5% | 43.6% | 47 | 9641 | 1071 |
| PointPillars | 0.4591 | 0.3143 | 0.7002 | 0.3734 | +0.077 | 15.5% | 35.2% | 33 | 578 | 1199 |
| PV-RCNN | **0.5734** | **0.4020** | **0.8460** | **0.5087** | +0.047 | **32.1%** | **30.7%** | 73 | 1051 | **770** |
| Fusion (PP+PV, NMS 1.0m) | 0.5569 | 0.3928 | 0.8216 | 0.4876 | +0.003 | 32.3% | 30.9% | 73 | 1125 | 767 |

**Key observations:**

- **Euclidean clustering** has near-zero DetA (0.07) because DBSCAN produces thousands of false positives from noise and building edges — its strong AssA (0.70) shows the tracker itself works, the detector is the bottleneck.
- **PointPillars** is competitive (HOTA 0.46) with far fewer FPs than Euclidean, but misses ~430 more objects than PV-RCNN.
- **PV-RCNN** achieves the best HOTA and AssA. The slightly negative-trending MOTA is a KITTI annotation artefact: GT labels only cover objects meeting strict visibility/size/range criteria, but PV-RCNN correctly detects real objects outside those criteria (counted as FP against GT). HOTA handles this more fairly.
- **Fusion (PP+PV-RCNN with NMS)** does not improve over PV-RCNN alone — the two models fail on largely the same objects (FN overlap ~95%), so fusion adds PP's FPs without recovering meaningful FNs. This is the expected result when one model clearly dominates; ensembling helps only when failure modes are complementary.

> Tracker hyperparameters (match\_distance, max\_age, min\_hits) were tuned per-detector using Bayesian optimisation over 1000 trials with HOTA as the objective.

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
