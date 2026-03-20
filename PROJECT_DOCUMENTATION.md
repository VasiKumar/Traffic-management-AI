# Traffic Manage AI - Full Project Documentation

This document explains the full codebase, theory, and working of your ML project in simple language.

## 1. Project Summary

Traffic Manage AI is a computer vision + machine learning project that:

- takes 2 or 3 road videos,
- detects vehicles using your trained YOLO model (`best.pt`),
- estimates traffic pressure for each road,
- decides which road should get more green signal time,
- trains an ML classifier (Random Forest or KNN) for traffic level prediction,
- shows evaluation outputs like accuracy and confusion matrix,
- displays processed videos with bounding boxes and labels.

The main goal is adaptive traffic control: allow heavily loaded roads to pass first and reduce congestion.

## 2. Prerequisites

### 2.1 Software

- Python 3.11 (recommended)
- pip
- Streamlit
- Ultralytics YOLO
- OpenCV
- NumPy
- Pandas
- scikit-learn

All required packages are listed in [requirements.txt](requirements.txt).

### 2.2 Model and Input

- Trained YOLO model file: `best.pt`
- Input videos: 2 or 3 road videos (`.mp4`, `.avi`, `.mov`, `.mkv`)

### 2.3 Hardware (Recommended)

- CPU: modern multi-core processor
- RAM: at least 8 GB
- GPU (optional): improves YOLO inference speed significantly

## 3. Codebase Components

### 3.1 [app.py](app.py)

This is the Streamlit frontend and orchestration layer.

Responsibilities:

- builds the web interface,
- accepts user videos,
- loads YOLO model,
- calls backend analysis for each road,
- displays summary tables/charts,
- displays processed labeled videos,
- runs ML classifier and shows confusion matrix,
- displays final signal control decision.

### 3.2 [traffic_engine.py](traffic_engine.py)

This is the backend logic engine.

Responsibilities:

- runs frame-level YOLO inference,
- counts vehicles per class,
- computes weighted and traffic features,
- computes congestion scores,
- generates signal timing plan,
- prepares ML dataset from sampled frames,
- returns processed video paths.

### 3.3 [requirements.txt](requirements.txt)

Dependency list for reproducible setup.

### 3.4 [README.md](README.md)

Quick-start guide (short form). This document is the full form.

## 4. End-to-End Workflow

1. User opens app and uploads 2 or 3 road videos.
2. YOLO model loads from `best.pt`.
3. Each road video is sampled frame-by-frame (based on `sample_every_n_frames`).
4. YOLO detects vehicles in sampled frames.
5. Vehicle class counts are accumulated.
6. Per-road traffic metrics are computed.
7. Congestion scores are compared among roads.
8. Signal plan allocates green time by congestion pressure.
9. Processed videos are saved with boxes/labels and shown in UI.
10. ML dataset is built from frame-level features.
11. Random Forest or KNN is trained and tested.
12. Accuracy and confusion matrix are shown.
13. Predicted traffic level (Low/Medium/High) is shown per road.

## 5. Vehicle Classes and Weights

The app is aligned with your trained classes:

- bicycle
- bike
- car
- bus
- truck
- rickshaw
- covered_van

Default traffic weights in [traffic_engine.py](traffic_engine.py):

- bicycle = 0.8
- bike = 1.2
- car = 1.0
- bus = 2.8
- truck = 3.0
- rickshaw = 1.5
- covered_van = 1.7

Why weights?

A truck blocks road capacity more than a bicycle. Weighted counting better reflects real road occupancy.

## 6. Core Traffic Formulas

### 6.1 Total Vehicles

Total vehicles is the sum of all detected class counts.

### 6.2 Heavy Vehicle Ratio

Heavy classes = bus, truck, covered_van.

heavy_ratio = heavy_vehicle_count / total_vehicle_count

### 6.3 Weighted Count

weighted_count = sum(class_count x class_weight)

### 6.4 Density Component

density_component = weighted_count / sampled_frames

### 6.5 Congestion Score

congestion_score = density_component x (1 + 0.65 x heavy_ratio)

Interpretation:

- high weighted density -> more congestion,
- high heavy ratio -> extra congestion penalty,
- final score helps prioritize roads.

## 7. Signal Decision Algorithm

Implemented in `build_signal_plan()`.

Inputs:

- list of road analyses (2 or 3 roads),
- cycle time (default 180 sec),
- minimum green time per road (default 20 sec),
- amber time per phase (default 3 sec).

Logic:

1. Reserve amber time for each road phase.
2. Remaining cycle time is available green time.
3. Compute score share for each road from congestion scores.
4. Allocate green seconds proportional to score share.
5. Enforce minimum green limit for safety and fairness.
6. If rounding causes overflow, cut from lower-priority roads first.
7. Rank roads by congestion score and generate action text.

Output columns:

- road
- priority_rank
- congestion_score
- recommended_green_s
- action

## 8. ML Part (For Academic Requirement)

The project includes explicit ML training in-app.

### 8.1 Feature Engineering

Frame-level features include:

- per-class counts,
- per-class densities,
- total_vehicles,
- total_density,
- weighted_count,
- weighted_density,
- heavy_ratio.

These are generated in [traffic_engine.py](traffic_engine.py).

### 8.2 Label Generation (Pseudo Labels)

Traffic levels are generated from weighted density percentiles:

- Low: <= 33rd percentile
- Medium: > 33rd and <= 66th percentile
- High: > 66th percentile

This allows supervised ML training without manual annotation.

### 8.3 Algorithms Used

Selectable in sidebar:

- Random Forest (default)
- KNN

Yes, the app currently uses Random Forest by default unless changed.

### 8.4 Train/Test Setup

- data split is configurable (`test_size` slider),
- stratified split is used when possible,
- model is trained on train set,
- evaluated on test set.

### 8.5 ML Outputs

- test accuracy,
- confusion matrix (3 classes: Low, Medium, High),
- per-road predicted traffic level.

## 9. Metrics and Definitions

### 9.1 Detection Metrics (YOLO)

Used to evaluate object detection model quality:

- Precision: among predicted boxes, how many are correct.
- Recall: among true objects, how many are detected.
- mAP@0.5: mean Average Precision at IoU 0.5.
- mAP@0.5:0.95: stricter metric averaged across IoU thresholds.

From your shared validation screenshot (example run):

- Precision (all): about 0.615
- Recall (all): about 0.332
- mAP@0.5 (all): about 0.346
- mAP@0.5:0.95 (all): about 0.130

Interpretation:

- Model catches some objects but misses many (recall is relatively low).
- Detection quality is moderate and can be improved with more data, better labels, or more training.

### 9.2 Classification Metrics (ML Section)

- Accuracy: fraction of correct traffic-level predictions.
- Confusion Matrix: table of true vs predicted classes.

Confusion matrix helps identify class-specific weaknesses:

- if many High are predicted as Medium, the model underestimates severe traffic.
- if many Low are predicted as High, model overestimates congestion.

## 10. Thresholds and Tunable Parameters

In UI sidebar:

- Number of roads: 2 or 3
- Sample every Nth frame: controls speed vs detail
- Max sampled frames: controls runtime and dataset size
- Confidence threshold (`conf`): lower value -> more detections, more false positives possible
- IoU threshold (`iou`): controls NMS merging behavior
- Signal cycle length
- Minimum green per road
- ML algorithm selection
- ML test split

## 11. Processed Video Output

After analysis, each road produces a processed video with:

- bounding boxes around detected vehicles,
- class labels,
- same playback ready for viewing in Streamlit.

Stored under `runs/processed/` with timestamped names.

## 12. How to Run

1. Open terminal in project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start app:

```bash
streamlit run app.py
```

4. In browser:

- set model path (default `best.pt`),
- choose 2 or 3 roads,
- upload videos,
- click Analyze.

## 13. Practical Tips for Better Results

- Use videos with clear visibility and stable camera angles.
- Avoid very dark/rainy clips unless model trained for those conditions.
- Keep frame skip moderate (for example 4 to 8) for balance.
- Use longer videos to improve ML dataset quality.
- Ensure class names in model match expected classes.

## 14. Known Limitations

- No object tracking between frames yet (same vehicle may be counted in multiple frames).
- ML labels are pseudo labels, not manually ground-truth labeled.
- Performance depends strongly on YOLO model quality.
- Current signal planning is rule + score based, not reinforcement learning.

## 15. Ideas for Next Improvements

- Add object tracking (SORT/DeepSORT/ByteTrack) to estimate unique vehicles.
- Add queue length estimation per lane.
- Add emergency vehicle priority override.
- Add downloadable CSV reports and processed videos.
- Add historical analytics dashboard.
- Add manually labeled traffic-level dataset for stronger supervised ML.

## 16. Viva / Presentation Ready Explanation (Short)

You can explain your system in one minute:

"Our system takes road videos and runs YOLO to detect vehicles by class. We convert detections into traffic features like total vehicles, heavy vehicle ratio, and weighted density. From this we compute a congestion score and allocate adaptive green times for each road in a signal cycle. For ML validation, we train Random Forest or KNN to classify traffic level into Low, Medium, and High, and show confusion matrix and accuracy. The app also displays processed videos with bounding boxes so users can visually verify detections."

## 17. File Map

- Main app: [app.py](app.py)
- Logic engine: [traffic_engine.py](traffic_engine.py)
- Setup dependencies: [requirements.txt](requirements.txt)
- Quick start: [README.md](README.md)
- Full document: [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md)
