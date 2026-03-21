from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import time

import cv2
import numpy as np
from ultralytics import YOLO


DEFAULT_CLASS_WEIGHTS: Dict[str, float] = {
    "bicycle": 0.8,
    "bike": 1.2,
    "car": 1.0,
    "bus": 2.8,
    "truck": 3.0,
    "rickshaw": 1.5,
    "covered_van": 1.7,
}

HEAVY_VEHICLE_CLASSES = {"bus", "truck", "covered_van"}
CLASS_COLUMNS = ["bicycle", "bike", "car", "bus", "truck", "rickshaw", "covered_van"]
TRAFFIC_LEVEL_LABELS = {0: "Low", 1: "Medium", 2: "High"}


@dataclass
class RoadAnalysis:
    road_name: str
    frame_count: int
    sampled_frames: int
    counts: Dict[str, int]
    weighted_count: float
    heavy_ratio: float
    congestion_score: float
    sample_features: List[Dict[str, float]]
    annotated_video_path: str | None


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def counts_to_features(counts: Dict[str, int], sampled_frames: int, class_weights: Dict[str, float]) -> Dict[str, float]:
    """Convert class counts into ML-ready numeric features."""
    safe_frames = max(sampled_frames, 1)
    total_vehicles = float(sum(counts.values()))

    weighted_count = 0.0
    heavy_count = 0.0
    features: Dict[str, float] = {}

    for class_name in CLASS_COLUMNS:
        count = float(counts.get(class_name, 0))
        features[f"count_{class_name}"] = count
        features[f"density_{class_name}"] = count / safe_frames
        weighted_count += class_weights.get(class_name, 1.0) * count
        if class_name in HEAVY_VEHICLE_CLASSES:
            heavy_count += count

    heavy_ratio = (heavy_count / total_vehicles) if total_vehicles else 0.0
    weighted_density = weighted_count / safe_frames

    features["total_vehicles"] = total_vehicles
    features["total_density"] = total_vehicles / safe_frames
    features["weighted_count"] = weighted_count
    features["weighted_density"] = weighted_density
    features["heavy_ratio"] = heavy_ratio
    return features


def analyze_road_video(
    road_name: str,
    video_path: Path,
    model: YOLO,
    class_weights: Dict[str, float] | None = None,
    sample_every_n_frames: int = 8,
    max_sampled_frames: int = 500,
    conf: float = 0.25,
    iou: float = 0.45,
    save_annotated_video: bool = False,
    annotated_output_dir: Path | None = None,
) -> RoadAnalysis:
    """Analyze one road video and produce traffic metrics."""
    weights = class_weights or DEFAULT_CLASS_WEIGHTS

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    safe_fps = fps if fps and fps > 0 else 20.0
    counts: Dict[str, int] = {}
    sample_features: List[Dict[str, float]] = []
    sampled = 0
    frame_index = 0
    writer = None
    annotated_video_path: str | None = None

    if save_annotated_video:
        output_dir = annotated_output_dir or Path("runs") / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = _normalize_name(road_name)
        out_path = output_dir / f"{safe_name}_{int(time.time())}.mp4"
        annotated_video_path = str(out_path)

    while cap.isOpened() and sampled < max_sampled_frames:
        ok, frame = cap.read()
        if not ok:
            break

        if save_annotated_video and writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                annotated_video_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                safe_fps,
                (w, h),
            )

        annotated_frame = frame

        if frame_index % max(sample_every_n_frames, 1) != 0:
            if writer is not None:
                writer.write(annotated_frame)
            frame_index += 1
            continue

        sampled += 1
        result = model.predict(frame, conf=conf, iou=iou, verbose=False)
        boxes = result[0].boxes
        frame_counts: Dict[str, int] = {}
        annotated_frame = result[0].plot()

        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
            for class_id in class_ids:
                class_name = _normalize_name(result[0].names.get(int(class_id), str(class_id)))
                counts[class_name] = counts.get(class_name, 0) + 1
                frame_counts[class_name] = frame_counts.get(class_name, 0) + 1

        frame_features = counts_to_features(frame_counts, sampled_frames=1, class_weights=weights)
        frame_features["frame_index"] = float(frame_index)
        sample_features.append(frame_features)

        if writer is not None:
            writer.write(annotated_frame)

        frame_index += 1

    cap.release()
    if writer is not None:
        writer.release()

    weighted_count = 0.0
    heavy_count = 0
    total_count = sum(counts.values())

    for class_name, count in counts.items():
        weighted_count += weights.get(class_name, 1.0) * count
        if class_name in HEAVY_VEHICLE_CLASSES:
            heavy_count += count

    heavy_ratio = (heavy_count / total_count) if total_count else 0.0

    # Score combines weighted vehicle load and heavy-vehicle pressure.
    density_component = weighted_count / max(sampled, 1)
    congestion_score = density_component * (1.0 + 0.65 * heavy_ratio)

    return RoadAnalysis(
        road_name=road_name,
        frame_count=frame_count,
        sampled_frames=sampled,
        counts=counts,
        weighted_count=weighted_count,
        heavy_ratio=heavy_ratio,
        congestion_score=congestion_score,
        sample_features=sample_features,
        annotated_video_path=annotated_video_path,
    )


def build_signal_plan(
    analyses: List[RoadAnalysis],
    cycle_seconds: int = 180,
    min_green_seconds: int = 20,
    amber_seconds_per_phase: int = 3,
) -> List[Dict[str, float]]:
    """Return prioritized green-time recommendations from road analyses."""
    if len(analyses) < 2:
        raise ValueError("This planner expects at least two roads.")

    available_green = cycle_seconds - amber_seconds_per_phase * len(analyses)
    if available_green <= min_green_seconds * len(analyses):
        available_green = min_green_seconds * len(analyses)

    raw_scores = np.array([max(a.congestion_score, 0.01) for a in analyses], dtype=float)
    score_sum = float(raw_scores.sum())

    if score_sum == 0.0:
        shares = np.array([1 / len(analyses)] * len(analyses), dtype=float)
    else:
        shares = raw_scores / score_sum

    greens = np.maximum(np.round(shares * available_green), min_green_seconds)

    # If rounding overshoots cycle budget, trim from lowest-priority roads first.
    overflow = int(greens.sum() - available_green)
    if overflow > 0:
        order = np.argsort(raw_scores)
        for idx in order:
            if overflow <= 0:
                break
            reducible = int(greens[idx] - min_green_seconds)
            if reducible <= 0:
                continue
            cut = min(reducible, overflow)
            greens[idx] -= cut
            overflow -= cut

    ranking = np.argsort(-raw_scores)
    rank_map = {int(idx): rank + 1 for rank, idx in enumerate(ranking)}

    plan: List[Dict[str, float]] = []
    for i, analysis in enumerate(analyses):
        score = analysis.congestion_score
        if score >= np.percentile(raw_scores, 66):
            action = "Highest pressure: extend green and consider upstream hold."
        elif score >= np.percentile(raw_scores, 33):
            action = "Moderate pressure: normal adaptive green."
        else:
            action = "Lower pressure: shorter green, monitor spillback."

        plan.append(
            {
                "road": analysis.road_name,
                "priority_rank": rank_map[i],
                "congestion_score": round(score, 3),
                "recommended_green_s": int(greens[i]),
                "action": action,
            }
        )

    plan.sort(key=lambda x: x["priority_rank"])
    return plan


def build_ml_dataset(analyses: List[RoadAnalysis]) -> tuple[np.ndarray, np.ndarray, List[str], Dict[str, float]]:
    """Build feature matrix and pseudo-labels (low/medium/high traffic) from frame-level samples."""
    all_samples: List[Dict[str, float]] = []
    for analysis in analyses:
        all_samples.extend(analysis.sample_features)

    if len(all_samples) < 6:
        raise ValueError("Not enough samples to train classifier. Use longer videos or smaller frame-skip value.")

    feature_names = [
        "count_bicycle",
        "count_bike",
        "count_car",
        "count_bus",
        "count_truck",
        "count_rickshaw",
        "count_covered_van",
        "total_vehicles",
        "weighted_count",
        "heavy_ratio",
        "total_density",
        "weighted_density",
    ]

    weighted_density = np.array([sample["weighted_density"] for sample in all_samples], dtype=float)
    q_low = float(np.percentile(weighted_density, 33))
    q_high = float(np.percentile(weighted_density, 66))

    labels: List[int] = []
    rows: List[List[float]] = []
    for sample in all_samples:
        rows.append([float(sample[name]) for name in feature_names])
        value = float(sample["weighted_density"])
        if value <= q_low:
            labels.append(0)  # low
        elif value <= q_high:
            labels.append(1)  # medium
        else:
            labels.append(2)  # high

    return (
        np.array(rows, dtype=float),
        np.array(labels, dtype=int),
        feature_names,
        {"q_low": q_low, "q_high": q_high},
    )


def road_feature_row(analysis: RoadAnalysis) -> Dict[str, float]:
    """Create one aggregate feature row from full-road counts for model inference."""
    return counts_to_features(
        counts=analysis.counts,
        sampled_frames=analysis.sampled_frames,
        class_weights=DEFAULT_CLASS_WEIGHTS,
    )


def _green_targets_from_level(
    levels: np.ndarray,
    min_green_seconds: int,
    moderate_green_seconds: int,
    max_green_seconds: int,
) -> np.ndarray:
    targets = []
    for level in levels:
        if int(level) <= 0:
            targets.append(int(min_green_seconds))
        elif int(level) == 1:
            targets.append(int(moderate_green_seconds))
        else:
            targets.append(int(max_green_seconds))
    return np.array(targets, dtype=int)


def build_ml_signal_plan(
    analyses: List[RoadAnalysis],
    predicted_levels: np.ndarray,
    cycle_seconds: int = 180,
    min_green_seconds: int = 20,
    moderate_green_seconds: int | None = None,
    max_green_seconds: int | None = None,
    amber_seconds_per_phase: int = 3,
) -> List[Dict[str, float]]:
    """Build signal plan where ML traffic levels are the main decision source."""
    if len(analyses) < 2:
        raise ValueError("This planner expects at least two roads.")
    if len(analyses) != len(predicted_levels):
        raise ValueError("Predicted traffic levels must match number of roads.")

    available_green = cycle_seconds - amber_seconds_per_phase * len(analyses)
    baseline_total = min_green_seconds * len(analyses)
    if available_green < baseline_total:
        available_green = baseline_total

    dynamic_max = int(max(min_green_seconds + 10, round(available_green * 0.55)))
    max_green = int(max_green_seconds) if max_green_seconds is not None else dynamic_max
    max_green = max(max_green, min_green_seconds)
    moderate_green = int(moderate_green_seconds) if moderate_green_seconds is not None else int(
        round((min_green_seconds + max_green) / 2)
    )
    moderate_green = int(np.clip(moderate_green, min_green_seconds, max_green))

    levels = np.array(predicted_levels, dtype=int)
    targets = _green_targets_from_level(levels, min_green_seconds, moderate_green, max_green)
    greens = targets.astype(int)

    if int(greens.sum()) < available_green:
        remainder = int(available_green - greens.sum())
        # Prioritize extra seconds for high, then medium, then low level roads.
        order = np.argsort(-levels)
        i = 0
        while remainder > 0:
            idx = int(order[i % len(order)])
            greens[idx] += 1
            remainder -= 1
            i += 1
    elif int(greens.sum()) > available_green:
        overflow = int(greens.sum() - available_green)
        # Remove seconds from low first, preserving minimum safety green.
        order = np.argsort(levels)
        for idx in order:
            if overflow <= 0:
                break
            reducible = int(greens[idx] - min_green_seconds)
            if reducible <= 0:
                continue
            cut = min(reducible, overflow)
            greens[idx] -= cut
            overflow -= cut

    ranking = sorted(
        range(len(analyses)),
        key=lambda i: (
            -int(levels[i]),
            -road_feature_row(analyses[i])["weighted_density"],
            -analyses[i].congestion_score,
        ),
    )
    rank_map = {idx: rank + 1 for rank, idx in enumerate(ranking)}

    plan: List[Dict[str, float]] = []
    for i, analysis in enumerate(analyses):
        level = int(levels[i])
        level_text = TRAFFIC_LEVEL_LABELS.get(level, "Unknown")
        if level >= 2:
            action = "High ML-priority: keep corridor moving with maximum green."
        elif level == 1:
            action = "Medium ML-priority: balanced green allocation."
        else:
            action = "Low ML-priority: assign minimum green and monitor flow."

        plan.append(
            {
                "road": analysis.road_name,
                "priority_rank": rank_map[i],
                "predicted_level": level_text,
                "predicted_level_id": level,
                "recommended_green_s": int(greens[i]),
                "weighted_density": round(road_feature_row(analysis)["weighted_density"], 3),
                "rule_congestion_score": round(analysis.congestion_score, 3),
                "action": action,
            }
        )

    plan.sort(key=lambda x: x["priority_rank"])
    return plan
