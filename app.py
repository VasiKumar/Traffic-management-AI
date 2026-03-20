from __future__ import annotations

import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from ultralytics import YOLO

from traffic_engine import analyze_road_video, build_ml_dataset, build_signal_plan, road_feature_row


st.set_page_config(page_title="Traffic Manage AI", page_icon="🚦", layout="wide")


def inject_custom_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Source+Serif+4:wght@500;700&display=swap');

        :root {
            --bg-soft: #f6fbff;
            --panel: #ffffff;
            --ink: #0f2a3a;
            --muted: #436173;
            --line: #cbe3f2;
            --accent: #0c8db3;
            --accent-2: #1ab394;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 10%, #d9f6ff 0%, transparent 35%),
                radial-gradient(circle at 90% 15%, #d9ffe8 0%, transparent 30%),
                linear-gradient(180deg, #fafdff 0%, #f2f9ff 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
        }

        .block-container {
            max-width: 1280px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3 {
            font-family: 'Source Serif 4', serif;
            color: #0f3042;
            letter-spacing: 0.2px;
        }

        h1 {
            font-size: 2.35rem !important;
        }

        h2 {
            font-size: 1.65rem !important;
        }

        h3 {
            font-size: 1.3rem !important;
        }

        p, li, label, .stMarkdown, .stCaption {
            font-size: 1.06rem !important;
            color: var(--ink);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #e6f6ff 0%, #effff7 100%);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p {
            color: #0d3448 !important;
            font-size: 1.02rem !important;
        }

        div[data-baseweb="select"] > div,
        .stTextInput > div > div > input,
        .stNumberInput input,
        .stSlider,
        .stFileUploader {
            font-size: 1.02rem !important;
        }

        .stButton > button {
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
            color: #ffffff;
            border: none;
            border-radius: 12px;
            font-size: 1.02rem;
            font-weight: 700;
            padding: 0.65rem 1rem;
            box-shadow: 0 6px 20px rgba(12, 141, 179, 0.25);
        }

        .stButton > button:hover {
            filter: brightness(1.05);
            transform: translateY(-1px);
        }

        .stDataFrame, .stTable {
            border: 1px solid var(--line);
            border-radius: 12px;
            overflow: hidden;
            background: var(--panel);
        }

        [data-testid="stMetric"] {
            background: #ffffffcc;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.5rem 0.75rem;
        }

        [data-testid="stAlert"] {
            font-size: 1.02rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_custom_theme()
st.title("Traffic Manage AI - Adaptive Signal Planner")
st.caption("Upload 2 or 3 road videos, detect vehicles with your YOLO model, and get ML-based traffic decisions.")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def save_upload_to_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        return Path(temp_file.name)


def counts_to_row(counts: dict) -> dict:
    row = {
        "bicycle": counts.get("bicycle", 0),
        "bike": counts.get("bike", 0),
        "car": counts.get("car", 0),
        "bus": counts.get("bus", 0),
        "truck": counts.get("truck", 0),
        "rickshaw": counts.get("rickshaw", 0),
        "covered_van": counts.get("covered_van", 0),
    }
    row["total"] = sum(row.values())
    return row


def label_to_text(label: int) -> str:
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    return mapping.get(int(label), "Unknown")


with st.sidebar:
    st.header("Settings")
    model_path = st.text_input("YOLO model path", value="best.pt")
    num_roads = st.selectbox("Number of roads", options=[2, 3], index=1)
    sample_every = st.slider("Sample every Nth frame", min_value=1, max_value=20, value=8)
    max_sampled_frames = st.slider("Max sampled frames per road", min_value=50, max_value=1200, value=500, step=50)
    conf = st.slider("Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05)
    iou = st.slider("IoU threshold", min_value=0.05, max_value=0.95, value=0.45, step=0.05)
    cycle_seconds = st.slider("Signal cycle length (s)", min_value=60, max_value=300, value=180, step=10)
    min_green_seconds = st.slider("Minimum green per road (s)", min_value=10, max_value=60, value=20, step=5)
    st.subheader("ML Classifier")
    ml_algo = st.selectbox("Algorithm", options=["Random Forest", "KNN"], index=0)
    test_size = st.slider("ML test split", min_value=0.2, max_value=0.5, value=0.3, step=0.05)

col1, col2, col3 = st.columns(3)
with col1:
    road1 = st.file_uploader("Road 1 video", type=["mp4", "avi", "mov", "mkv"], key="r1")
with col2:
    road2 = st.file_uploader("Road 2 video", type=["mp4", "avi", "mov", "mkv"], key="r2")
with col3:
    road3 = st.file_uploader("Road 3 video", type=["mp4", "avi", "mov", "mkv"], key="r3")

run_btn = st.button("Analyze Traffic and Build Plan", type="primary", use_container_width=True)

if run_btn:
    uploads = [road1, road2] if num_roads == 2 else [road1, road2, road3]
    if any(u is None for u in uploads):
        st.error(f"Please upload all {num_roads} road videos before running analysis.")
        st.stop()

    model_file = Path(model_path)
    if not model_file.exists():
        st.error(f"Model file not found: {model_file}")
        st.stop()

    try:
        model = load_model(str(model_file))
    except Exception as exc:
        st.exception(exc)
        st.stop()

    temp_paths: List[Path] = []
    analyses = []
    progress = st.progress(0)
    status = st.empty()
    processed_output_dir = Path("runs") / "processed"

    try:
        for idx, upload in enumerate(uploads, start=1):
            status.info(f"Analyzing Road {idx}...")
            temp_path = save_upload_to_temp(upload)
            temp_paths.append(temp_path)

            analysis = analyze_road_video(
                road_name=f"Road {idx}",
                video_path=temp_path,
                model=model,
                sample_every_n_frames=sample_every,
                max_sampled_frames=max_sampled_frames,
                conf=conf,
                iou=iou,
                save_annotated_video=True,
                annotated_output_dir=processed_output_dir,
            )
            analyses.append(analysis)
            progress.progress(idx / num_roads)

        status.success("Analysis complete.")

        road_rows = []
        for a in analyses:
            row = {
                "road": a.road_name,
                "sampled_frames": a.sampled_frames,
                "weighted_count": round(a.weighted_count, 2),
                "heavy_ratio": round(a.heavy_ratio, 3),
                "congestion_score": round(a.congestion_score, 3),
            }
            row.update(counts_to_row(a.counts))
            road_rows.append(row)

        road_df = pd.DataFrame(road_rows)
        st.subheader("Per-Road Detection Summary")
        st.dataframe(road_df, use_container_width=True)

        st.subheader("Congestion Score Comparison")
        st.bar_chart(road_df.set_index("road")["congestion_score"])

        plan = build_signal_plan(
            analyses,
            cycle_seconds=cycle_seconds,
            min_green_seconds=min_green_seconds,
        )
        plan_df = pd.DataFrame(plan)
        st.subheader("Recommended Signal Plan")
        st.dataframe(plan_df, use_container_width=True)

        top_road = plan_df.iloc[0]["road"]
        top_green = int(plan_df.iloc[0]["recommended_green_s"])
        st.success(f"Highest traffic pressure: {top_road}. Recommended immediate green: {top_green} seconds.")

        st.subheader("Processed Videos (Boxes + Labels)")
        video_cols = st.columns(len(analyses))
        for i, analysis in enumerate(analyses):
            with video_cols[i]:
                st.markdown(f"**{analysis.road_name}**")
                if analysis.annotated_video_path:
                    st.video(analysis.annotated_video_path)
                else:
                    st.warning("Processed video was not generated.")

        st.subheader("ML Traffic Classifier (Project Section)")
        st.caption("Pseudo-labels are created from weighted vehicle density percentiles per sampled frame: low, medium, high.")

        X, y, feature_names, quantiles = build_ml_dataset(analyses)

        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            st.warning("ML training skipped: sampled data has only one class. Use longer or more varied videos.")
        else:
            stratify = y if len(unique_labels) > 1 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=42,
                stratify=stratify,
            )

            if ml_algo == "Random Forest":
                clf = RandomForestClassifier(n_estimators=200, random_state=42)
            else:
                clf = KNeighborsClassifier(n_neighbors=7)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

            st.write(f"Algorithm used: {ml_algo}")
            st.write(f"Test accuracy: {acc:.3f}")
            st.write(
                "Traffic thresholds (weighted density): "
                f"Low <= {quantiles['q_low']:.3f}, "
                f"Medium <= {quantiles['q_high']:.3f}, "
                "High > medium threshold"
            )

            cm_df = pd.DataFrame(
                cm,
                index=["True Low", "True Medium", "True High"],
                columns=["Pred Low", "Pred Medium", "Pred High"],
            )
            st.write("Confusion Matrix")
            st.dataframe(cm_df, use_container_width=True)

            road_feature_vectors = []
            for analysis in analyses:
                row = road_feature_row(analysis)
                road_feature_vectors.append([row[name] for name in feature_names])

            road_pred = clf.predict(np.array(road_feature_vectors, dtype=float))
            ml_rows = []
            for analysis, pred in zip(analyses, road_pred):
                ml_rows.append(
                    {
                        "road": analysis.road_name,
                        "predicted_level": label_to_text(int(pred)),
                        "congestion_score": round(analysis.congestion_score, 3),
                    }
                )

            ml_df = pd.DataFrame(ml_rows)
            st.write("Predicted Traffic Level per Road")
            st.dataframe(ml_df, use_container_width=True)

    except Exception as exc:
        st.exception(exc)
    finally:
        for path in temp_paths:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
