# Traffic Manage AI (YOLO + Streamlit)

This project uses your trained YOLO model (`best.pt`) to analyze **2 or 3 road videos** and generate an adaptive traffic signal decision plan.

Full project explanation is available in [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Streamlit Cloud Notes

If deployment fails while importing `cv2` or `ultralytics`, this repo includes:

- `opencv-python-headless` in `requirements.txt`
- Linux runtime dependencies in `packages.txt` (`libgl1`, `libglib2.0-0`, `ffmpeg`)

After updating dependencies, in Streamlit Cloud do:

1. Reboot app
2. Clear cache
3. Redeploy from latest commit

## What It Does

- Accepts 2 or 3 videos (Road 1, Road 2, optional Road 3)
- Detects vehicles per road using YOLO
- Computes congestion score using vehicle counts and heavy vehicle ratio
- Recommends green-time allocation for each road in one signal cycle
- Prioritizes the highest-pressure road
- Includes ML classifier option: Random Forest or KNN
- Shows confusion matrix for traffic-level classification

## Vehicle Classes Supported

The UI and scoring logic are aligned to your trained classes:

- bicycle
- bike
- bus
- car
- rickshaw
- truck
- covered_van

## Run

1. Create and activate your environment (Python 3.11 recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. In the sidebar, ensure model path points to `best.pt`.
5. Choose number of roads (2 or 3), upload videos, and click **Analyze Traffic and Build Plan**.

## Scoring Logic (Current)

For each road:

- Count detected vehicles by class on sampled frames
- Compute weighted count (heavy vehicles contribute more)
- Compute heavy ratio = heavy vehicles / total vehicles
- Congestion score:

```text
density_component = weighted_count / sampled_frames
congestion_score = density_component * (1 + 0.65 * heavy_ratio)
```

Signal planning:

- Split cycle green time proportionally to congestion scores
- Respect minimum green per road
- Rank roads by congestion score

## ML Project Section (Random Forest / KNN)

The app includes an ML pipeline for your project report:

- Features are extracted from sampled frames using YOLO counts:
	- class counts
	- weighted count
	- total vehicles
	- heavy vehicle ratio
- Pseudo-labels are assigned from weighted density percentiles:
	- Low, Medium, High traffic
- Choose classifier in sidebar:
	- Random Forest
	- KNN
- The app trains/test-splits data and displays:
	- accuracy
	- confusion matrix
	- predicted traffic level per road

This gives you a complete computer-vision + ML workflow for academic submission.

You can modify the weights and formulas in `traffic_engine.py`.
