# Traffic Manage AI (YOLO + Streamlit)

This project uses your trained YOLO model (`best.pt`) to analyze **2 or 3 road videos** and generate an adaptive traffic signal decision plan.

Full project explanation is available in [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

## Streamlit Cloud Notes

If deployment fails while importing `cv2` or `ultralytics`, this repo includes:

- `opencv-python-headless` in `requirements.txt`
- Linux runtime dependency in `packages.txt` (`libgl1`)

After updating dependencies, in Streamlit Cloud do:

1. Reboot app
2. Clear cache
3. Redeploy from latest commit

## What It Does

- Accepts 2 or 3 videos (Road 1, Road 2, optional Road 3)
- Detects vehicles per road using YOLO
- Computes traffic features from YOLO detections (total vehicles, weighted density, heavy vehicle ratio, per-class counts)
- Accepts per-road context from user dropdowns (Day/Night and Good/Bad road)
- Adds context features to ML input (is_night, is_bad_road, night_bad_interaction)
- Uses Random Forest or KNN as the **primary signal decision engine**
- Maps predicted traffic level to signal timing: Low -> minimum, Medium -> moderate, High -> maximum
- Recommends green-time allocation and road priority from ML predictions
- Shows confusion matrix and compares ML decisions against rule-based decisions

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

## ML-Driven Decision Logic (Current)

For each sampled frame and each road:

- Count detected vehicles by class
- Compute feature vector including:
	- total_vehicles
	- weighted_density
	- heavy_ratio
	- per-class counts
- Add user-provided context features per road:
	- Day/Night
	- Good/Bad road
	- encoded as is_night, is_bad_road, and interaction term
- Generate pseudo-labels (Low/Medium/High) from weighted-density percentiles
- Apply a small context-aware adjustment when creating pseudo-labels so Night/Bad-road scenarios receive higher operational pressure
- Train Random Forest or KNN on internally generated YOLO feature data
- Predict traffic level per road
- Convert predictions to green time:
	- Low -> minimum green
	- Medium -> moderate green
	- High -> maximum green

The older congestion score is still computed for comparison and analysis:

```text
density_component = weighted_count / sampled_frames
congestion_score = density_component * (1 + 0.65 * heavy_ratio)
```

Rule-based planning is shown side-by-side with ML planning for evaluation.

## ML Project Section (Random Forest / KNN)

The app includes an ML pipeline for your project report:

- Features are extracted from sampled frames using YOLO counts:
	- class counts
	- weighted count
	- weighted density
	- total vehicles
	- heavy vehicle ratio
	- context flags from dropdown (Day/Night, Good/Bad road)
- Pseudo-labels are assigned from weighted density percentiles:
	- Low, Medium, High traffic
- Choose classifier in sidebar:
	- Random Forest
	- KNN
- The app trains/test-splits data and displays:
	- accuracy
	- confusion matrix
	- predicted traffic level per road
	- ML-driven signal plan
	- ML vs rule-based comparison table

This gives you a complete computer-vision + ML workflow for academic submission.

You can modify feature weights, ML mapping thresholds, and signal-time mapping in `traffic_engine.py`.
