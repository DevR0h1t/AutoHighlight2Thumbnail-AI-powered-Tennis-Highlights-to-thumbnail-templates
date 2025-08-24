# 🎾 AProj: Real‑Time Tennis Analytics

**A pipeline for automated tennis match analysis** — detect players, track the ball, reconstruct the court, and compute shot & player speed statistics in real time.

<img width="1037" height="590" alt="{35F04BFD-4684-431B-843A-23DAC86F5208}" src="https://github.com/user-attachments/assets/4771218a-7f38-4f5d-bd99-cc71c7981fe4" />

---

## 🚀 Features

* **YOLOv8 Object Detection** for players & ball 🏃‍♂️⚪
* **Court Keypoint Regression** using a ResNet‑50 head 🎯
* **Mini‑Court Overlay**: Bird’s‑eye schematic of player & ball positions 🗺️
* **Shot & Movement Metrics**: Ball shot speed, opponent run speed, cumulative & average stats 📊
* **Highlight Exporter**: Annotated video + JSON metadata of all frame‑level stats 🎞️
* **Flexible Training CLI**: Retrain detectors & keypoint model with your own data 🎓
* **Stub Caching**: Speed up development with pickle‑based I/O shortcuts 🐾

---

## 🔧 Installation

1. **Clone the repo**:

   ```bash
   git clone https://github.com/yourusername/aproj.git
   cd aproj
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/macOS
   venv\\Scripts\\activate  # Windows PowerShell
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download pretrained weights** (or train your own):

   ```bash
   # Into weights/ folder
   gdown --id <YOLO_WEIGHTS_ID>  -O weights/yolov8-tennis.pt
   gdown --id <KEYPTS_WEIGHTS_ID> -O weights/court_keypoints.pt
   ```

---

## 🎓 Training Models

Use the unified CLI script to (re)train your detectors and keypoint regressor:

```bash
python model_training/training.py --task all \
    --data_yaml data/tennis_ball.yaml \
    --keypoint_dir data/court_keypoints \
    --epochs 50 --imgsz 640 --batch_size 16
```

* Outputs go to `weights/` by default:

  * `weights/yolov8-tennis/best.pt` (YOLO detector)
  * `weights/court_keypoints.pt` (ResNet‑50 keypoint model)

---

## 🎬 Running the Pipeline

Open `main.ipynb` in Jupyter or run the Python script:

```bash
jupyter lab main.ipynb
# Or in plain Python:
python run_pipeline.py \
    --input_video input_videos/match.mp4 \
    --output_dir output
```

What you’ll get:

* `output/annotated.mp4` : Full match video with boxes, overlays, and stats
* `output/highlights.json`: Frame‑by‑frame JSON of ball speeds, shot counts, run speeds

---

## 📂 File Structure

```
aproj/
├── data/                     # Your datasets (YOLO .yaml, keypoint images/labels)
├── model_training/           # CLI script to train models
├── models/                   # CourtKeypointModel definition
├── preprocessing/            # KeypointsDataset + Spark demo
├── trackers/                 # Player, Ball, CourtLine, MiniCourt modules
├── weights/                  # Trained .pt files (committed or generated)
├── utils.py                  # Video I/O & helper functions
├── highlight_exporter.py     # Video + metadata export logic
├── main.ipynb                # End‑to‑end notebook
├── requirements.txt
└── README.md
```

---

## 🤝 Contributing

We welcome improvements! Please:

1. Fork the repo 🏴
2. Create a feature branch 🌿
3. Submit a pull request 🚀


*Game, set, match — automated.* 🏆
