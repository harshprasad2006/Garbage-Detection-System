---
title: Garbage Detection System
emoji: 🗑️
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
---

# 🗑️ Garbage Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-brightgreen)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

An end-to-end deep learning application that detects and classifies garbage
in **images**, **uploaded videos**, and **live webcam captures**, using a
fine-tuned YOLOv8s object detection model. Includes a FastAPI backend, a
custom-themed Streamlit frontend, and a public live deployment.

---

## 🚀 Live Demo

**[Try it live →](https://garbage-detection-system-levvdwgyxwf2wfepzuc7rb.streamlit.app/)**

Deployed on Streamlit Community Cloud. (Note: Hugging Face Spaces recently
moved compute-based SDKs — Gradio and Docker — behind a paid plan for new
accounts, with only static, non-compute Spaces remaining free. Since this
project requires a running Python backend, Streamlit Community Cloud was
used instead to provide a genuinely free, permanent, public deployment. A
`Dockerfile` is still included in this repo for Hugging Face Spaces
Docker-SDK deployment, should that path become available again.)

---

## 🧠 Model Overview

| | |
|---|---|
| **Architecture** | YOLOv8s (fine-tuned, not pretrained inference) |
| **Classes (10)** | battery, biological, cardboard, clothing, glass, metal, paper, plastic, shoes, trash |
| **Training hardware** | Google Colab (Tesla T4 GPU, free tier) |
| **Epochs** | 50 |
| **Image size** | 640 × 640 |
| **Dataset** | [Roboflow Garbage Detection (v7)](https://universe.roboflow.com/idk-expry/garbage-detection-4fx3k/dataset/7) — 5,338 images |
| **Split** | 70% train / 20% valid / 10% test (leakage-safe, grouped by source image) |

Full methodology, augmentation configuration, and evaluation metrics are in
[`training_report.md`](training_report.md).

---

## 📊 Results

| Metric | Value |
|---|---|
| Precision | 0.806 |
| Recall | 0.703 |
| mAP@50 | 0.787 |
| mAP@50-95 | 0.650 |

**Strongest classes:** clothing (mAP50 = 0.965), shoes (mAP50 = 0.926)
**Weakest classes:** biological, trash, paper (recall 0.54–0.58, due to
visual ambiguity and class imbalance)

Full per-class breakdown, confusion matrix, PR curve, and F1 curve are in
[`training_report.md`](training_report.md) and `training/results/`.

---

## 🏗️ Project Structure

```
Garbage-Detection-System/
├── dataset/            # train/valid/test images + YOLO labels (leakage-safe split)
├── dataset_old/        # original Roboflow export (kept for reference)
├── notebooks/          # Colab training notebook
├── models/             # trained best.pt weights
├── backend/            # detector.py (shared core), FastAPI app.py (/predict), inference scripts
├── frontend/           # Streamlit UI (image/video/webcam/history tabs)
├── training/           # data.yaml, split_dataset.py, results/ (plots, curves)
├── screenshots/        # UI + result screenshots
├── Dockerfile          # For Hugging Face Spaces / Docker-based deployment
├── packages.txt        # System dependencies for Streamlit Community Cloud
├── README.md
├── requirements.txt
└── training_report.md
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/harshprasad2006/Garbage-Detection-System.git
cd Garbage-Detection-System

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Running Locally

**Backend (FastAPI):**
```bash
cd backend
uvicorn app:app --reload --port 8000
```
Interactive API docs available at `http://localhost:8000/docs`.

**Frontend (Streamlit):**
```bash
cd frontend
streamlit run app.py
```
Then open `http://localhost:8501` in your browser.

**Standalone inference scripts** (image / video / webcam), useful for
quick local testing without the API or UI:
```bash
cd backend
python image_detector.py --image path/to/image.jpg
python video_detector.py --video path/to/video.mp4
python webcam_detector.py
```

---

## 🔌 API Reference

### `POST /predict`

Accepts an image file and returns detected garbage objects.

**Request:** `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | image file | The image to run detection on (jpg/png) |

**Response:** `200 OK`
```json
{
  "detections": [
    {
      "class": "battery",
      "confidence": 0.9518,
      "bounding_box": {
        "x1": 8.24,
        "y1": 38.85,
        "x2": 624.31,
        "y2": 598.55
      }
    }
  ]
}
```

### `GET /`
Health check — confirms the API is running.

---

## 🖼️ Dataset & Preprocessing

Two candidate sources were considered: **TACO** (Trash Annotations in
Context) and **Roboflow's Garbage Detection dataset**. TACO offers
realistic, real-world litter images but uses COCO-format annotations
across 60 fine-grained, imbalanced classes, requiring conversion to YOLO
format and additional class-grouping work. The Roboflow dataset was
selected instead because it is natively in YOLO format, has a manageable
size for training on a free Colab T4 GPU within reasonable time, and
provides 10 practical material-based classes suited to a real-world
garbage classification use case.

The original Roboflow export used a 92.3 / 3.8 / 3.9 train/valid/test
split — too imbalanced for reliable evaluation, and contained augmented
copies of the same source images (3 outputs per training example: flips,
rotations, crops, color jitter). To prevent data leakage, a custom script
(`training/split_dataset.py`) grouped images by source-image identity
(filename prefix before `.rf.<hash>`) and split **groups** — not
individual files — into a proper 70/20/10 ratio using a fixed random seed
(42), guaranteeing no source image or its augmented variants appear in
more than one subset.

Final split: 3,734 train / 1,068 valid / 536 test images (5,338 total).

Full reasoning and annotation format details are in
[`training_report.md`](training_report.md).

---

## 🏋️ Training

Training was performed in Google Colab using a free Tesla T4 GPU, fine-tuning
YOLOv8s (pretrained on COCO) for 50 epochs at 640×640 resolution.

**Augmentations explicitly configured** (not left as silent defaults):

| Augmentation | Parameter | Value |
|---|---|---|
| Horizontal Flip | `fliplr` | 0.5 |
| Mosaic | `mosaic` | 1.0 |
| Brightness variation | `hsv_v` | 0.4 |
| Contrast/Saturation variation | `hsv_s` | 0.7 |
| Scale jitter | `scale` | 0.5 |

See [`notebooks/train_yolov8_garbage.ipynb`](notebooks/train_yolov8_garbage.ipynb)
for the full training pipeline, and [`training_report.md`](training_report.md)
for complete configuration details, results, and analysis.

---

## 🧪 Evaluation Metrics

See [Results](#-results) above and [`training_report.md`](training_report.md)
for full per-class metrics, confusion matrix, precision-recall curve, and
F1 curve.

---

## 📦 Deployment

The Streamlit frontend is deployed on **Streamlit Community Cloud**,
connected directly to this GitHub repository (`main` branch,
`frontend/app.py` as the entry point). `packages.txt` installs the system
libraries OpenCV requires (`libgl1`, `libglib2.0-0`) in the cloud
environment, and `requirements.txt` installs all Python dependencies.

Live app: https://garbage-detection-system-levvdwgyxwf2wfepzuc7rb.streamlit.app/

A `Dockerfile` is also included for Hugging Face Spaces (Docker SDK)
deployment as an alternative path.

---

## 🛠️ Tech Stack

- **Model:** YOLOv8s (Ultralytics)
- **Backend:** FastAPI
- **Frontend:** Streamlit
- **Training:** Google Colab (T4 GPU)
- **Deployment:** Streamlit Community Cloud

---

## 📄 License

This project is licensed under the MIT License.
