---
title: Garbage Detection System
emoji: 🗑️
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: "1.38.0"
app_file: frontend/app.py
pinned: false
---

# 🗑️ Garbage Detection System

...# 🗑️ Garbage Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-brightgreen)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](https://streamlit.io/)

An end-to-end deep learning application that detects and classifies garbage in **images**, **uploaded videos**, and **live webcam feeds** using a fine-tuned **YOLOv8s** object detection model.

The project includes:

- YOLOv8s object detection model
- FastAPI backend
- Streamlit frontend
- Google Colab training pipeline
- Public deployment using Hugging Face Spaces

---

# ✨ Features

- Detect garbage in uploaded images
- Detect garbage in uploaded videos
- Real-time webcam garbage detection
- Fine-tuned YOLOv8s object detection model
- Leakage-safe dataset preprocessing
- FastAPI REST API
- Streamlit web interface
- Google Colab training pipeline
- Hugging Face Spaces deployment

---

# 🚧 Project Status

This project is currently under active development.

### ✅ Completed

- Dataset selection
- Leakage-safe dataset preprocessing
- Custom dataset splitting (70/20/10)
- YOLOv8 training pipeline
- Google Colab training setup
- Training documentation

### 🔄 In Progress

- Model training
- Model evaluation

### ⏳ Remaining

- FastAPI backend
- Streamlit frontend
- Hugging Face deployment
- Final project testing

---

# 🚀 Live Demo

> *(Hugging Face Spaces deployment link will be added after Phase 9.)*

---

# 📸 Demo Screenshots

> *(Frontend screenshots, prediction examples and UI images will be added after Phase 8.)*

---

# 🧠 Model Overview

| Property | Value |
|-----------|-------|
| Model | YOLOv8s |
| Training Method | Transfer Learning (Fine-tuning) |
| Number of Classes | 10 |
| Classes | battery, biological, cardboard, clothing, glass, metal, paper, plastic, shoes, trash |
| Dataset | Roboflow Garbage Detection Dataset (Version 7) |
| Total Images | 5,338 |
| Dataset Split | 70% Train / 20% Validation / 10% Test |
| Training Hardware | Google Colab (Tesla T4 GPU) |
| Image Size | 640 × 640 |
| Epochs | 50 |

The complete methodology, preprocessing pipeline, augmentation strategy and evaluation details are available in **training_report.md**.

---

# 📊 Results

> *(Will be updated after training completes.)*

The following evaluation metrics will be included:

- mAP@50
- mAP@50-95
- Precision
- Recall
- Per-class Accuracy
- Confusion Matrix
- Precision-Recall Curve
- F1 Curve
- Validation Predictions

---

# 🏗️ Project Structure

```text
Garbage-Detection-System/
│
├── backend/                  # FastAPI backend
├── frontend/                 # Streamlit frontend
├── models/                   # Trained YOLO model (best.pt)
├── notebooks/                # Google Colab notebook
├── screenshots/              # README screenshots
├── training/
│   ├── data.yaml
│   └── split_dataset.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── training_report.md
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/harshprasad2006/Garbage-Detection-System.git

cd Garbage-Detection-System
```

Create a virtual environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

Install all dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

> *(Commands will be added after backend and frontend development.)*

Backend

```bash
Coming Soon...
```

Frontend

```bash
Coming Soon...
```

---

# 🔌 API Reference

## POST /predict

> *(Detailed API documentation will be added after Phase 7.)*

---

# 🖼️ Dataset & Preprocessing

The project uses the **Roboflow Garbage Detection Dataset (Version 7)** containing **5,338 annotated images** across **10 garbage categories**.

The original dataset distribution supplied by Roboflow was:

- Train : 4929 images
- Validation : 203 images
- Test : 206 images

This distribution was unsuitable because it produced an approximate **92% / 4% / 4%** split.

A custom Python script (`training/split_dataset.py`) was developed to generate a proper:

- 70% Training
- 20% Validation
- 10% Testing

split while preventing **data leakage** by grouping augmented images originating from the same source photograph.

Complete preprocessing details are documented in **training_report.md**.

---

# 🏋️ Model Training

The model is trained using **Ultralytics YOLOv8s** with transfer learning.

Training Environment

- Google Colab
- Tesla T4 GPU
- Python 3.12
- Ultralytics 8.4.110

Training Configuration

- Epochs : 50
- Image Size : 640 × 640
- Batch Size : 16
- Early Stopping Patience : 15

Configured Data Augmentations

- Horizontal Flip
- Mosaic Augmentation
- Brightness Variation
- Saturation / Contrast Variation
- Scale Jitter

The complete notebook will be available in:

```
notebooks/train_yolov8_garbage.ipynb
```

---

# 🧪 Evaluation

> *(Will be updated after training completes.)*

Evaluation will include:

- Validation Metrics
- Loss Curves
- PR Curve
- F1 Curve
- Confusion Matrix
- Detection Samples

---

# 📦 Deployment

> *(Deployment instructions will be added after Hugging Face Spaces deployment.)*

Deployment Stack

- FastAPI
- Streamlit
- Hugging Face Spaces

---

# 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Programming Language | Python |
| Deep Learning | YOLOv8 (Ultralytics) |
| Computer Vision | OpenCV |
| Backend | FastAPI |
| Frontend | Streamlit |
| Training Platform | Google Colab |
| Deployment | Hugging Face Spaces |

---

# 📚 Documentation

Project documentation includes:

- README.md
- training_report.md
- Google Colab Training Notebook
- Source Code

---

# 🤝 Future Improvements

Possible future enhancements include:

- Mobile Application
- Multi-language Interface
- Garbage Segmentation
- Waste Volume Estimation
- Recycling Suggestions
- Edge Deployment using TensorRT
- Drone-based Garbage Detection

---

# 👨‍💻 Author

**Harsh Prasad**

GitHub:
https://github.com/harshprasad2006

---

# ⭐ Acknowledgements

- Ultralytics YOLOv8
- Roboflow Universe
- Google Colab
- FastAPI
- Streamlit

---

**If you found this project useful, consider giving it a ⭐ on GitHub.**