# Garbage Detection System – Training Report

## 1. Project Overview

This project aims to develop an AI-powered Garbage Detection System capable of detecting and localizing different types of garbage in images, videos, and live webcam streams. The system is built using the Ultralytics YOLOv8 object detection framework and follows a complete machine learning pipeline, including dataset preparation, preprocessing, model training, evaluation, and deployment.

---

## 2. Dataset Selection

The dataset used for this project was downloaded from Roboflow Universe.

Dataset Name:
Garbage Detection (Version 7)

Total Images:
5338

Number of Classes:
10

Classes:

- Battery
- Biological
- Cardboard
- Clothing
- Glass
- Metal
- Paper
- Plastic
- Shoes
- Trash

The dataset already contained object annotations in YOLO format, making it suitable for direct training using Ultralytics YOLOv8.

---

## 3. Original Dataset Analysis

The original dataset contained:

Training Images:
4929

Validation Images:
203

Testing Images:
206

This corresponds approximately to:

Training:
92.3%

Validation:
3.8%

Testing:
3.9%

Such an imbalanced split is not recommended because the validation and testing sets are too small to provide reliable model evaluation.

---

## 4. Dataset Preprocessing

To improve evaluation quality, a custom preprocessing pipeline was created.

A Python script (`split_dataset.py`) was developed to:

- Read every image and label.
- Preserve image-label pairs.
- Randomly shuffle using a fixed seed.
- Create a new dataset split.

Final dataset distribution:

Training Images:
3734 (70%)

Validation Images:
1068 (20%)

Testing Images:
536 (10%)

This follows the commonly recommended 70/20/10 split for object detection datasets.

---

## 5. Data Leakage Prevention

A custom dataset splitting strategy was used to avoid data leakage.

The script ensured:

- Images were assigned to only one subset.
- Labels remained paired with their corresponding images.
- The same image never appeared in multiple subsets.

This produces a more reliable evaluation of model performance.

---

## 6. Annotation Format

The dataset uses the YOLO annotation format.

Each image has a corresponding text file.

Each annotation follows:

<class_id> <x_center> <y_center> <width> <height>

where all coordinates are normalized between 0 and 1.

---

## 7. Training Configuration

Model:
YOLOv8s

Image Size:
640 × 640

Epochs:
50

Batch Size:
16

Transfer Learning:
Enabled

Hardware:
Google Colab Tesla T4 GPU

---

## 8. Data Augmentation

The following augmentations were applied during training:

- Horizontal Flip
- Mosaic Augmentation
- Brightness Variation
- Saturation Variation
- Scale Jitter

These augmentations improve model robustness and generalization.

---

## 9. Model Evaluation

(To be completed after training finishes.)

This section will include:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix
- Precision-Recall Curve
- F1 Curve

---

## 10. Conclusion

(To be completed after training finishes.)

This section will summarize the overall model performance and discuss possible future improvements.