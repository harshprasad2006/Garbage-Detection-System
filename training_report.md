# Garbage Detection System – Training Report

## 1. Project Overview

This project aims to develop an AI-powered Garbage Detection System capable of detecting and localizing different types of garbage in images, videos, and live webcam streams. The system is built using the Ultralytics YOLOv8 object detection framework and follows a complete machine learning pipeline, including dataset preparation, preprocessing, model training, evaluation, and deployment.

---

## 2. Dataset Selection

The dataset used for this project was downloaded from Roboflow Universe.

**Dataset Name:**
Garbage Detection (Version 7)

**Total Images:**
5338

**Number of Classes:**
10

**Classes:**

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

### Why this dataset was selected

Two candidate datasets were considered for this project:

- **TACO (Trash Annotations in Context)**
- **Roboflow Garbage Detection Dataset**

Although the TACO dataset contains realistic real-world litter images, it uses COCO-format annotations with approximately 60 fine-grained and highly imbalanced classes. Using TACO would require annotation conversion to YOLO format, additional preprocessing, and class regrouping before model training.

The Roboflow Garbage Detection dataset was selected because it is already provided in YOLO format, contains high-quality object annotations, has a manageable dataset size suitable for training on the free Google Colab Tesla T4 GPU, and provides 10 practical material-based garbage categories. This allowed the project effort to focus on model training, evaluation, and deployment instead of dataset engineering.

---

## 3. Original Dataset Analysis

The original Roboflow dataset contained:

| Dataset | Images |
|---------|-------:|
| Training | 4929 |
| Validation | 203 |
| Testing | 206 |
| **Total** | **5338** |

Original distribution:

- Training: **92.3%**
- Validation: **3.8%**
- Testing: **3.9%**

This split was highly imbalanced because the validation and testing sets were too small to provide reliable evaluation metrics.

---

## 4. Dataset Preprocessing

To improve model evaluation, a custom preprocessing pipeline was developed.

A Python script (`split_dataset.py`) was created to:

- Read every image and its corresponding YOLO label.
- Preserve image-label pairs.
- Shuffle the dataset using a fixed random seed.
- Create a new balanced dataset split.

Final dataset distribution:

| Dataset | Images | Percentage |
|---------|-------:|-----------:|
| Training | 3734 | 70% |
| Validation | 1068 | 20% |
| Testing | 536 | 10% |
| **Total** | **5338** | **100%** |

This follows the commonly recommended **70/20/10** split for object detection tasks.

---

## 5. Data Leakage Prevention

The Roboflow export included pre-applied augmentation (such as flips, rotations, crops, color changes, blur, and noise). Images generated from the same original photograph share a common filename prefix before the `.rf.<hash>` portion of the filename.

For example:

```
battery_7_jpg.rf.123abc.jpg
battery_7_jpg.rf.456def.jpg
battery_7_jpg.rf.789ghi.jpg
```

Although these files are different images, they all originate from the same source photograph.

If the dataset were randomly split image-by-image, different augmented versions of the same original image could appear in both the training and testing sets. This is known as **data leakage**, because the model would effectively see almost identical images during training and evaluation, resulting in artificially inflated performance metrics.

To prevent this issue, the custom `split_dataset.py` script performed the following steps:

1. Grouped all images using the original source-image filename prefix.
2. Shuffled groups (instead of individual images) using a fixed random seed (`42`) for reproducibility.
3. Split the groups into **70% training**, **20% validation**, and **10% testing**.

This guarantees that every augmented version of a source image remains in the same subset, ensuring no source image or its augmented variants appear across multiple subsets.

---

## 6. Annotation Format

The dataset uses the standard YOLO object detection annotation format.

Each image has one corresponding text file containing object annotations.

Each annotation follows the format:

```
<class_id> <x_center> <y_center> <width> <height>
```

where:

- **class_id** represents the garbage category.
- **x_center** and **y_center** represent the center coordinates of the bounding box.
- **width** and **height** represent the bounding box dimensions.

All coordinate values are normalized between **0** and **1**, making the annotations independent of image resolution.

---

## 7. Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | YOLOv8s |
| Transfer Learning | Enabled |
| Image Size | 640 × 640 |
| Epochs | 50 |
| Batch Size | 16 |
| Early Stopping Patience | 15 |
| Optimizer | Default Ultralytics Optimizer |
| Hardware | Google Colab Tesla T4 GPU |

The model was initialized using pretrained COCO weights (`yolov8s.pt`) and fine-tuned on the custom garbage detection dataset.

---

## 8. Data Augmentation

The following augmentations were explicitly configured during training:

| Augmentation | Parameter | Value | Purpose |
|--------------|-----------|------:|---------|
| Horizontal Flip | `fliplr` | 0.5 | 50% probability of horizontal flipping to improve viewpoint diversity |
| Vertical Flip | `flipud` | 0.0 | Disabled because garbage objects have realistic upright orientations |
| Mosaic | `mosaic` | 1.0 | Combines four images into one training sample, improving small-object detection |
| Brightness Variation | `hsv_v` | 0.4 | Random brightness variation up to ±40% |
| Saturation Variation | `hsv_s` | 0.7 | Random saturation variation up to ±70% |
| Scale Jitter | `scale` | 0.5 | Random zoom in/out up to 50% for scale invariance |

In addition to the configured augmentations, Ultralytics automatically applies several robustness augmentations (such as Blur, Median Blur, CLAHE, and Grayscale conversion) through the Albumentations library whenever appropriate.

---

## 9. Model Evaluation

Training completed in 1.195 hours on a Tesla T4 GPU. Final validation
metrics (on the 1068-image validation set, 2023 total annotated instances):

### Overall Metrics

| Metric | Value |
|---|---|
| Precision | 0.806 |
| Recall | 0.703 |
| mAP@50 | 0.787 |
| mAP@50-95 | 0.650 |

### Per-Class Performance

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| battery | 0.849 | 0.671 | 0.802 | 0.693 |
| biological | 0.700 | 0.544 | 0.662 | 0.468 |
| cardboard | 0.902 | 0.593 | 0.788 | 0.653 |
| clothing | 0.917 | 0.926 | 0.965 | 0.835 |
| glass | 0.806 | 0.765 | 0.792 | 0.652 |
| metal | 0.747 | 0.830 | 0.864 | 0.778 |
| paper | 0.732 | 0.549 | 0.661 | 0.436 |
| plastic | 0.773 | 0.718 | 0.771 | 0.708 |
| shoes | 0.852 | 0.852 | 0.926 | 0.769 |
| trash | 0.779 | 0.583 | 0.642 | 0.508 |

Supporting artifacts (`results.png`, `confusion_matrix.png`,
`confusion_matrix_normalized.png`, `PR_curve.png`, `F1_curve.png`, and
validation prediction samples) are stored in `training/results/`.

### Observations

- **Strongest classes**: `clothing` (mAP50 = 0.965) and `shoes` (mAP50 =
  0.926) achieved the best results, likely due to consistent shape and
  visual distinctiveness relative to other classes.
- **Weakest classes**: `biological`, `trash`, and `paper` show the lowest
  recall (0.54-0.58), meaning the model misses a meaningful fraction of
  these objects. This is consistent with the inherent visual ambiguity of
  these categories — "trash" is a broad catch-all with high visual
  variance, and "paper"/"biological" waste can visually overlap with other
  classes or blend into backgrounds.
- Precision remains reasonably high across all classes (0.70-0.92),
  indicating the model rarely produces confident false positives; its main
  weakness is under-detection (missed objects) rather than misclassification.

---

## 10. Conclusion

The fine-tuned YOLOv8s model achieved an overall mAP@50 of 0.787 and
mAP@50-95 of 0.650 after 50 epochs of training on a leakage-safe,
70/20/10 split of the Roboflow Garbage Detection dataset. Precision (0.806)
indicates reliable detections with few false positives, while recall (0.703)
shows room for improvement, particularly for visually ambiguous classes
such as biological waste, paper, and general trash.

**Possible future improvements:**
- Collecting additional training images for the weaker classes
  (biological, trash, paper) to address class imbalance and visual
  ambiguity.
- Training for more epochs (80-100) with a learning-rate schedule tuned
  specifically for the weaker classes.
- Applying class-weighted loss to give under-performing classes more
  weight during training.
- Experimenting with a larger backbone (YOLOv8m) if additional GPU budget
  is available, trading inference speed for potential accuracy gains.

Overall, the model meets the project's functional requirements and is
suitable for integration into the full detection pipeline (API, UI, and
deployment).
This section will summarize the final model performance, discuss strengths and limitations of the trained detector, and suggest future improvements such as increasing dataset diversity, experimenting with larger YOLO models, hyperparameter tuning, and deployment as a real-time web application.