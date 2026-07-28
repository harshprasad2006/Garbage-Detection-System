# Training Report

## Dataset Selection

After evaluating multiple public datasets, the Roboflow Garbage Detection dataset was selected for this project.

### Reasons for Selecting This Dataset

- It is already formatted for YOLO object detection.
- Images are pre-labeled.
- No annotation conversion is required.
- Faster training setup.
- Lower preprocessing effort.
- Allows more project time to be spent on model training, API development, frontend development, deployment, and documentation.

### Alternative Dataset Considered

**TACO (Trash Annotations in Context)**

Reasons for not selecting TACO:

- COCO annotations require conversion to YOLO format.
- More preprocessing effort.
- Class imbalance.
- Longer setup time.

### Conclusion

The Roboflow dataset provides the best balance between training quality, development speed, and deployment readiness for this project.