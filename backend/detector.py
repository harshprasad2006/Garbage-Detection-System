"""
detector.py

Purpose:
    Single shared detection function used everywhere in this project -
    image inference, video inference, webcam inference, the FastAPI
    /predict endpoint, and the Streamlit UI all call THIS function.

    Writing detection logic once here (instead of duplicating it in every
    script) guarantees the API and the UI always behave identically.

Usage:
    from detector import GarbageDetector

    detector = GarbageDetector(model_path="../models/best.pt")
    annotated_image, results = detector.detect(image)
"""

from ultralytics import YOLO
import cv2
import numpy as np


class GarbageDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.4):
        """
        Loads the trained YOLOv8s model ONCE when the class is created.
        Reloading the model on every single prediction would be extremely
        slow - this is why model loading lives in __init__, not in detect().

        model_path: path to best.pt (the fine-tuned weights from Phase 4)
        conf_threshold: minimum confidence required to report a detection.
                        Default 0.4 - chosen to balance catching real
                        detections (some classes have lower recall) against
                        showing too many low-confidence false positives.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names  # dict: {0: 'battery', 1: 'biological', ...}

    def detect(self, image):
        """
        Runs detection on a single image.

        Input:
            image: can be a file path (str), a PIL Image, or a numpy array
                   (e.g. a frame read from OpenCV/webcam/video).

        Returns:
            annotated_image (numpy array, BGR): the image with bounding
                boxes, class labels, and confidence scores drawn on it -
                ready to display or save.
            detections (list of dict): structured results, e.g.
                [
                    {
                        "class": "plastic",
                        "confidence": 0.87,
                        "bounding_box": {"x1": 34, "y1": 12, "x2": 210, "y2": 190}
                    },
                    ...
                ]
                This exact structure is what the FastAPI /predict endpoint
                (Phase 7) will return as JSON.
        """
        # Run YOLO inference. conf= filters out low-confidence boxes
        # directly at the model level (faster than filtering afterward).
        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            verbose=False,
        )

        result = results[0]  # single image -> single result object

        # Ultralytics can draw boxes for us directly - this returns a
        # numpy array (BGR, same format OpenCV uses) with boxes+labels
        # already rendered. Saves us from manually drawing rectangles.
        annotated_image = result.plot()

        # Build our own clean structured list (JSON-serializable) instead
        # of returning Ultralytics' internal objects directly - this keeps
        # the output stable and simple for both the API and UI to consume.
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # pixel coordinates

            detections.append({
                "class": self.class_names[class_id],
                "confidence": round(confidence, 4),
                "bounding_box": {
                    "x1": round(x1, 2),
                    "y1": round(y1, 2),
                    "x2": round(x2, 2),
                    "y2": round(y2, 2),
                }
            })

        return annotated_image, detections
