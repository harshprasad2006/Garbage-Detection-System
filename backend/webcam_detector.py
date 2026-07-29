"""
webcam_detector.py

Purpose:
    Standalone script for real-time garbage detection using a local
    webcam. Uses the same shared GarbageDetector core (detector.py) as
    the image and video scripts, so detection behavior is identical.

    This script is for LOCAL testing/demo only - it opens a live OpenCV
    window, which only works on a machine with a physical webcam and a
    display. The deployed app (Hugging Face Spaces, Phase 9) will instead
    use Streamlit's browser-based webcam component (Phase 8), since
    Spaces containers have no physical camera or display of their own.

Usage:
    python webcam_detector.py
    python webcam_detector.py --conf 0.5

Controls:
    Press 'q' to quit the live window.
"""

import argparse
import time
import cv2
from detector import GarbageDetector


def main():
    parser = argparse.ArgumentParser(description="Real-time garbage detection via webcam.")
    parser.add_argument("--model", default="../models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    parser.add_argument("--camera_index", type=int, default=0, help="Webcam device index (default: 0).")
    args = parser.parse_args()

    detector = GarbageDetector(model_path=args.model, conf_threshold=args.conf)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise ValueError(f"Could not open webcam at index {args.camera_index}.")

    print("Webcam started. Press 'q' in the window to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam. Exiting.")
            break

        # Run detection on the live frame - identical function used by
        # the image and video scripts.
        annotated_frame, detections = detector.detect(frame)

        # Calculate and overlay a live FPS counter - useful to gauge
        # real-time performance on your specific machine (CPU vs GPU).
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time
        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Garbage Detection - Live Webcam (press 'q' to quit)", annotated_frame)

        # Exit loop when 'q' is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received. Closing webcam.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
