"""
image_detector.py
 
Purpose:
    Standalone script to run garbage detection on a single image from the
    command line. Uses the shared GarbageDetector core (detector.py) so
    behavior is guaranteed identical to the API and Streamlit UI.
 
Usage:
    python image_detector.py --image path/to/image.jpg
    python image_detector.py --image path/to/image.jpg --conf 0.5
    python image_detector.py --image path/to/image.jpg --output result.jpg
 
Output:
    - Saves an annotated image (boxes + labels + confidence drawn on it).
    - Prints detections as JSON to the console.
"""
 
import argparse
import json
import cv2
from detector import GarbageDetector
 
 
def main():
    parser = argparse.ArgumentParser(description="Garbage detection on a single image.")
    parser.add_argument("--image", required=True, help="Path to input image.")
    parser.add_argument("--model", default="../models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    parser.add_argument("--output", default="output.jpg", help="Path to save the annotated image.")
    args = parser.parse_args()
 
    # Load the shared detector (loads model once).
    detector = GarbageDetector(model_path=args.model, conf_threshold=args.conf)
 
    # Run detection - same function the API and UI will call.
    annotated_image, detections = detector.detect(args.image)
 
    # Save the annotated image to disk.
    cv2.imwrite(args.output, annotated_image)
    print(f"Annotated image saved to: {args.output}")
 
    # Print structured detections as JSON (same shape the API will return).
    print("\nDetections:")
    print(json.dumps(detections, indent=2))
 
 
if __name__ == "__main__":
    main()
 
