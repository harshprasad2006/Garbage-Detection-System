"""
video_detector.py

Purpose:
    Standalone script to run garbage detection on a video file, frame by
    frame, using the shared GarbageDetector core (detector.py) - so
    behavior is identical to the image script, the API, and the UI.

Usage:
    python video_detector.py --video path/to/input.mp4
    python video_detector.py --video path/to/input.mp4 --output result.mp4 --skip_frames 2

Output:
    - Saves an annotated output video (boxes + labels + confidence drawn
      on every processed frame).
    - Prints a summary of total detections per class across the whole video.
"""

import argparse
import cv2
from collections import defaultdict
from detector import GarbageDetector


def main():
    parser = argparse.ArgumentParser(description="Garbage detection on a video file.")
    parser.add_argument("--video", required=True, help="Path to input video.")
    parser.add_argument("--model", default="../models/best.pt", help="Path to trained YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    parser.add_argument("--output", default="output_video.mp4", help="Path to save the annotated video.")
    parser.add_argument("--skip_frames", type=int, default=1,
                         help="Process every Nth frame (1 = every frame, 2 = every other frame, etc.). "
                              "Skipped frames reuse the previous frame's detections. Default: 1.")
    args = parser.parse_args()

    detector = GarbageDetector(model_path=args.model, conf_threshold=args.conf)

    # Open the input video.
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {args.video}")

    # Read video properties so the output video matches input format.
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # VideoWriter to save the annotated output.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Tracks total detections per class across the entire video - this
    # feeds directly into the "detection summary" feature required in the
    # Streamlit UI (Phase 8).
    class_counts = defaultdict(int)

    frame_index = 0
    last_annotated_frame = None

    print(f"Processing video: {args.video} ({total_frames} frames, {fps:.1f} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # end of video

        if frame_index % args.skip_frames == 0:
            # Run detection on this frame.
            annotated_frame, detections = detector.detect(frame)
            last_annotated_frame = annotated_frame

            for det in detections:
                class_counts[det["class"]] += 1
        else:
            # Reuse the last annotated frame instead of re-running inference,
            # to save processing time when --skip_frames > 1.
            annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame

        writer.write(annotated_frame)
        frame_index += 1

        # Simple progress indicator every 30 frames.
        if frame_index % 30 == 0:
            print(f"  Processed {frame_index}/{total_frames} frames...")

    cap.release()
    writer.release()

    print(f"\nDone. Annotated video saved to: {args.output}")
    print("\nDetection summary (total instances per class across video):")
    if class_counts:
        for class_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {count}")
    else:
        print("  No detections found.")


if __name__ == "__main__":
    main()