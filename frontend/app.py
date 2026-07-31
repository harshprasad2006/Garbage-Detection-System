"""
frontend/app.py

Streamlit frontend for the Garbage Detection System.
Custom-themed (industrial/sanitation palette + recycling-bin color-coded
class badges) instead of default Streamlit styling.

Run locally:
    cd frontend
    streamlit run app.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "backend"))

import streamlit as st
import cv2
import numpy as np
from PIL import Image

from detector import GarbageDetector

st.set_page_config(
    page_title="Garbage Detection System",
    page_icon="🗑️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Class -> color mapping, inspired by real-world recycling bin conventions.
# This is the visual "signature" of the app: color-coding matches what
# these materials are actually sorted into in real waste-management systems.
# ---------------------------------------------------------------------------
CLASS_COLORS = {
    "battery":    "#E63946",  # hazardous waste - red
    "biological": "#8B5E3C",  # organic/compost - brown
    "cardboard":  "#C69B5D",  # kraft tan
    "clothing":   "#8E5DB0",  # textile recycling - purple
    "glass":      "#2E8B57",  # glass bin - green
    "metal":      "#8A94A6",  # steel gray
    "paper":      "#3B82C4",  # paper bin - blue
    "plastic":    "#2BB3A3",  # plastic recycling - teal
    "shoes":      "#A66DD4",  # violet (textile-adjacent)
    "trash":      "#6B6E64",  # general waste - charcoal gray
}


def inject_custom_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background-color: #1B1E1A;
        color: #F2F1EA;
    }

    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: #F2F1EA !important;
    }

    /* Hazard-stripe divider under the header - grounded in sanitation/
       industrial visual language (warning stripes on bins/trucks). */
    .hazard-divider {
        height: 6px;
        margin: 0.5rem 0 1.5rem 0;
        background: repeating-linear-gradient(
            45deg,
            #D7E600,
            #D7E600 14px,
            #1B1E1A 14px,
            #1B1E1A 28px
        );
        border-radius: 3px;
    }

    .app-subtitle {
        color: #9A9C8E;
        font-size: 0.95rem;
        margin-top: -0.6rem;
    }

    /* Tabs styled as physical toggle buttons rather than default underline tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #262922;
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: transparent;
        border-radius: 8px;
        color: #9A9C8E;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #D7E600 !important;
        color: #1B1E1A !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #262922;
        border: 1px dashed #4A4D45;
        border-radius: 10px;
        padding: 0.5rem;
    }

    /* Detection result cards */
    .detection-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: #262922;
        border-left: 6px solid var(--card-color);
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 8px;
    }
    .detection-card .class-name {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        text-transform: capitalize;
        color: #F2F1EA;
    }
    .detection-card .confidence {
        font-family: 'IBM Plex Mono', monospace;
        color: #D7E600;
        font-size: 0.95rem;
    }
    .detection-card .bbox {
        font-family: 'IBM Plex Mono', monospace;
        color: #9A9C8E;
        font-size: 0.8rem;
    }

    .swatch {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)


def render_detection_cards(detections):
    """Renders each detection as a styled, color-coded card instead of a plain table."""
    if not detections:
        st.info("No garbage objects detected above the confidence threshold.")
        return

    for det in detections:
        color = CLASS_COLORS.get(det["class"], "#D7E600")
        bbox = det["bounding_box"]
        st.markdown(f"""
        <div class="detection-card" style="--card-color:{color}">
            <div>
                <span class="swatch" style="background-color:{color}"></span>
                <span class="class-name">{det['class']}</span><br/>
                <span class="bbox">({bbox['x1']:.0f}, {bbox['y1']:.0f}) → ({bbox['x2']:.0f}, {bbox['y2']:.0f})</span>
            </div>
            <div class="confidence">{det['confidence']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


def render_legend():
    """Small legend showing the class -> color mapping, reinforcing the recycling-bin theme."""
    swatches_html = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;font-size:0.8rem;color:#9A9C8E;">'
        f'<span class="swatch" style="background-color:{color}"></span>{cls}</span>'
        for cls, color in CLASS_COLORS.items()
    )
    st.markdown(f'<div style="margin:0.5rem 0 1.5rem 0;">{swatches_html}</div>', unsafe_allow_html=True)


inject_custom_css()

@st.cache_resource
def load_detector():
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "best.pt"
    )
    return GarbageDetector(model_path=model_path, conf_threshold=0.4)


detector = load_detector()

if "detection_history" not in st.session_state:
    st.session_state.detection_history = []

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🗑️ Garbage Detection System")
st.markdown(
    '<p class="app-subtitle">Fine-tuned YOLOv8s model · image, video, and webcam garbage detection</p>',
    unsafe_allow_html=True,
)
st.markdown('<div class="hazard-divider"></div>', unsafe_allow_html=True)

render_legend()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_image, tab_video, tab_webcam, tab_history = st.tabs(
    ["Image Detection", "Video Detection", "Webcam Detection", "Detection History"]
)

# ---------------------------------------------------------------------------
# IMAGE DETECTION TAB
# ---------------------------------------------------------------------------
with tab_image:
    st.subheader("Upload an image to detect garbage")

    uploaded_image = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        key="image_uploader",
    )

    if uploaded_image is not None:
        file_bytes = np.frombuffer(uploaded_image.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            annotated_bgr, detections = detector.detect(image_bgr)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Image**")
            st.image(Image.open(uploaded_image), use_container_width=True)
        with col2:
            st.markdown("**Detected Objects**")
            st.image(annotated_rgb, use_container_width=True)

        st.markdown("### Detection Results")
        render_detection_cards(detections)

        st.session_state.detection_history.append({
            "source": "Image",
            "filename": uploaded_image.name,
            "detections": detections,
        })

# ---------------------------------------------------------------------------
# VIDEO DETECTION TAB
# ---------------------------------------------------------------------------
with tab_video:
    st.subheader("Upload a video to detect garbage frame-by-frame")

    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov"],
        key="video_uploader",
    )

    skip_frames = st.slider(
        "Process every Nth frame (higher = faster, lower quality tracking)",
        min_value=1, max_value=10, value=2,
    )

    if uploaded_video is not None:
        if st.button("Run Detection on Video"):
            import tempfile
            from collections import defaultdict

            # OpenCV needs a real file path to open a video - can't read
            # directly from the uploaded file's in-memory bytes like we did
            # for images. So we write it to a temp file first.
            input_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            input_temp.write(uploaded_video.read())
            input_temp.close()

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(input_temp.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # avc1 (H.264) is far more reliably playable in-browser via
            # st.video() than mp4v, which some browsers refuse to decode.
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            class_counts = defaultdict(int)
            progress_bar = st.progress(0, text="Processing video...")

            frame_index = 0
            last_annotated_frame = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_index % skip_frames == 0:
                    annotated_frame, detections = detector.detect(frame)
                    last_annotated_frame = annotated_frame
                    for det in detections:
                        class_counts[det["class"]] += 1
                else:
                    annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame

                writer.write(annotated_frame)
                frame_index += 1

                if total_frames > 0:
                    progress_bar.progress(
                        min(frame_index / total_frames, 1.0),
                        text=f"Processing frame {frame_index}/{total_frames}...",
                    )

            cap.release()
            writer.release()
            progress_bar.empty()

            st.success("Video processing complete.")
            st.video(output_path)

            st.markdown("### Detection Summary")
            if class_counts:
                summary_detections = [
                    {"class": cls, "confidence": 1.0, "bounding_box": {"x1": 0, "y1": 0, "x2": 0, "y2": 0}}
                    for cls in class_counts
                ]
                # Reuse the same card style, but show counts instead of confidence.
                for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                    color = CLASS_COLORS.get(cls, "#D7E600")
                    st.markdown(f"""
                    <div class="detection-card" style="--card-color:{color}">
                        <div>
                            <span class="swatch" style="background-color:{color}"></span>
                            <span class="class-name">{cls}</span>
                        </div>
                        <div class="confidence">{count} detections</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No garbage objects detected in this video.")

            st.session_state.detection_history.append({
                "source": "Video",
                "filename": uploaded_video.name,
                "detections": [{"class": cls, "count": count} for cls, count in class_counts.items()],
            })

# ---------------------------------------------------------------------------
# WEBCAM DETECTION TAB
# ---------------------------------------------------------------------------
with tab_webcam:
    st.subheader("Capture a photo from your webcam to detect garbage")
    st.caption(
        "Uses your browser's camera to capture a snapshot - reliable across "
        "local and cloud deployment. For continuous live-feed detection, "
        "run backend/webcam_detector.py locally."
    )

    camera_photo = st.camera_input("Take a photo", key="webcam_input")

    if camera_photo is not None:
        file_bytes = np.frombuffer(camera_photo.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running detection..."):
            annotated_bgr, detections = detector.detect(image_bgr)

        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        st.markdown("**Detected Objects**")
        st.image(annotated_rgb, use_container_width=True)

        st.markdown("### Detection Results")
        render_detection_cards(detections)

        st.session_state.detection_history.append({
            "source": "Webcam",
            "filename": "webcam_capture.jpg",
            "detections": detections,
        })


# ---------------------------------------------------------------------------
# DETECTION HISTORY TAB
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Detection History")

    history = st.session_state.detection_history

    if len(history) == 0:
        st.info("No detections yet. Run image, video, or webcam detection first.")
    else:
        st.success(f"Total Detection Sessions: {len(history)}")

        if st.button("🗑️ Clear History"):
            st.session_state.detection_history = []
            st.rerun()

        st.markdown("---")

        for index, item in enumerate(reversed(history), start=1):
            st.markdown(
                f"""
### Session {len(history)-index+1}
**Source:** {item["source"]}  
**File:** {item["filename"]}
"""
            )

            detections = item["detections"]

            if not detections:
                st.info("No garbage detected.")
                st.markdown("---")
                continue

            if "confidence" in detections[0]:
                render_detection_cards(detections)
            elif "count" in detections[0]:
                for det in detections:
                    color = CLASS_COLORS.get(det["class"], "#D7E600")
                    st.markdown(
                        f"""
<div class="detection-card" style="--card-color:{color}">
    <div>
        <span class="swatch" style="background-color:{color}"></span>
        <span class="class-name">{det["class"]}</span>
    </div>
    <div class="confidence">{det["count"]} detections</div>
</div>
""",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
