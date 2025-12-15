import streamlit as st
import os
import tempfile
from datetime import datetime

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# CONFIG
st.set_page_config(page_title="Crowd Detection using YOLOv8", layout="wide")

# HEADER
st.markdown(
    """
    <style>
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }

    .header-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin-bottom: 0.5rem;
    }

    .header-desc {
        font-size: 1.05rem;
        max-width: 720px;
        margin: 0 auto;
        opacity: 0.85;
        line-height: 1.6;
    }
    </style>

    <div class="header-container">
        <div class="header-title">
            Crowd Detection and Density Classification
        </div>
        <div class="header-desc">
            A YOLOv8-based system for detecting, counting, and classifying crowd levels
            in indoor CCTV footage with configurable inference and preprocessing parameters.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# LOAD MODEL
@st.cache_resource(show_spinner="Loading YOLO model..")
def load_model():
    model = YOLO("app/model/best.pt")
    model.to("cpu")
    return model


model = load_model()


# PREPROCESSING
def gamma_correction(image, gamma):
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


def preprocess_frame(frame, enable, gamma, blur_k):
    if not enable:
        return frame
    try:
        frame = gamma_correction(frame, gamma)
        frame = cv2.GaussianBlur(frame, (blur_k, blur_k), 1.0)
        return frame
    except Exception as e:
        st.warning(f"Preprocessing error: {e}")
        return frame


# UTILS
def classify_crowd(count):
    if count <= 3:
        return "Few", "low"
    elif count <= 30:
        return "Medium", "mid"
    else:
        return "Crowded", "high"


def draw_boxes(image, results, conf):
    count = 0
    try:
        for box in results.boxes:
            if int(box.cls[0]) == 0 and float(box.conf[0]) >= conf:
                count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (99, 112, 116), 2)
    except Exception as e:
        st.warning(f"Drawing boxes error: {e}")
    return image, count


# DISPLAY TABLE
st.markdown(
    """
<table style="width:50%; border-collapse:collapse; margin-left:auto; margin-right:auto;">
<tr>
    <th>People Count</th>
    <th>Crowd Label</th>
</tr>
<tr style="background-color:#d1fae5;">
    <td>≤ 3</td>
    <td><b>Few</b></td>
</tr>
<tr style="background-color:#fef3c7;">
    <td>4 – 30</td>
    <td><b>Medium</b></td>
</tr>
<tr style="background-color:#fee2e2;">
    <td>&gt; 30</td>
    <td><b>Crowded</b></td>
</tr>
</table>
""",
    unsafe_allow_html=True,
)


# SIDEBAR
st.sidebar.header("🎛️ Application Demo")
col1, col2 = st.sidebar.columns(2)
with col1:
    theme_mode = st.radio("Theme", ["Light", "Dark"], index=0)
with col2:
    input_type = st.radio("Input", ["Image", "Video"], index=0)

st.sidebar.divider()

st.sidebar.header("🛠 Preprocessing")
st.sidebar.subheader("Please enable the preprocessing tick for better experience!")
enable_preprocess = st.sidebar.checkbox("Enable Preprocessing", True)
gamma_val = st.sidebar.slider(
    "Gamma Correction",
    min_value=0.8,
    max_value=1.6,
    value=1.2,
    step=0.1,
    help="Increase gamma to brighten low light indoor CCTV images",
)

blur_k = st.sidebar.selectbox(
    "Gaussian Blur Kernel",
    [3, 5, 7],
    index=1,
    help="To reduce noise in low light circumstances",
)

st.sidebar.divider()
st.sidebar.header("⚙️ Inference Settings")
st.sidebar.subheader("These settings affect accuracy, speed, and noise reduction")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05,
    help="Minimum confidence score required to display a detection",
)

iou_thres = st.sidebar.slider(
    "IoU Threshold",
    min_value=0.3,
    max_value=0.8,
    value=0.6,
    step=0.05,
    help="Controls how overlapping bounding boxes are merged",
)

max_det = st.sidebar.slider(
    "Max Detections",
    10,
    500,
    300,
    50,
    help="Maximum of object that you want to detect?",
)

# IMAGE MODE
if input_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Decode image
        img = cv2.imdecode(
            np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
        )

        if img is None:
            st.error("Failed to read image file.")
            st.stop()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Preprocess
        frame = preprocess_frame(img.copy(), enable_preprocess, gamma_val, blur_k)

        # Model
        results = model.predict(
            frame,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            device="cpu",
            imgsz=640,
            verbose=False,
        )[0]

        # Draw results
        output, count = draw_boxes(frame.copy(), results, conf_thres)
        label, badge = classify_crowd(count)

        # DISPLAY
        col_img, col_info = st.columns([1.4, 1])

        with col_img:
            st.image(output, channels="BGR", caption="Detection Result")

        with col_info:
            st.subheader("📊 Image Analysis")
            st.metric("People Count", count)

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                    padding:0.75rem 1rem;
                    border-radius:0.75rem;
                    background:rgba(0,0,0,0.03);
                    margin-top:0.75rem;
                    margin-bottom:1.25rem;
                ">
                    <span style="font-weight:600;">Average Crowd Level</span>
                    <span class="badge {badge}">{label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # DOWNLOAD BUTTON
        success, encoded_img = cv2.imencode(".png", output)
        if success:
            st.download_button(
                label="Download Image with Detections",
                data=encoded_img.tobytes(),
                file_name=f"crowd_detection_{timestamp}.png",
                mime="image/png",
                use_container_width=True,
            )
        else:
            st.warning("Failed to prepare image for download.")


# VIDEO MODE
else:
    vid = st.file_uploader(
        "Upload Video (≤10 seconds recommended)", ["mp4", "avi", "mov"]
    )

    if vid:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(vid.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        MAX_FRAMES = 200
        FRAME_SKIP = 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"processed_video_{timestamp}.mp4"

        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        st.info("Processing video...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        frame_id = 0
        total_people = 0
        frame_count = 0
        max_people = 0
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_id > MAX_FRAMES:
                break

            frame_id += 1
            if frame_id % FRAME_SKIP != 0:
                out.write(frame)
                continue

            frame = preprocess_frame(frame, enable_preprocess, gamma_val, blur_k)

            results = model.predict(
                frame,
                conf=conf_thres,
                iou=iou_thres,
                max_det=50,
                device="cpu",
                verbose=False,
                imgsz=640,
            )[0]

            output, count = draw_boxes(frame.copy(), results, conf_thres)
            label, badge = classify_crowd(count)

            # Track statistics
            total_people += count
            frame_count += 1
            max_people = max(max_people, count)

            # Add info overlay to frame
            cv2.rectangle(output, (10, 5), (420, 55), (0, 0, 0), -1)
            cv2.putText(
                output,
                f"People: {count} | Crowd: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            out.write(output)
            last_frame = output.copy()

            # Update progress
            progress_bar.progress(min(frame_id / MAX_FRAMES, 1.0))
            status_text.text(f"Processing frame {frame_id}/{MAX_FRAMES}")

        cap.release()
        out.release()
        os.remove(tfile.name)

        # Calculate average
        avg_people = round(total_people / frame_count) if frame_count > 0 else 0
        avg_label, avg_badge = classify_crowd(avg_people)

        st.success("Video processed successfully!")

        # Show preview of last frame
        if last_frame is not None:
            with st.expander("Preview Last Frame"):
                st.caption(
                    "You can custom your preprocessing and inference on the sidebar by adjusting to the frame here"
                )
                st.image(
                    last_frame,
                    channels="BGR",
                    caption="Last processed frame with detections",
                )

        # Final results with download option
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("Download Processed Video")

            # Read the processed video file
            with open(out_path, "rb") as f:
                video_bytes = f.read()

            # Download button
            st.download_button(
                label="Download Video with Detections",
                data=video_bytes,
                file_name=f"crowd_detection_{timestamp}.mp4",
                mime="video/mp4",
                use_container_width=True,
            )

            st.info("Click the button above to download your processed video")

        with col2:
            st.header("📊 Video Analysis")
            st.metric("Average Count", avg_people)
            st.metric("Total Count", max_people)
            st.metric("Total Frames", frame_count)
            st.markdown(
                f"""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        padding:0.75rem 1rem;
        border-radius:0.75rem;
        background:rgba(0,0,0,0.03);
        margin-top:0.5rem;
        margin-bottom:1.25rem; 
    ">
        <span style="font-weight:600;">Average Crowd Level</span>
        <span class="badge {avg_badge}">{avg_label}</span>
    </div>
    """,
                unsafe_allow_html=True,
            )

            # File info
            file_size = len(video_bytes) / (1024 * 1024)  # Convert to MB
            st.metric("File Size", f"{file_size:.2f} MB")


# THEME STYLING
if theme_mode == "Light":
    bg_color = "#f6eef4"
    card_bg = "rgba(255, 255, 255, 0.75)"
    sidebar_bg = "#eddee9"
    text_color = "#21121d"
    text_secondary = "#633656"
    border_color = "rgba(166, 89, 144, 0.15)"
    accent_color = "#a65990"
    shadow = "0 8px 32px rgba(166, 89, 144, 0.12)"
else:
    bg_color = "#3a2e2c"
    card_bg = "rgba(87, 70, 66, 0.65)"
    sidebar_bg = "#1d1716"
    text_color = "#e9e3e2"
    text_secondary = "#bdaca8"
    border_color = "rgba(189, 172, 168, 0.12)"
    accent_color = "#c99cbc"
    shadow = "0 8px 32px rgba(0, 0, 0, 0.4)"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}
    
    .stApp {{
        background: {bg_color};
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        backdrop-filter: blur(20px);
        border-right: 1px solid {border_color};
    }}

    section[data-testid="stSidebar"] > div {{
        padding: 1rem 1.5rem;
    }}

    section[data-testid="stSidebar"] h1 {{
        font-size: 1em;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: {text_secondary};
    }}

    section[data-testid="stSidebar"] h2 {{
        font-size: 1.2em;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: {text_color};
    }}

    section[data-testid="stSidebar"] h3 {{
        font-size: 0.7em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
    }}

    section[data-testid="stSidebar"] label {{
        font-size: 0.5em;
        font-weight: 500;
        color: {text_color};
    }}

    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stImage"]),
    div[data-testid="stVerticalBlock"] > div:has(div[data-testid="stMetric"]) {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border-radius: 1rem;
        border: 1px solid {border_color};
        box-shadow: {shadow};
        padding: 1.5rem;
        margin: 0.5rem 0;
    }}

    div[data-testid="stMetric"] {{
        background: transparent !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
        border: none !important;
    }}

    div[data-testid="stMetric"] label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: {text_secondary};
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {text_color};
    }}

    h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        color: {text_color};
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }}

    p, .stMarkdown p {{
        font-size: 1rem;
        color: {text_secondary};
        line-height: 1.6;
    }}

    .badge {{
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }}

    .low {{ 
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }}
    
    .mid {{ 
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }}
    
    .high {{ 
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }}

    .stButton > button {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border: 1px solid {border_color};
        border-radius: 0.75rem;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.875rem;
        color: {text_color};
        transition: all 0.2s ease;
    }}

    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: {shadow};
    }}

    div[data-testid="stFileUploader"] {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border-radius: 1rem;
        border: 2px dashed {border_color};
        padding: 2rem;
    }}

    div[data-testid="stFileUploader"] label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: {text_color};
    }}

    .stSuccess {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border-radius: 0.75rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 1rem;
        font-size: 0.875rem;
    }}

    .stCaption {{
        font-size: 0.75rem;
        color: {text_secondary};
        text-align: center;
        margin-top: 2rem;
    }}

    hr {{
        margin: 2rem 0;
        border: none;
        border-top: 1px solid {border_color};
    }}

    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# FOOTER
st.markdown("---")
st.caption("Final Project – Deep Learning | YOLOv8 Crowd Detection | BINUS University")
