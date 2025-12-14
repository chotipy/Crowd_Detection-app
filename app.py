import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from datetime import datetime

# CONFIG
st.set_page_config(page_title="Crowd Detection using YOLOv8", layout="wide")

# THEME
theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])

# THEME STYLING
if theme_mode == "Light":
    bg_gradient = "rgba(189,212,231,0.35), rgba(170,185,207,0.35)"
    sidebar_bg = "rgba(134,147,171,0.45)"
    text_color = "#212227"
else:
    bg_gradient = "rgba(33,34,39,0.85), rgba(33,34,39,0.85)"
    sidebar_bg = "rgba(33,34,39,0.95)"
    text_color = "#ffffff"

st.markdown(
    f"""
    <style>
    .stApp {{
        background: linear-gradient(135deg, {bg_gradient});
        font-family: "Segoe UI", sans-serif;
        color: {text_color};
    }}

    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        color: white;
    }}

    section[data-testid="stSidebar"] * {{
        color: white !important;
    }}

    div[data-testid="stMetric"],
    div[data-testid="stImage"],
    div[data-testid="stVerticalBlock"] > div {{
        background: rgba(255,255,255,0.35);
        backdrop-filter: blur(12px);
        border-radius: 0.25rem;
        border: 1px solid rgba(255,255,255,0.25);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }}

    .badge {{
        padding: 6px 12px;
        border-radius: 0.25rem;
        font-weight: 600;
        display: inline-block;
    }}

    .low {{ background: #4CAF50; color: white; }}
    .mid {{ background: #FFC107; color: black; }}
    .high {{ background: #F44336; color: white; }}

    footer {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# HEADER
st.title("Crowd Detection & Crowd Level Classification")
st.write(
    "YOLOv8-based crowd detection system for indoor CCTV with customizable "
    "inference and preprocessing parameters."
)


# LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO(os.path.join("model", "best.pt"))


model = load_model()


# PREPROCESSING
def gamma_correction(image, gamma):  # Mencerahkan gambar yang gelap
    inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)


# CALL PREPROCESSING
def preprocess_frame(frame, enable, gamma, blur_k):
    if not enable:
        return frame
    frame = gamma_correction(frame, gamma)
    frame = cv2.GaussianBlur(
        frame, (blur_k, blur_k), 1.0
    )  # Mengurangi noise karena kita pakai lighting indoor
    return frame


# UTILS
def classify_crowd(count):  # Mapping jumlah orang
    if count <= 3:
        return "Sedikit", "low"
    elif count <= 30:
        return "Sedang", "mid"
    else:
        return "Ramai", "high"


def draw_boxes(image, results, conf):  # Output kotak pada image
    count = 0
    for box in results.boxes:
        if int(box.cls[0]) == 0 and float(box.conf[0]) >= conf:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (99, 112, 116), 2)
    return image, count


def save_screenshot(image, prefix):  # Save screenshot
    os.makedirs("screenshots", exist_ok=True)
    filename = f"screenshots/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(filename, image)
    return filename


# SIDEBAR
st.sidebar.header("⚙️ Inference Settings")
conf_thres = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.4, 0.05
)  # Atur sensitivitas detection
iou_thres = st.sidebar.slider(
    "IoU Threshold", 0.3, 0.8, 0.6, 0.05
)  # Atur overlap antar bounding box
max_det = st.sidebar.slider(
    "Max Detections", 50, 500, 300, 50
)  # Batas maksimum objek yang dapat terdeteksi

st.sidebar.divider()
st.sidebar.header("🛠 Preprocessing")
enable_preprocess = st.sidebar.checkbox("Enable Preprocessing", True)
gamma_val = st.sidebar.slider(
    "Gamma", 0.8, 1.6, 1.2, 0.1
)  # Untuk mengatur seberapa cerah gambar
blur_k = st.sidebar.selectbox(
    "Gaussian Blur Kernel", [3, 5, 7], index=1
)  # Untuk reduce noise di low light karena indoor

st.sidebar.divider()
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

# IMAGE MODE
if input_type == "Image":
    file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        frame = preprocess_frame(img.copy(), enable_preprocess, gamma_val, blur_k)

        results = model.predict(
            frame, conf=conf_thres, iou=iou_thres, max_det=max_det, verbose=False
        )[0]

        output, count = draw_boxes(frame.copy(), results, conf_thres)
        label, badge = classify_crowd(count)

        col1, col2 = st.columns(2)
        with col1:
            st.image(output, channels="BGR")
        with col2:
            st.metric("People Count", count)
            st.markdown(
                f'<span class="badge {badge}">{label}</span>', unsafe_allow_html=True
            )

        path = save_screenshot(output, "image")
        st.success(f"Your screenshot is successfully saved as: {path}")

# VIDEO MODE
else:
    vid = st.file_uploader("Upload Video", ["mp4", "avi", "mov"])
    if vid:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(vid.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        last_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame, enable_preprocess, gamma_val, blur_k)
            results = model.predict(
                frame, conf=conf_thres, iou=iou_thres, max_det=max_det, verbose=False
            )[0]

            output, count = draw_boxes(frame.copy(), results, conf_thres)
            label, _ = classify_crowd(count)
            last_frame = output.copy()

            cv2.putText(
                output,
                f"People: {count} | Crowd: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            stframe.image(output, channels="BGR")

        cap.release()
        os.remove(tfile.name)

        if last_frame is not None:
            path = save_screenshot(last_frame, "video")
            st.success(f"Your screenshot is successfully saved as: {path}")

# FOOTER
st.markdown("---")
st.caption("Final Project – Deep Learning | YOLOv8 Crowd Detection | BINUS University")
