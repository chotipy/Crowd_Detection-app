import streamlit as st
import os

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
from datetime import datetime

# CONFIG
st.set_page_config(page_title="Crowd Detection using YOLOv8", layout="wide")

# THEME
theme_mode = st.sidebar.radio("Theme Mode", ["Light", "Dark"])

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

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: {sidebar_bg};
        backdrop-filter: blur(20px);
        border-right: 1px solid {border_color};
    }}

    section[data-testid="stSidebar"] > div {{
        padding: 2rem 1.5rem;
    }}

    section[data-testid="stSidebar"] h2 {{
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 1rem;
    }}

    section[data-testid="stSidebar"] label {{
        font-size: 0.875rem;
        font-weight: 500;
        color: {text_color};
    }}

    /* Card Styling */
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

    /* Metric Styling */
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

    /* Title Styling */
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

    /* Badge Styling */
    .badge {{
        padding: 0.5rem 1.25rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 1rem;
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

    /* Button Styling */
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

    /* File Uploader */
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

    /* Success Message */
    .stSuccess {{
        background: {card_bg};
        backdrop-filter: blur(20px);
        border-radius: 0.75rem;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 1rem;
        font-size: 0.875rem;
    }}

    /* Caption */
    .stCaption {{
        font-size: 0.75rem;
        color: {text_secondary};
        text-align: center;
        margin-top: 2rem;
    }}

    /* Divider */
    hr {{
        margin: 2rem 0;
        border: none;
        border-top: 1px solid {border_color};
    }}

    /* Hide Streamlit branding */
    footer {{ visibility: hidden; }}
    #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# HEADER
st.title("Crowd Detection & Classification")
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
input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

st.sidebar.divider()

st.sidebar.header("🛠 Preprocessing")
enable_preprocess = st.sidebar.checkbox("Enable Preprocessing", True)
gamma_val = st.sidebar.slider(
    "Gamma", 0.8, 1.6, 1.2, 0.1, help=("To help brighten low light picture")
)  # Untuk mengatur seberapa cerah gambar
blur_k = st.sidebar.selectbox(
    "Gaussian Blur Kernel",
    [3, 5, 7],
    index=1,
    help=("To reduce noise in low light circumstances"),
)  # Untuk reduce noise di low light karena indoor

st.sidebar.header("⚙️ Inference Settings")
st.caption(
    "Adjust how the model detects people. "
    "These settings affect accuracy, speed, and noise reduction."
)
conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    0.9,
    0.4,
    0.05,
    help=(
        "Minimum confidence score required to display a detection.\n"
        "Low value = more detections but may include false positives\n"
        "High value = fewer but usually more reliable detections"
    ),
)  # Atur sensitivitas detection

iou_thres = st.sidebar.slider(
    "IoU Threshold",
    0.3,
    0.8,
    0.6,
    0.05,
    help=(
        "Controls how overlapping bounding boxes are merged.\n"
        "Lower IoU = stricter suppression"
        "Higher IoU = allows more overlapping boxes"
    ),
)  # Atur overlap antar bounding box

max_det = st.sidebar.slider(
    "Max Detections",
    50,
    500,
    300,
    50,
    help=("Maximum of object that you want to detect?"),
)  # Batas maksimum objek yang dapat terdeteksi

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
        st.success(f"Screenshot saved: {path}")

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
            st.success(f"Screenshot saved: {path}")

# FOOTER
st.markdown("---")
st.caption("Final Project – Deep Learning | YOLOv8 Crowd Detection | BINUS University")
