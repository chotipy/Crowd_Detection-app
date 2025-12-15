import streamlit as st
import os
import sys

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

# Add error handling for imports
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import tempfile
    from datetime import datetime
except ImportError as e:
    st.error(f"Missing required package: {e}")
    st.stop()

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


# LOAD MODEL with proper error handling
@st.cache_resource(show_spinner="Loading YOLO model..")
def load_model():
    try:
        model_path = os.path.join("model", "best.pt")

        # Check if model exists
        if not os.path.exists(model_path):
            st.error(f"No model file not found at: {model_path}")
            st.info("Please ensure 'model/best.pt' exists in your repository")
            return None

        # Load model with minimal settings for cloud environment
        model = YOLO(model_path)
        model.to("cpu")

        # Warm up with a small dummy prediction to verify model works
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(dummy, verbose=False, imgsz=640)

        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info(
            "Possible solutions:\n- Ensure ultralytics is installed\n- Check model file integrity\n- Verify sufficient memory"
        )
        return None


# LOADING STATUS
with st.spinner("Initializing model.."):
    model = load_model()

if model is None:
    st.error("⚠️ Cannot proceed without a valid model. Please fix the issues above.")
    st.stop()


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
        return "Sedikit", "low"
    elif count <= 30:
        return "Sedang", "mid"
    else:
        return "Ramai", "high"


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


def save_screenshot(image, prefix):
    try:
        os.makedirs("screenshots", exist_ok=True)
        filename = (
            f"screenshots/{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        cv2.imwrite(filename, image)
        return filename
    except Exception as e:
        st.warning(f"Failed to save screenshot: {e}")
        return None


# SIDEBAR
st.sidebar.title("🎛️ Application Demo")
col1, col2 = st.sidebar.columns(2)
with col1:
    theme_mode = st.radio("Theme", ["Light", "Dark"], index=0)
with col2:
    input_type = st.radio("Input", ["Image", "Video"], index=0)

st.sidebar.divider()

st.sidebar.subheader("🛠 Preprocessing")
st.sidebar.header("Please enable the preprocessing tick for better experience!")
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
st.sidebar.subheader("⚙️ Inference Settings")
st.sidebar.header("These settings affect accuracy, speed, and noise reduction")

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
    file = st.file_uploader("Upload Image", ["jpg", "png", "jpeg"])
    if file:
        try:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                st.error("Failed to decode image. Please upload a valid image file.")
                st.stop()

            frame = preprocess_frame(img.copy(), enable_preprocess, gamma_val, blur_k)

            with st.spinner("Running detection..."):
                results = model.predict(
                    frame,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=max_det,
                    verbose=False,
                    imgsz=640,  # Fixed size for consistency
                )[0]

            output, count = draw_boxes(frame.copy(), results, conf_thres)
            label, badge = classify_crowd(count)

            col1, col2 = st.columns(2)
            with col1:
                st.image(output, channels="BGR")
            with col2:
                st.metric("People Count", count)
                st.markdown(f"**Crowd Level:** {label}")

            path = save_screenshot(output, "image")
            if path:
                st.success(f"Screenshot saved: {path}")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.exception(e)

# VIDEO MODE
else:
    vid = st.file_uploader(
        "Upload Video (Max 10 seconds recommended)", ["mp4", "avi", "mov"]
    )

    if vid:
        try:
            # Save temp input video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid.read())
            tfile.flush()
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)

            if not cap.isOpened():
                st.error("Failed to open video file. Please upload a valid video.")
                os.remove(tfile.name)
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Cloud-safe limits
            MAX_FRAMES = min(300, total_frames)  # ~10 seconds @ 30 FPS
            FRAME_SKIP = 3  # Process every 3rd frame

            # Output video
            out_path = f"output_cloud_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            if not out.isOpened():
                st.error("Failed to create output video writer.")
                cap.release()
                os.remove(tfile.name)
                st.stop()

            st.info(
                f"Processing video: {total_frames} frames, processing every {FRAME_SKIP}rd frame..."
            )
            progress = st.progress(0)

            frame_id = 0
            processed = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1
                if frame_id > MAX_FRAMES:
                    st.warning(f"Stopped at {MAX_FRAMES} frames to prevent timeout")
                    break

                # Skip frames to reduce load
                if frame_id % FRAME_SKIP != 0:
                    out.write(frame)
                    continue

                frame = preprocess_frame(frame, enable_preprocess, gamma_val, blur_k)

                # YOLO inference
                results = model.predict(
                    frame,
                    conf=conf_thres,
                    iou=iou_thres,
                    max_det=min(max_det, 100),
                    device="cpu",
                    verbose=False,
                    imgsz=640,
                )[0]

                output, count = draw_boxes(frame.copy(), results, conf_thres)
                label, _ = classify_crowd(count)

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
                processed += 1

                progress.progress(min(frame_id / MAX_FRAMES, 1.0))

            cap.release()
            out.release()
            os.remove(tfile.name)

            st.success(f"Video processed successfully! ({processed} frames analyzed)")

            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)

                # Clean up output file after displaying
                try:
                    os.remove(out_path)
                except:
                    pass
            else:
                st.error("Output video file was not created")

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.exception(e)
            # Cleanup
            try:
                if "cap" in locals():
                    cap.release()
                if "out" in locals():
                    out.release()
                if "tfile" in locals() and os.path.exists(tfile.name):
                    os.remove(tfile.name)
            except:
                pass

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
        padding: 2rem 1.5rem;
    }}

    section[data-testid="stSidebar"] h1 {{
        font-size: 1.5em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 1rem;
    }}

    section[data-testid="stSidebar"] h2 {{
        font-size: 0.800em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 1rem;
    }}

    section[data-testid="stSidebar"] h3 {{
        font-size: 1.2em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 1rem;
    }}

    section[data-testid="stSidebar"] label {{
        font-size: 0.500em;
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
    }}

    .low {{ 
        background: linear-gradient(135deg, #10b981, #059669);
        color: {text_color};
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }}
    
    .mid {{ 
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: {text_color};
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }}
    
    .high {{ 
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: {text_color};
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
