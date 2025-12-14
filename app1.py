import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# CONFIG
st.set_page_config(page_title="Crowd Detection using YOLOv8", layout="wide")

# HEADER
st.title("👥 Crowd Detection & Crowd Level Classification")
st.write(
    "A Deep Learning application using YOLOv8 to detect people and "
    "classify crowd levels from indoor CCTV footage."
)


# LOAD MODEL
@st.cache_resource
def load_model():
    model_path = os.path.join("model", "best.pt")
    return YOLO(model_path)


model = load_model()


# PREPROCESSING
def gamma_correction(image, gamma=1.2):
    # Mencerahkan gambar yang gelap
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(
        "uint8"
    )

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
# Mapping jumlah orang
def classify_crowd(count):
    if count <= 3:
        return "Sedikit"
    elif count <= 30:
        return "Sedang"
    else:
        return "Ramai"


# Gambar kotak di image
def draw_boxes(image, results, conf_thres):
    count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf >= conf_thres:
            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return image, count


# SIDEBAR
st.sidebar.header("⚙️ Customize Inference Settings")

# Atur sensitivitas detection
conf_thres = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)

# Atur overlap antar bounding box
iou_thres = st.sidebar.slider("IoU Threshold", 0.3, 0.8, 0.6, 0.05)

# Batas maksimum objek yang dapat terdeteksi
max_det = st.sidebar.slider("Max Detections", 50, 500, 300, 50)

st.sidebar.divider()

st.sidebar.header("🛠 Customize Preprocessing")

enable_preprocess = st.sidebar.checkbox("Enable Preprocessing", value=True)

# Untuk mengatur seberapa cerah gambar
gamma_value = st.sidebar.slider("Gamma Correction", 0.8, 1.6, 1.2, 0.1)

# Untuk reduce noise di low light karena indoor
blur_kernel = st.sidebar.selectbox("Gaussian Blur Kernel", [3, 5, 7], index=1)

st.sidebar.divider()

input_type = st.sidebar.radio(  # Bisa pilih input disini mau image atau video dari CCTV
    "Input Type", ["Image", "Video"]
)

# IMAGE MODE
if input_type == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image (jpg / png)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        preprocessed = preprocess_frame(
            image.copy(), enable_preprocess, gamma_value, blur_kernel
        )

        results = model.predict(
            source=preprocessed,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            verbose=False,
        )[0]

        output_img, people_count = draw_boxes(preprocessed.copy(), results, conf_thres)

        crowd_level = classify_crowd(people_count)

        col1, col2 = st.columns(2)

        with col1:
            st.image(output_img, channels="BGR", caption="Detection Result")

        with col2:
            st.subheader("📊 Result")
            st.metric("People Count", people_count)
            st.metric("Crowd Level", crowd_level)

# VIDEO MODE
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = preprocess_frame(frame, enable_preprocess, gamma_value, blur_kernel)

            results = model.predict(
                source=frame,
                conf=conf_thres,
                iou=iou_thres,
                max_det=max_det,
                verbose=False,
            )[0]

            output_frame, people_count = draw_boxes(frame.copy(), results, conf_thres)

            crowd_level = classify_crowd(people_count)

            cv2.putText(
                output_frame,
                f"People: {people_count} | Crowd: {crowd_level}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

            stframe.image(output_frame, channels="BGR")

        cap.release()
        os.remove(tfile.name)

# FOOTER
st.markdown("---")
st.caption(
    "Final Project – Deep Learning | " "YOLOv8 Crowd Detection | BINUS University"
)
