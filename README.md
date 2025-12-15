# Crowd Detection & Density Classification using YOLOv8 - Complete Code Explanation

## Overview

This project implements a **complete crowd detection and crowd density classification system** using **YOLOv8** integrated into a **Streamlit web application**.
The system detects people in **indoor CCTV images and short video footage**, counts the number of detected individuals, and classifies crowd density levels into **Sedikit (Low), Sedang (Medium), and Ramai (High)**.

🔗 **Live Demo (Streamlit):**  
👉 https://https://cctv-crowd-detection.streamlit.app

---

## Section 1: Environment Setup

### Dependencies

The project relies on the following libraries:

* `streamlit` – Web-based application framework
* `ultralytics` – YOLOv8 object detection framework
* `opencv-python-headless` – Image and video processing
* `numpy` – Numerical computation
* `torch` & `torchvision` – Deep learning backend
* `Pillow` - Image processing library

All dependencies are defined in `requirements.txt` to ensure consistent execution across environments.

### Python Version

The application uses:

```
Python 3.11
```

This version ensures compatibility with YOLOv8, OpenCV, and NumPy.

---

## Section 2: Application Architecture

The system is composed of three main components:

1. **User Interface (Streamlit)**

   * Image and video upload
   * Parameter configuration (preprocessing & inference)
   * Result visualization

2. **Inference Engine (YOLOv8)**

   * Loads trained YOLOv8 model (`best.pt`)
   * Performs person detection
   * Outputs bounding boxes and confidence scores

3. **Post-processing & Visualization**

   * Counts detected people
   * Classifies crowd density
   * Displays results using metrics, badges, and video overlays

---

## Section 3: Model Configuration

### YOLOv8 Model

The project uses a **YOLOv8-based person detection model** trained on a crowd dataset.

* **Detected class:** `person` (class ID = 0)
* **Input size:** 640 × 640
* **Inference device:** CPU (for cloud compatibility)

The model is loaded once using Streamlit’s caching mechanism to improve performance and prevent redundant loading.

---

## Section 4: Preprocessing Pipeline

To improve detection accuracy in indoor CCTV conditions, optional preprocessing steps are provided:

### Gamma Correction

* Enhances brightness in low-light environments
* Improves visibility of people in dim indoor scenes

### Gaussian Blur

* Reduces noise caused by CCTV compression
* Helps stabilize detection in low-quality footage

Users can enable or disable preprocessing via the sidebar.

---

## Section 5: Person Detection Logic

For each image or video frame:

1. The frame is passed to the YOLOv8 model
2. The model outputs bounding boxes and confidence values
3. Only detections satisfying:

   * `class_id = 0 (person)`
   * `confidence ≥ user-defined threshold`
     are counted as valid detections

Bounding boxes are drawn on the image or video frame for visual verification.

---

## Section 6: Crowd Density Classification

After counting detected people, the system classifies crowd density using predefined thresholds:

| Number of People | Crowd Level     |
| ---------------- | --------------- |
| ≤ 3              | Sedikit (Low)   |
| 4 – 30           | Sedang (Medium) |
| > 30             | Ramai (High)    |

This classification simplifies interpretation for monitoring and decision-making.

---

## Section 7: Image Inference Mode

In **Image Mode**, the system:

1. Accepts a single image upload
2. Applies optional preprocessing
3. Performs YOLOv8 inference
4. Displays:

   * Detected bounding boxes
   * Total people count
   * Crowd density badge (color-coded)

| Crowd Level | Bahasa      | Color     | Description                                                                                                   |
| ----------- | ----------- | --------- | ------------------------------------------------------------------------------------------------------------- |
| **Low**     | **Sedikit** | 🟢 Green  | Indicates a small number of people. The area is relatively empty and safe.                                    |
| **Medium**  | **Sedang**  | 🟡 Yellow | Indicates a moderate crowd level. Normal activity is present, but monitoring is recommended.                  |
| **High**    | **Ramai**   | 🔴 Red    | Indicates a high crowd density. The area is crowded and may require attention for safety or capacity control. |

---

## Section 8: Video Inference Mode

To ensure stability on Streamlit Cloud, video processing is optimized with the following constraints:

### Optimization Strategies

* Frame skipping (process every N-th frame)
* Maximum frame limit (~10 seconds)
* CPU-only inference
* Offline video processing before display

### Output

The processed video includes:

* Bounding boxes around detected people
* Text overlay:

  ```
  People: X | Crowd: Level
  ```

This design balances performance and usability in resource-limited environments.

---

## Section 9: User Interface and Visualization

The user interface is designed for clarity and consistency:

* **Metrics** (`st.metric`) display numerical results
* **Badges** visualize crowd density using color coding
* **Sidebar controls** allow real-time parameter adjustment
* **Consistent spacing and layout** improve readability

The interface follows a minimal and professional style suitable for academic evaluation.

---

## Section 10: System Limitations

Despite good performance, the system has several limitations:

1. Occlusion in crowded scenes may reduce detection accuracy
2. No person tracking across frames
3. CPU-only inference limits real-time performance
4. Video duration is restricted for cloud deployment

These limitations are acknowledged and can be addressed in future work.

---

## Complete Workflow Summary

```
1. INPUT
   ├─ Image or Video Upload
   ├─ User-defined parameters

2. PREPROCESSING
   ├─ Gamma correction (optional)
   ├─ Gaussian blur (optional)

3. INFERENCE
   ├─ YOLOv8 person detection
   ├─ Confidence filtering

4. POST-PROCESSING
   ├─ Count detected people
   ├─ Classify crowd density

5. OUTPUT
   ├─ Bounding box visualization
   ├─ Crowd metrics and badges
   └─ Processed video (if applicable)
```

---

## Applications

This system can be applied to:

1. Indoor CCTV monitoring
2. Crowd density analysis
3. Retail and mall analytics
4. Public safety monitoring
5. Academic research and demonstrations

---

## Potential Improvements

Future enhancements may include:

* Person tracking across frames
* Heatmap-based crowd density visualization
* Real-time camera integration
* GPU-accelerated deployment
* Automatic alert generation

---

## Conclusion

This project demonstrates a **complete applied deep learning workflow**, integrating:

* YOLOv8-based object detection
* Interactive web-based visualization
* Practical deployment considerations
