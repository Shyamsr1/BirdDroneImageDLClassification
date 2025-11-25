import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging

import time
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Rescaling
from pathlib import Path

# ============================================================
# 1.  YOLO import
# ============================================================
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception as e:
    YOLO_AVAILABLE = False

# Path to the trained YOLO weights (Can change if needed)
YOLO_MODEL_PATH = "runs/detect/bird_drone_yolov8/weights/best.pt"

# ============================================================
# 2. Existing Keras model loading 
# ============================================================

# Project base directory (folder where this app.py is located)
# Folder containing app.py (streamlit_app)
APP_DIR = Path(__file__).resolve().parent

# Project root: folder 
ROOT_DIR = APP_DIR.parent

# models folder is here
MODELS_DIR = ROOT_DIR / "models"

# Paths to the Keras model files - CNN and MobileNetV2
CNN_MODEL_PATH = MODELS_DIR / "best_custom_cnn.h5"
MOBILENET_MODEL_PATH = MODELS_DIR / "best_mobilenetv2.h5"

@st.cache_resource
def load_keras_model(model_path: Path):
    """Safely load a Keras model from disk."""
    if not model_path.exists():
        st.error(f" Model file not found at: {model_path}")
        st.stop()
    return tf.keras.models.load_model(str(model_path))

@st.cache_resource
def get_cnn_model():
    return load_keras_model(CNN_MODEL_PATH)

@st.cache_resource
def get_mobilenet_model():
    return load_keras_model(MOBILENET_MODEL_PATH)

# ============================================================
# 3. YOLO model loading (NEW)
# ============================================================

@st.cache_resource
def get_yolo_model():
    """
    Load YOLO model only if ultralytics is available and weights file exists.
    Returns None if not usable.
    """
    if not YOLO_AVAILABLE:
        return None
    if not os.path.exists(YOLO_MODEL_PATH):
        return None
    try:
        model = YOLO(YOLO_MODEL_PATH)
        return model
    except Exception:
        return None

# ============================================================
# 4. Utility: preprocess for classification models (UNCHANGED)
# ============================================================

IMG_SIZE = (224, 224)  # adapt if the models expect different size

def preprocess_for_classification(img: Image.Image):
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

CLASS_NAMES = ["Bird", "Drone"]  # adapt to the labels

def classify_with_keras(model, img: Image.Image):
    arr = preprocess_for_classification(img)
    preds = model.predict(arr)
    prob = float(np.max(preds))
    cls_idx = int(np.argmax(preds))
    return CLASS_NAMES[cls_idx], prob

# ============================================================
# 5. Utility: run YOLO and handle "not possible" logic (NEW)
# ============================================================

def run_yolo_inference(yolo_model, img: Image.Image, conf_threshold: float = 0.25):
    """
    Runs YOLO inference on the given PIL image.
    Returns:
        - annotated_image (np.ndarray RGB) or None
        - detections: list of dicts with label & confidence
        - message: None if success else string explaining why YOLO not applied
    """
    if yolo_model is None:
        return None, [], "YOLO model is not available in this environment."

    # Run inference
    results = yolo_model(img)

    if not results:
        return None, [], "YOLO could not process this image. For this image, YOLO cannot be applied."

    result = results[0]

    # If no boxes detected, treat as "not possible"
    if result.boxes is None or len(result.boxes) == 0:
        return None, [], (
            "YOLO could not detect any bird or drone in this image. "
            "For this image, YOLO cannot be applied."
        )

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < conf_threshold:
            # Ignore very low-confidence predictions
            continue
        label = result.names[cls_id]
        detections.append({"label": label, "confidence": conf})

    if not detections:
        return None, [], (
            "YOLO found only very low-confidence detections. "
            "For this image, YOLO cannot be applied reliably."
        )

    # Get annotated image (BGR -> RGB)
    annotated_bgr = result.plot()
    annotated_rgb = annotated_bgr[..., ::-1]  # flip channels

    return annotated_rgb, detections, None  # success, no error message

# ============================================================
# 6. Streamlit UI
# ============================================================

st.set_page_config(page_title="Bird vs Drone - DL + YOLO", layout="centered")
st.title("Bird vs Drone Image Classification & Detection")

st.write("Upload an image and choose a model for prediction.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# --- Model selection: add YOLO to dropdown (NEW) ---
model_options = ["Baseline CNN", "MobileNetV2", "YOLOv8 (Object Detection)"]
model_choice = st.selectbox("Select model", model_options)

# Show uploaded image
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image")
else:
    img = None

# Predict button
if st.button("Predict"):
    if img is None:
        st.warning("Please upload an image first.")
    else:
        # Clear old outputs by using containers if you want to, but no need to break existing logic

        if model_choice == "Baseline CNN":
            model = get_cnn_model()
            label, prob = classify_with_keras(model, img)
            st.subheader("Baseline CNN Prediction")
            st.write(f"**Class:** {label}")
            st.write(f"**Confidence:** {prob:.2%}")

        elif model_choice == "MobileNetV2":
            model = get_mobilenet_model()
            label, prob = classify_with_keras(model, img)
            st.subheader("MobileNetV2 Prediction")
            st.write(f"**Class:** {label}")
            st.write(f"**Confidence:** {prob:.2%}")

        elif model_choice == "YOLOv8 (Object Detection)":
            st.subheader("YOLOv8 Detection")

            if not YOLO_AVAILABLE:
                st.error(
                    "YOLO (Ultralytics) is not installed in this deployment. "
                    "Please run locally with YOLO installed, or use the CNN models above."
                )
            else:
                yolo_model = get_yolo_model()

                annotated_img, detections, msg = run_yolo_inference(yolo_model, img)

                if msg is not None:
                    # This is the **Not possible** message when image is not as per YOLO
                    st.warning(msg)
                else:
                    # Show annotated image
                    st.image(annotated_img, caption="YOLO detections")

                    # List detections
                    st.write("**Detections:**")
                    for det in detections:
                        st.write(f"- {det['label']} ({det['confidence']:.2%})")

