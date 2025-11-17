"""
=========================================================
FACIAL EMOTION RECOGNITION — STREAMLIT WEB APP (L5/L6)
Author: Trojan3877 (Corey Leath)
Description:
    - Provides a clean web UI for emotion prediction
    - Allows image uploads
    - Displays confidence scores
=========================================================
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from predict import preprocess_image, EMOTION_LABELS

MODEL_PATH = "emotion_model_final.h5"

# ---------------------------------------------------------
# Load model once (performance optimized)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------------
# Page Settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="🙂",
    layout="centered"
)

st.title("🎭 Facial Emotion Recognition System")
st.write("Upload an image and let the model detect the emotion.")

# ---------------------------------------------------------
# File Upload
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    resized = cv2.resize(image, (48, 48))
    normalized = resized.astype("float32") / 255.0
    input_image = np.expand_dims(normalized, axis=-1)
    input_image = np.expand_dims(input_image, axis=0)

    # Prediction
    preds = model.predict(input_image)
    pred_idx = np.argmax(preds)
    emotion = EMOTION_LABELS[pred_idx]
    confidence = float(np.max(preds))

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------
    st.subheader("🎯 Prediction Result")
    st.write(f"**Emotion:** {emotion}")
    st.write(f"**Confidence:** {confidence:.4f}")

    # Confidence bar chart
    st.subheader("📊 Confidence Scores")
    confidence_dict = {EMOTION_LABELS[i]: float(preds[0][i]) for i in range(7)}
    st.bar_chart(confidence_dict)

else:
    st.info("Please upload an image to get started.")
