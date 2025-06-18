import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load model and class labels
model = load_model("model/emotion_model.h5")
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Set up Streamlit UI
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")
st.title("🤖 Facial Emotion Recognition System")
st.write("Upload a face image (48x48 grayscale) to predict the emotion.")

# File uploader
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# Predict and display results
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)
    
    input_image = preprocess_image(image)
    prediction = model.predict(input_image)
    
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)), 4)

    st.subheader(f"🎯 Predicted Emotion: `{predicted_class}`")
    st.write(f"Confidence: `{confidence}`")

    st.success("Prediction complete!")

