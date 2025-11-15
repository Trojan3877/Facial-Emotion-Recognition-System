import streamlit as st
from src.predict import predict_emotion

st.title("🎭 Facial Emotion Recognition (Scikit-Learn)")

uploaded = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded:
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.getbuffer())
    
    st.image("temp.jpg")
    pred = predict_emotion("temp.jpg")
    st.success(f"Predicted Emotion: **{pred}**")
