import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from src.inference.stream_processor.py import AsyncStreamProcessor

st.set_page_config(page_title="Facial Emotion Telemetry Room", layout="wide")

st.title("👁️ Edge Vision Inference & Observability Matrix")
st.caption("Real-Time Face Segmentation, Probability Vectors, and Compute Performance Diagnostics")

# Initialize Session Frame Processors
if 'processor' not in st.session_state:
    # Set to local webcam (0) or path to target video clip mockup
    st.session_state.processor = AsyncStreamProcessor(source=0).start()

processor = st.session_state.processor

# Layout Structure Configuration
left_canvas, right_telemetry = st.columns([2, 1])

with left_canvas:
    st.subheader("🎥 Live Edge Inference Stream")
    video_placeholder = st.empty()

with right_telemetry:
    st.subheader("📊 Class Probability Distribution")
    chart_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("⚡ Hardware Runtime Metrics")
    fps_metric = st.metric("Processing Throughput", "0 FPS")
    latency_metric = st.metric("Inference Pipeline Latency", "0.00 ms")

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Active UI Ingestion Event Loop
while True:
    start_time = time.time()
    frame = processor.read_frame()
    
    if frame is not None:
        # 1. Simulate target model matrix lookup transformation pipeline
        # (Replace with model.predict output arrays)
        mock_predictions = np.random.dirichlet(np.ones(7), size=1)[0]
        dominant_emotion = emotions[np.argmax(mock_predictions)]
        
        # 2. Render localized bounding graphics directly onto OpenCV Matrix
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(w*0.35), int(h*0.25)), (int(w*0.65), int(h*0.75)), (46, 213, 115), 2)
        cv2.putText(frame, f"Emotion: {dominant_emotion}", (int(w*0.35), int(h*0.23)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 213, 115), 2)
        
        # Transform BGR matrix states to RGB for native Streamlit displays
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
        
        # 3. Update Visual Metric DataFrames
        prob_df = pd.DataFrame({
            'Emotion Index': emotions,
            'Confidence Score': mock_predictions
        }).sort_values(by='Confidence Score', ascending=True)
        
        chart_placeholder.bar_chart(data=prob_df, x='Emotion Index', y='Confidence Score', use_container_width=True)
        
        # 4. Latency calculations
        loop_latency = (time.time() - start_time) * 1000
        computed_fps = 1000 / loop_latency if loop_latency > 0 else 0
        
        fps_metric.metric("Processing Throughput", f"{computed_fps:.1f} FPS")
        latency_metric.metric("Inference Pipeline Latency", f"{loop_latency:.2f} ms")
        
    time.sleep(0.03) # Cap execution intervals roughly to 30 FPS targeting structural balance
