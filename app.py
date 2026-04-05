import streamlit as st
import tempfile
import os
from src.detector import DeepfakeVideoDetector
import json

st.title("GOD Level Deepfake Video Detector")

st.markdown("""
Upload a video to detect if it's a deepfake using advanced AI models.
This tool analyzes frames for manipulation artifacts.
""")

uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])

weights_path = st.text_input("Model weights path (optional)", "")

threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    try:
        detector = DeepfakeVideoDetector(weights_path=weights_path if weights_path else None, threshold=threshold)
        
        with st.spinner("Analyzing video... This may take a few minutes."):
            result = detector.detect_video(video_path)
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Fake Probability", f"{result['average_fake_probability']:.3f}")
                st.metric("Max Fake Probability", f"{result['max_fake_probability']:.3f}")
            
            with col2:
                st.metric("Min Fake Probability", f"{result['min_fake_probability']:.3f}")
                st.metric("Frames Analyzed", result['num_frames_analyzed'])
            
            if result['is_fake']:
                st.error("⚠️ This video appears to be a DEEPFAKE!")
            else:
                st.success("✅ This video appears to be AUTHENTIC!")
            
            # Display frame-by-frame analysis
            if result['frame_predictions']:
                st.subheader("Frame-by-Frame Analysis")
                st.line_chart(result['frame_predictions'])
            
            # Show raw results
            with st.expander("Raw Results"):
                st.json(result)
    
    finally:
        # Clean up temp file
        os.unlink(video_path)

st.markdown("---")
st.markdown("**Note:** For best results, use pre-trained weights. Download from MesoNet repository.")