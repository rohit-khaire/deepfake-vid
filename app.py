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

# Display available pre-trained weights
with st.expander("📚 Available Pre-trained Weights"):
    st.markdown("""
    **Pre-trained models available in `weights/` folder:**
    - `Meso4_DF.h5` - MesoNet-4 for DeepFakes
    - `Meso4_F2F.h5` - MesoNet-4 for Face2Face
    - `MesoInception_DF.h5` - MesoInception for DeepFakes
    - `MesoInception_F2F.h5` - MesoInception for Face2Face
    - `MesoInception_F2F.pth` - PyTorch format (recommended)
    
    **How to use custom weights:**
    1. Convert .h5 weights to .pth format (if needed)
    2. Enter the full path in "Model weights path" field
    3. Path examples: `weights/Meso4_DF.h5` or `/full/path/to/weights.pth`
    """)

uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'avi', 'mov', 'mkv'])

st.markdown("### Model Configuration")
col1, col2 = st.columns(2)

with col1:
    weights_path = st.text_input("Model weights path (optional)", "", help="Path to custom .pth weights file. Leave empty to use default weights.")
    if weights_path:
        st.info(f"📁 Will use custom weights from: `{weights_path}`")

with col2:
    threshold = st.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01, help="Higher = more conservative (fewer false positives)")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        video_path = tmp_file.name
    
    try:
        try:
            detector = DeepfakeVideoDetector(weights_path=weights_path if weights_path else None, threshold=threshold)
        except FileNotFoundError as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
        except ValueError as e:
            st.error(f"❌ Error: {str(e)}")
            st.stop()
        
        with st.spinner("Analyzing video... This may take a few minutes."):
            result = detector.detect_video(video_path)
        
        if 'error' in result:
            st.error(f"Error: {result['error']}")
        else:
            st.success(f"✅ Analysis complete using {result['weights_source']} model")
            
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
st.markdown("""
**Model Usage Guide:**
- **Default mode**: Uses built-in Meso4 architecture
- **Custom weights**: Provide a .pth PyTorch weights file path
  - For .h5 (Keras) files: Use `convert_weights.py` to convert first
  - Command: `python convert_weights.py weights/model.h5`

**Troubleshooting:**
- If weights fail to load, verify the path and file format (.pth recommended)
- For .h5 files: Install h5py with `pip install h5py`
- Check that weights match Meso4 model architecture
""")