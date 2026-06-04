import os
import streamlit as st
from PIL import Image
from src.predict import Predictor

# Define model and directory paths
MODEL_PATH = os.path.join("model", "saved", "mobilenet_latest.h5")
TEMP_IMAGE_PATH = "temp_upload.jpeg"
HEATMAP_SAVE_PATH = "reports/gradcam_streamlit.png"

# Ensure the reports folder exists for saving heatmaps
os.makedirs("reports", exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="MediScan AI - Medical Image Diagnosis",
    page_icon="🩻",
    layout="wide"
)

# Initialize predictor
@st.cache_resource
def load_inference_engine():
    if os.path.exists(MODEL_PATH):
        return Predictor(MODEL_PATH)
    return None

predictor = load_inference_engine()

# App Header
st.title("🩻 MediScan AI — AI-Based Chest X-Ray Diagnosis")
st.write("Upload a chest X-ray image (JPEG/PNG) to classify it as **NORMAL** or **PNEUMONIA** with Grad-CAM explainability.")

# Check if model is loaded successfully
if predictor is None:
    st.error(f"Error: Model weights not found at `{MODEL_PATH}`. Please train the model using your notebook first.")
else:
    # App layout splits into sidebar (for upload) and main panel (for results)
    st.sidebar.header("Upload Center")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a Chest X-Ray Image...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Save the uploaded file temporarily to process it
        image = Image.open(uploaded_file)
        image.save(TEMP_IMAGE_PATH)
        
        # UI Columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
            
        with col2:
            st.subheader("AI Analysis")
            with st.spinner("Analyzing image..."):
                # Run prediction and generate Grad-CAM heatmap
                predicted_class, confidence = predictor.predict(
                    TEMP_IMAGE_PATH, 
                    save_heatmap_path=HEATMAP_SAVE_PATH
                )
                
            # Display results
            if predicted_class == "PNEUMONIA":
                st.error(f"Diagnosis: **{predicted_class}**")
            else:
                st.success(f"Diagnosis: **{predicted_class}**")
                
            st.write(f"Confidence: **{confidence:.2%}**")
            st.progress(confidence)
            
            # Display Grad-CAM heatmap if saved successfully
            if os.path.exists(HEATMAP_SAVE_PATH):
                st.subheader("Grad-CAM Activation Map")
                st.image(HEATMAP_SAVE_PATH, use_container_width=True)
                st.caption("Warm colors (red/yellow) indicate the regions that influenced the AI prediction the most.")
                
        # Clean up temporary upload file
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            
    else:
        st.info("Please upload a chest X-ray image in the sidebar to begin analysis.")

# Clinical Disclaimer
st.markdown("---")
st.warning(
    "⚠️ **Disclaimer**: This AI model is developed for educational and industrial training research purposes. "
    "It is NOT a certified medical device and should not be used as a replacement for professional clinical advice, "
    "diagnosis, or treatment by a certified radiologist."
)