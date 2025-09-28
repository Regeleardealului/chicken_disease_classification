import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import backend as K
import numpy as np
from PIL import Image
from pathlib import Path
import logging
import os # Added os for potential troubleshooting

# 🚨 Page configuration must be the first Streamlit command
st.set_page_config(
    page_title='Poultry Health Analyzer',
    page_icon='virus.png',
    layout='wide'
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- MODEL PATH ADJUSTMENT ---
# Use pathlib to construct the correct path relative to the current script.
# This is the most reliable way to reference files in Python deployments.
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
model_file_name = "best_vgg_enhanced.keras"
model_path = current_dir / model_file_name

logger.info(f"Attempting to load model from: {model_path}")
# --- END MODEL PATH ADJUSTMENT ---

# Load model with error handling
# Load model with error handling
try:
    # Check if the model file is a small LFS pointer file (common Streamlit/GitHub LFS issue)
    if os.path.exists(model_path) and os.path.getsize(model_path) < 1024:
        raise ValueError(f"Model file '{model_file_name}' appears to be a small Git LFS pointer file, not the actual model data.")
    
    K.clear_session()  # Clear previous sessions
    # Keras will handle the correct path format regardless of the OS when using pathlib's output
    model = load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Model loading failed: {str(e)}")
    st.error("Failed to load the AI model. Please check the logs.")

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: 2px solid #3d8b40;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        color: white !important;
        border-color: #367c39;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .prediction-box {
        background-color: #1e1e1e;
        color: #f1f1f1;
        border-left: 6px solid #4CAF50;
        border-radius: 12px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .condition-label { color: #FFB22C; }
    .confidence-label { color: #95D2B3; }
    </style>
    """, unsafe_allow_html=True)

CONFIDENCE_THRESHOLD = 0.95  # Only show predictions if confidence ≥ 95%
REJECTION_MESSAGE = "⚠️ This image doesn't appear to be suitable chicken feces for reliable analysis."

# App UI
st.title('🐔 Disease Prediction')
st.markdown("Upload an image of chicken feces for health analysis")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', width=300)
        
        # Preprocess image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        if st.button('Analyze Image'):
            with st.spinner('Analyzing...'):
                try:
                    # Check if model loaded before predicting
                    if 'model' not in locals():
                        st.error("The AI model is unavailable for prediction.")
                        return

                    predictions = model.predict(img_array)
                    predicted_class = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])
                    
                    class_names = {
                        0: 'Coccidiosis',
                        1: 'Healthy',
                        2: 'New Castle Disease',
                        3: 'Salmonella'
                    }
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h3>🧪 Analysis Results</h3>
                            <p><span class="condition-label">Predicted Condition:</span> {class_names.get(predicted_class, "Unknown")}</p>
                            <p><span class="confidence-label">Confidence Level:</span> {confidence * 100:.2f}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning(REJECTION_MESSAGE)
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    logger.exception("Prediction error")
                    
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        logger.exception("Image processing failed")
