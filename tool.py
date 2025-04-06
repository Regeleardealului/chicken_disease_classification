import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model('best_vgg_enhanced.keras')

st.set_page_config(
    page_title='Poultry Health Analyzer',
    page_icon='virus.png',
    layout='wide'
)

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
    .stButton>button:active {
        transform: translateY(0);  
        background-color: #3d8b40; 
    }
    .prediction-box {
        background-color: white;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .title {
        color: #2c3e50;
    }
    .prediction-box {
        background-color: white;
        color: #000;
        border-radius: 10px;
        padding: 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

st.title('🐔 Disease Prediction')
st.markdown("""
    Upload an image of a chicken feces to analyze its health condition. 
    My AI model will predict potential diseases with confidence scores.
    """)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', width=300)
    
    img = img.resize((224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  
    
    if st.button('Analyze Image'):
        with st.spinner('Analyzing the image...'):
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            class_names = {
                0: 'Coccidiosis',
                1: 'Healthy',
                2: 'New Castle Disease',
                3: 'Salmonella'
            }
            
            st.markdown(f"""
            <style>
                .prediction-box {{
                    background-color: #1e1e1e;
                    color: #f1f1f1;
                    border-left: 6px solid #4CAF50;
                    border-radius: 12px;
                    padding: 2rem;
                    margin-top: 2rem;
                    box-shadow: 0 6px 12px rgba(0,0,0,0.3);
                    font-family: 'Segoe UI', sans-serif;
                }}
                .prediction-title {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    margin-bottom: 1rem;
                }}
                .prediction-item {{
                    font-size: 1.2rem;
                    margin-bottom: 0.6rem;
                }}
                .condition-label {{
                    font-weight: bold;
                    color: #FFB22C;  
                }}
                .confidence-label {{
                    font-weight: bold;
                    color: #95D2B3;  
                }}
            </style>

            <div class="prediction-box">
                <div class="prediction-title">🧪 Analysis Results</div>
                <div class="prediction-item"><span class="condition-label">Predicted Condition:</span> {class_names.get(predicted_class, "Unknown")}</div>
                <div class="prediction-item"><span class="confidence-label">Confidence Level:</span> {confidence * 100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)