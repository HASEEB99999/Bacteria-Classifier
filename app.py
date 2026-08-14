import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import plotly.graph_objects as go

st.set_page_config(page_title="🧫 Bacteria Classifier", layout="wide")

# ============ CORRECT CLASS NAMES ============
# These must match the EXACT order your model was trained on
class_names = [
    "Staphylococcus_aureus",
    "Staphylococcus_saprophyticus",
    "Staphylococcus_epidermidis",
    "Streptococcus_pneumoniae",
    "Streptococcus_pyogenes",
    "Streptococcus_agalactiae"
]

# ============ DISPLAY NAMES (for UI) ============
display_names = {
    "Staphylococcus_aureus": "Staphylococcus aureus",
    "Staphylococcus_saprophyticus": "Staphylococcus saprophyticus",
    "Staphylococcus_epidermidis": "Staphylococcus epidermidis",
    "Streptococcus_pneumoniae": "Streptococcus pneumoniae",
    "Streptococcus_pyogenes": "Streptococcus pyogenes",
    "Streptococcus_agalactiae": "Streptococcus agalactiae"
}

# ============ LOAD MODEL ============
@st.cache_resource
def load_onnx_model():
    try:
        session = ort.InferenceSession('bacteria_classifier.onnx')
        
        # Debug info
        st.sidebar.markdown("### 🔍 Model Info")
        for inp in session.get_inputs():
            st.sidebar.write(f"Input: {inp.name}, Shape: {inp.shape}")
        for out in session.get_outputs():
            st.sidebar.write(f"Output: {out.name}, Shape: {out.shape}")
        
        return session
    except Exception as e:
        st.error(f"Error: {e}")
        return None

# ============ PREDICTION ============
def predict_image(image, session):
    img = image.resize((224, 224))
    
    # Try BGR (most common for ONNX)
    img_array = np.array(img).astype(np.float32)
    img_array = img_array[:, :, ::-1]  # RGB to BGR
    img_array = np.expand_dims(img_array, axis=0)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img_array})[0]
    
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    all_predictions = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
    
    return class_names[predicted_class], confidence, all_predictions

# ============ UI ============
st.markdown("""
    <h1 style='text-align:center;'>🧫 Bacteria Colony Classifier</h1>
    <p style='text-align:center;'>🔬 Upload an image to identify bacteria species</p>
""", unsafe_allow_html=True)

session = load_onnx_model()

if session is None:
    st.warning("⚠️ Please make sure 'bacteria_classifier.onnx' is in the app directory.")
    st.stop()

uploaded = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("🔬 Classify", use_container_width=True):
        with st.spinner("Analyzing..."):
            predicted, confidence, all_predictions = predict_image(image, session)
            
            # Get display name
            display_name = display_names.get(predicted, predicted.replace('_', ' '))
            
            color = "#2e7d32" if confidence > 70 else "#f9a825" if confidence > 50 else "#c62828"
            
            st.markdown(f"""
                <div style='background:white; padding:2rem; border-radius:15px; border:3px solid #8bc34a; text-align:center;'>
                    <h2>{display_name}</h2>
                    <p style='font-size:2rem; font-weight:bold; color:{color};'>{confidence:.1f}%</p>
                    <span style='background:{color}; color:white; padding:0.3rem 1.5rem; border-radius:50px;'>
                        {'✅ High' if confidence > 70 else '⚠️ Medium' if confidence > 50 else '❌ Low'} Confidence
                    </span>
                </div>
            """, unsafe_allow_html=True)
            
            # Show all scores
            st.markdown("### 📊 All Confidence Scores")
            sorted_preds = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
            
            # Use display names
            display_preds = {display_names.get(k, k.replace('_', ' ')): v for k, v in sorted_preds.items()}
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(display_preds.keys()),
                    y=list(display_preds.values()),
                    marker_color=['#2e7d32' if i==0 else '#bdbdbd' for i in range(len(display_preds))],
                    text=[f"{v:.1f}%" for v in display_preds.values()],
                    textposition='outside'
                )
            ])
            fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False, yaxis_range=[0, 100])
            st.plotly_chart(fig, use_container_width=True)
