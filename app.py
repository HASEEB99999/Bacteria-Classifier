import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import plotly.graph_objects as go
import os

st.set_page_config(page_title="🧫 Bacteria Classifier", layout="wide")

class_names = [
    "Staphylococcus_aureus",
    "Staphylococcus_saprophyticus",
    "Staphylococcus_epidermidis",
    "Streptococcus_pneumoniae",
    "Streptococcus_pyogenes",
    "Streptococcus_agalactiae"
]

display_names = {
    "Staphylococcus_aureus": "Staphylococcus aureus",
    "Staphylococcus_saprophyticus": "Staphylococcus saprophyticus",
    "Staphylococcus_epidermidis": "Staphylococcus epidermidis",
    "Streptococcus_pneumoniae": "Streptococcus pneumoniae",
    "Streptococcus_pyogenes": "Streptococcus pyogenes",
    "Streptococcus_agalactiae": "Streptococcus agalactiae"
}

@st.cache_resource
def load_onnx_model():
    try:
        if not os.path.exists('bacteria_classifier.onnx'):
            st.error("❌ bacteria_classifier.onnx file not found!")
            return None
        
        session = ort.InferenceSession('bacteria_classifier.onnx')
        
        # Get input/output details
        st.sidebar.markdown("### 🔍 Model Info")
        for inp in session.get_inputs():
            st.sidebar.write(f"Input: {inp.name}, Shape: {inp.shape}")
        for out in session.get_outputs():
            st.sidebar.write(f"Output: {out.name}, Shape: {out.shape}")
        
        return session
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def predict_image(image, session, method="normal"):
    """
    Predict with different preprocessing methods
    method: "normal", "no_norm", "neg1_to_1", "bgr", "bgr_no_norm"
    """
    img = image.resize((224, 224))
    
    if method == "normal":
        # Standard: 0-1 normalization
        arr = np.array(img).astype(np.float32) / 255.0
    elif method == "no_norm":
        # No normalization (0-255)
        arr = np.array(img).astype(np.float32)
    elif method == "neg1_to_1":
        # -1 to 1 normalization
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
    elif method == "bgr":
        # BGR with 0-1 normalization
        arr = np.array(img).astype(np.float32)[:, :, ::-1] / 255.0
    elif method == "bgr_no_norm":
        # BGR without normalization
        arr = np.array(img).astype(np.float32)[:, :, ::-1]
    else:
        arr = np.array(img).astype(np.float32) / 255.0
    
    arr = np.expand_dims(arr, axis=0)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    try:
        predictions = session.run([output_name], {input_name: arr})[0]
        idx = np.argmax(predictions[0])
        confidence = predictions[0][idx] * 100
        all_preds = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
        return class_names[idx], confidence, all_preds
    except Exception as e:
        return f"Error: {e}", 0, {}

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
    
    # Method selector
    method = st.selectbox(
        "Select Preprocessing Method",
        [
            "normal (0-1)",
            "no_norm (0-255)",
            "neg1_to_1 (-1 to 1)",
            "bgr (BGR 0-1)",
            "bgr_no_norm (BGR 0-255)"
        ]
    )
    
    method_map = {
        "normal (0-1)": "normal",
        "no_norm (0-255)": "no_norm",
        "neg1_to_1 (-1 to 1)": "neg1_to_1",
        "bgr (BGR 0-1)": "bgr",
        "bgr_no_norm (BGR 0-255)": "bgr_no_norm"
    }
    
    if st.button("🔬 Classify", use_container_width=True):
        with st.spinner("Analyzing..."):
            predicted, confidence, all_preds = predict_image(
                image, session, method_map[method]
            )
            
            if isinstance(predicted, str) and predicted.startswith("Error"):
                st.error(predicted)
            else:
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
                sorted_preds = dict(sorted(all_preds.items(), key=lambda x: x[1], reverse=True))
                
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
