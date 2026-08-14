import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import os
import plotly.graph_objects as go

st.set_page_config(page_title="🧫 Bacteria Classifier", layout="wide")

class_names = [
    "Staphylococcus_aureus",
    "Staphylococcus_saprophyticus",
    "Staphylococcus_epidermidis",
    "Streptococcus_pneumoniae",
    "Streptococcus_pyogenes",
    "Streptococcus_agalactiae"
]

@st.cache_resource
def load_onnx_model():
    try:
        session = ort.InferenceSession('bacteria_classifier.onnx')
        
        # Print model info for debugging
        st.sidebar.markdown("### 🔍 Model Debug Info")
        for inp in session.get_inputs():
            st.sidebar.write(f"Input: {inp.name}, Shape: {inp.shape}, Type: {inp.type}")
        for out in session.get_outputs():
            st.sidebar.write(f"Output: {out.name}, Shape: {out.shape}")
        
        return session
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def predict_with_debug(image, session):
    """Test different preprocessing methods"""
    
    # Method 1: Standard (0-1 normalization)
    img1 = image.resize((224, 224))
    arr1 = np.array(img1).astype(np.float32) / 255.0
    arr1 = np.expand_dims(arr1, axis=0)
    
    # Method 2: No normalization (0-255)
    img2 = image.resize((224, 224))
    arr2 = np.array(img2).astype(np.float32)
    arr2 = np.expand_dims(arr2, axis=0)
    
    # Method 3: -1 to 1 normalization
    img3 = image.resize((224, 224))
    arr3 = np.array(img3).astype(np.float32) / 127.5 - 1.0
    arr3 = np.expand_dims(arr3, axis=0)
    
    # Method 4: BGR instead of RGB
    img4 = image.resize((224, 224))
    arr4 = np.array(img4).astype(np.float32)[:, :, ::-1] / 255.0  # BGR
    arr4 = np.expand_dims(arr4, axis=0)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    results = {}
    methods = {
        "Normal (0-1)": arr1,
        "No norm (0-255)": arr2,
        "-1 to 1": arr3,
        "BGR": arr4
    }
    
    for method_name, arr in methods.items():
        try:
            pred = session.run([output_name], {input_name: arr})[0]
            idx = np.argmax(pred[0])
            results[method_name] = {
                'predicted': class_names[idx],
                'confidence': pred[0][idx] * 100,
                'all': {class_names[i]: pred[0][i] * 100 for i in range(len(class_names))}
            }
        except Exception as e:
            results[method_name] = f"Error: {e}"
    
    return results

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
    
    if st.button("🔬 Classify with ALL methods", use_container_width=True):
        with st.spinner("Analyzing..."):
            results = predict_with_debug(image, session)
            
            for method, result in results.items():
                if isinstance(result, str):
                    st.error(f"{method}: {result}")
                else:
                    color = "#2e7d32" if result['confidence'] > 70 else "#f9a825" if result['confidence'] > 50 else "#c62828"
                    st.markdown(f"""
                        <div style='background:white; padding:1rem; border-radius:10px; border:2px solid #8bc34a; margin:0.5rem 0;'>
                            <h3>Method: {method}</h3>
                            <p><strong>Predicted:</strong> {result['predicted'].replace('_', ' ')}</p>
                            <p><strong>Confidence:</strong> <span style='color:{color};font-weight:bold;'>{result['confidence']:.1f}%</span></p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all scores
                    sorted_preds = dict(sorted(result['all'].items(), key=lambda x: x[1], reverse=True))
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(sorted_preds.keys()),
                            y=list(sorted_preds.values()),
                            marker_color=['#2e7d32' if i==0 else '#bdbdbd' for i in range(len(sorted_preds))],
                            text=[f"{v:.1f}%" for v in sorted_preds.values()],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(height=300, xaxis_tickangle=-45, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
