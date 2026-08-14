import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-box h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
        -webkit-text-fill-color: #2d3436;
    }
    .confidence-high { color: #00b894; }
    .confidence-medium { color: #fdcb6e; }
    .confidence-low { color: #e17055; }
    </style>
""", unsafe_allow_html=True)

# Load ONNX model
@st.cache_resource
def load_onnx_model():
    try:
        session = ort.InferenceSession('bacteria_classifier.onnx')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return session, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

def predict_image(image, session, class_names):
    # Resize and preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    predictions = session.run([output_name], {input_name: img_array})[0]
    
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    all_predictions = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
    
    return class_names[predicted_class], confidence, all_predictions

def get_bacteria_info(bacteria_name):
    info = {
        "Staphylococcus_aureus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "β-hemolytic",
            "key_id": "Catalase+, Coagulase+, Mannitol fermenter",
            "color": "Golden-yellow to cream",
            "size": "2-4 mm",
            "description": "Round, smooth, convex, opaque colonies with golden-yellow pigment",
            "virulence": "High",
            "treatment": "Methicillin (if MSSA), Vancomycin (if MRSA)"
        },
        "Staphylococcus_saprophyticus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "Usually γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin resistant",
            "color": "White to cream",
            "size": "1-3 mm",
            "description": "Smooth, convex, opaque colonies; usually non-pigmented",
            "virulence": "Moderate",
            "treatment": "Trimethoprim-sulfamethoxazole, Nitrofurantoin"
        },
        "Staphylococcus_epidermidis": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin sensitive",
            "color": "White to grayish-white",
            "size": "Small",
            "description": "Small, smooth, circular, convex, non-pigmented, glossy colonies",
            "virulence": "Low (Opportunistic)",
            "treatment": "Vancomycin, Rifampin"
        },
        "Streptococcus_pneumoniae": {
            "gram_stain": "Gram-positive lancet-shaped diplococci",
            "hemolysis": "α-hemolytic",
            "key_id": "Optochin sensitive, bile soluble",
            "color": "Gray, translucent",
            "size": "Small",
            "description": "Small, glistening, mucoid colonies; older colonies have central depression",
            "virulence": "High",
            "treatment": "Penicillin, Ceftriaxone, Vancomycin"
        },
        "Streptococcus_pyogenes": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Strong β-hemolytic",
            "key_id": "Bacitracin sensitive, PYR positive",
            "color": "Grayish-white, translucent",
            "size": "Tiny (0.5-1 mm)",
            "description": "Small, translucent, pinpoint colonies with strong beta hemolysis",
            "virulence": "High",
            "treatment": "Penicillin, Amoxicillin"
        },
        "Streptococcus_agalactiae": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Narrow β-hemolytic",
            "key_id": "CAMP positive, Hippurate positive",
            "color": "Grayish-white to cream",
            "size": "Medium",
            "description": "Medium-sized, smooth, grayish-white to cream, slightly mucoid colonies",
            "virulence": "Moderate",
            "treatment": "Penicillin, Ampicillin"
        }
    }
    return info.get(bacteria_name, {})

def main():
    st.markdown('<h1 class="main-title">🧫 Bacteria Colony Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload an image of a bacterial colony to identify the species based on morphology</p>', unsafe_allow_html=True)
    
    session, class_names = load_onnx_model()
    
    if session is None:
        st.warning("⚠️ Please make sure 'bacteria_classifier.onnx' and 'class_names.json' are in the same directory.")
        return
    
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
        help="Upload a clear image of a bacterial colony plate"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            image = Image.open(uploaded_file)
            # FIX: Changed use_column_width to use_container_width
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔬 Classify Bacteria", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    predicted, confidence, all_predictions = predict_image(
                        image, session, class_names
                    )
                    
                    # Confidence color
                    if confidence > 70:
                        conf_color = "confidence-high"
                        conf_text = "High"
                    elif confidence > 50:
                        conf_color = "confidence-medium"
                        conf_text = "Medium"
                    else:
                        conf_color = "confidence-low"
                        conf_text = "Low"
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="color:#636e72; margin:0;">🧬 Predicted Species</p>
                        <h2>{predicted.replace('_', ' ')}</h2>
                        <p class="{conf_color}" style="font-size:1.5rem; font-weight:600;">
                            {confidence:.1f}% Confidence ({conf_text})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(int(confidence))
                    
                    # Show morphology info
                    info = get_bacteria_info(predicted)
                    if info:
                        with st.expander("📊 Morphology & Clinical Characteristics", expanded=True):
                            tab1, tab2, tab3 = st.tabs(["🔬 Morphology", "🦠 Clinical", "💊 Treatment"])
                            
                            with tab1:
                                st.markdown(f"""
                                - **Gram Stain**: {info['gram_stain']}
                                - **Hemolysis**: {info['hemolysis']}
                                - **Colony Color**: {info['color']}
                                - **Colony Size**: {info['size']}
                                - **Description**: {info['description']}
                                """)
                            
                            with tab2:
                                st.markdown(f"""
                                - **Virulence**: {info['virulence']}
                                - **Key Identification**: {info['key_id']}
                                """)
                            
                            with tab3:
                                st.markdown(f"""
                                - **Treatment**: {info['treatment']}
                                """)
                    
                    # Confidence chart
                    st.markdown("### 📈 Confidence Scores for All Species")
                    
                    sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(sorted_predictions.keys()),
                            y=list(sorted_predictions.values()),
                            marker_color=['#00b894' if x == predicted else '#dfe6e9' 
                                         for x in sorted_predictions.keys()],
                            text=[f"{v:.1f}%" for v in sorted_predictions.values()],
                            textposition='outside'
                        )
                    ])
                    fig.update_layout(
                        xaxis_title="Bacteria Species",
                        yaxis_title="Confidence (%)",
                        yaxis_range=[0, 100],
                        height=400,
                        xaxis_tickangle=-45,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
