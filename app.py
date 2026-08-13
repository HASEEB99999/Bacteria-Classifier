import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import plotly.graph_objects as go
import base64
from io import BytesIO
import time

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Animated Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        animation: gradientShift 15s ease infinite;
        background-size: 400% 400%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.15);
    }
    
    /* Main Title */
    .main-title {
        text-align: center;
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        padding: 1rem 0;
        animation: fadeInDown 0.8s ease;
    }
    
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .sub-title {
        text-align: center;
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed rgba(102, 126, 234, 0.4);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.02);
    }
    
    .upload-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        display: block;
    }
    
    /* Prediction Box */
    .prediction-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(255,255,255,0.6));
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .prediction-box h2 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0.5rem 0;
        background: linear-gradient(135deg, #2d3436, #636e72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .confidence-high { 
        color: #00b894 !important;
        -webkit-text-fill-color: #00b894 !important;
        font-weight: 700;
    }
    .confidence-medium { 
        color: #fdcb6e !important;
        -webkit-text-fill-color: #fdcb6e !important;
        font-weight: 700;
    }
    .confidence-low { 
        color: #e17055 !important;
        -webkit-text-fill-color: #e17055 !important;
        font-weight: 700;
    }
    
    /* Badge Styles */
    .badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
        transition: all 0.3s ease;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        box-shadow: 0 4px 15px rgba(0, 206, 201, 0.3);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
        box-shadow: 0 4px 15px rgba(253, 203, 110, 0.3);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
        box-shadow: 0 4px 15px rgba(225, 112, 85, 0.3);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        padding: 0.8rem 2rem !important;
        border: none !important;
        border-radius: 50px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4) !important;
        width: 100% !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1d391kg {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        background: rgba(255, 255, 255, 0.25);
        transform: translateX(5px);
    }
    
    .info-card h4 {
        color: #2d3436;
        margin-bottom: 0.5rem;
        font-weight: 600;
    }
    
    .info-card p {
        color: rgba(45, 52, 54, 0.8);
        margin: 0.25rem 0;
    }
    
    /* Metrics */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .metric-label {
        color: rgba(45, 52, 54, 0.7);
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        border-radius: 50px !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
        margin-top: 2rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.3) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading-text {
        animation: pulse 1.5s ease-in-out infinite;
    }
    </style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============ LOAD MODEL ============
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
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
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
            "treatment": "Methicillin (if MSSA), Vancomycin (if MRSA)",
            "emoji": "🟡"
        },
        "Staphylococcus_saprophyticus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "Usually γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin resistant",
            "color": "White to cream",
            "size": "1-3 mm",
            "description": "Smooth, convex, opaque colonies; usually non-pigmented",
            "virulence": "Moderate",
            "treatment": "Trimethoprim-sulfamethoxazole, Nitrofurantoin",
            "emoji": "⚪"
        },
        "Staphylococcus_epidermidis": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin sensitive",
            "color": "White to grayish-white",
            "size": "Small",
            "description": "Small, smooth, circular, convex, non-pigmented, glossy colonies",
            "virulence": "Low (Opportunistic)",
            "treatment": "Vancomycin, Rifampin",
            "emoji": "🔘"
        },
        "Streptococcus_pneumoniae": {
            "gram_stain": "Gram-positive lancet-shaped diplococci",
            "hemolysis": "α-hemolytic",
            "key_id": "Optochin sensitive, bile soluble",
            "color": "Gray, translucent",
            "size": "Small",
            "description": "Small, glistening, mucoid colonies; older colonies have central depression",
            "virulence": "High",
            "treatment": "Penicillin, Ceftriaxone, Vancomycin",
            "emoji": "🟣"
        },
        "Streptococcus_pyogenes": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Strong β-hemolytic",
            "key_id": "Bacitracin sensitive, PYR positive",
            "color": "Grayish-white, translucent",
            "size": "Tiny (0.5-1 mm)",
            "description": "Small, translucent, pinpoint colonies with strong beta hemolysis",
            "virulence": "High",
            "treatment": "Penicillin, Amoxicillin",
            "emoji": "🔴"
        },
        "Streptococcus_agalactiae": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Narrow β-hemolytic",
            "key_id": "CAMP positive, Hippurate positive",
            "color": "Grayish-white to cream",
            "size": "Medium",
            "description": "Medium-sized, smooth, grayish-white to cream, slightly mucoid colonies",
            "virulence": "Moderate",
            "treatment": "Penicillin, Ampicillin",
            "emoji": "🟢"
        }
    }
    return info.get(bacteria_name, {})

# ============ MAIN APP ============
def main():
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 5rem; margin-bottom: -1rem;">🧫</div>
            <h1 class="main-title">Bacteria Colony Classifier</h1>
            <p class="sub-title">Upload an image of a bacterial colony to identify the species</p>
        </div>
    """, unsafe_allow_html=True)
    
    session, class_names = load_onnx_model()
    
    if session is None:
        st.error("⚠️ Please make sure 'bacteria_classifier.onnx' and 'class_names.json' are in the same directory.")
        return
    
    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Upload Section
        uploaded_file = st.file_uploader(
            " ",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            # Display image
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Classify button
            if st.button("🔬 Classify Bacteria", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    time.sleep(0.5)  # Smooth animation
                    predicted, confidence, all_predictions = predict_image(
                        image, session, class_names
                    )
                    
                    # Determine confidence level
                    if confidence > 70:
                        conf_level = "High"
                        conf_emoji = "✅"
                        conf_color = "confidence-high"
                        badge = "badge-success"
                    elif confidence > 50:
                        conf_level = "Medium"
                        conf_emoji = "⚠️"
                        conf_color = "confidence-medium"
                        badge = "badge-warning"
                    else:
                        conf_level = "Low"
                        conf_emoji = "❌"
                        conf_color = "confidence-low"
                        badge = "badge-danger"
                    
                    # Store in history
                    st.session_state.prediction_history.append({
                        'predicted': predicted,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                    
                    # Prediction Box
                    info = get_bacteria_info(predicted)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 3rem; margin-bottom: -0.5rem;">{info.get('emoji', '🧬')}</div>
                        <p style="color: rgba(45,52,54,0.6); margin: 0;">Predicted Species</p>
                        <h2>{predicted.replace('_', ' ')}</h2>
                        <div class="{conf_color}" style="font-size: 2rem; font-weight: 700; margin: 0.5rem 0;">
                            {confidence:.1f}%
                        </div>
                        <div>
                            <span class="badge {badge}">{conf_emoji} {conf_level} Confidence</span>
                        </div>
                        <div style="margin-top: 1rem;">
                            <p style="margin: 0; color: rgba(45,52,54,0.7); font-size: 0.9rem;">
                                {info.get('description', '')}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Bar
                    st.progress(int(confidence))
                    
                    # Morphology Info
                    if info:
                        with st.expander("📊 Morphology & Clinical Characteristics", expanded=True):
                            tab1, tab2, tab3 = st.tabs(["🔬 Morphology", "🦠 Clinical", "💊 Treatment"])
                            
                            with tab1:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>🔬 Microscopic Features</h4>
                                    <p><strong>Gram Stain:</strong> {info['gram_stain']}</p>
                                    <p><strong>Hemolysis:</strong> {info['hemolysis']}</p>
                                    <p><strong>Colony Color:</strong> {info['color']}</p>
                                    <p><strong>Colony Size:</strong> {info['size']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with tab2:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>🦠 Clinical Information</h4>
                                    <p><strong>Virulence:</strong> {info['virulence']}</p>
                                    <p><strong>Key Identification:</strong> {info['key_id']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with tab3:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>💊 Treatment</h4>
                                    <p>{info['treatment']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Confidence Chart
                    st.markdown("### 📈 Confidence Scores")
                    
                    sorted_preds = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                    
                    colors = ['#00b894' if x == predicted else '#dfe6e9' for x in sorted_preds.keys()]
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(sorted_preds.keys()),
                            y=list(sorted_preds.values()),
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in sorted_preds.values()],
                            textposition='outside',
                            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
                        )
                    ])
                    
                    fig.update_layout(
                        xaxis_title="Bacteria Species",
                        yaxis_title="Confidence (%)",
                        yaxis_range=[0, 100],
                        height=450,
                        xaxis_tickangle=-45,
                        showlegend=False,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=20, r=20, t=20, b=80),
                        hovermode='closest'
                    )
                    
                    fig.update_traces(
                        marker_line_color='rgba(0,0,0,0.1)',
                        marker_line_width=1,
                        opacity=0.9
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Sidebar Content
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🧫</div>
            <h3 style="font-weight: 700; margin: 0;">Bacteria AI</h3>
            <p style="color: rgba(45,52,54,0.6); font-size: 0.9rem;">Powered by ONNX Runtime</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Model Performance
        st.markdown("""
        <div style="text-align: center;">
            <div class="metric-value">82%</div>
            <div class="metric-label">Model Accuracy</div>
            <div style="margin-top: 0.5rem;">
                <span class="badge badge-info">6 Species</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Available Species
        st.markdown("### 🦠 Available Species")
        for species in class_names:
            display_name = species.replace('_', ' ')
            info = get_bacteria_info(species)
            emoji = info.get('emoji', '🦠')
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0;">
                <span>{emoji}</span>
                <span style="font-size: 0.9rem;">{display_name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # History
        if st.session_state.prediction_history:
            st.markdown("### 📜 Recent Predictions")
            for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                conf_color = "🟢" if pred['confidence'] > 70 else "🟡" if pred['confidence'] > 50 else "🔴"
                st.markdown(f"""
                <div class="info-card" style="padding: 0.75rem; margin: 0.25rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span><strong>{pred['predicted'].replace('_', ' ')}</strong></span>
                        <span>{conf_color} {pred['confidence']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Instructions
        st.divider()
        st.markdown("""
        <div style="font-size: 0.85rem; color: rgba(45,52,54,0.6);">
            <p>📤 Upload a clear image</p>
            <p>🔬 Click Classify</p>
            <p>📊 View detailed analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🧫 Bacteria Colony Classifier | Built with Streamlit, ONNX Runtime & ❤️</p>
        <p style="font-size: 0.8rem; opacity: 0.7;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                      
          

     
                   
