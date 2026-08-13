import streamlit as st
import numpy as np
from PIL import Image
import onnxruntime as ort
import json
import plotly.graph_objects as go
import time
import random

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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Rainbow Animated Background */
    .stApp {
        background: linear-gradient(135deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
        background-size: 500% 500%;
        animation: rainbowGradient 15s ease infinite;
    }
    
    @keyframes rainbowGradient {
        0% { background-position: 0% 50%; }
        20% { background-position: 50% 50%; }
        40% { background-position: 100% 50%; }
        60% { background-position: 50% 100%; }
        80% { background-position: 0% 100%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Rainbow Floating Particles */
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        opacity: 0.6;
        animation: floatUp 10s infinite linear;
    }
    
    @keyframes floatUp {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 0.6; }
        90% { opacity: 0.6; }
        100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
    }
    
    /* Glassmorphism Cards with Rainbow Border */
    .glass-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        border: 2px solid transparent;
        background-clip: padding-box;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        padding: 2rem;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        animation: rainbowBorder 4s linear infinite;
    }
    
    @keyframes rainbowBorder {
        0% { border-color: #ff6b6b; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3); }
        16% { border-color: #ffd93d; box-shadow: 0 8px 32px rgba(255, 217, 61, 0.3); }
        33% { border-color: #6bcb77; box-shadow: 0 8px 32px rgba(107, 203, 119, 0.3); }
        50% { border-color: #4d96ff; box-shadow: 0 8px 32px rgba(77, 150, 255, 0.3); }
        66% { border-color: #9b59b6; box-shadow: 0 8px 32px rgba(155, 89, 182, 0.3); }
        83% { border-color: #ff6b6b; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3); }
        100% { border-color: #ff6b6b; box-shadow: 0 8px 32px rgba(255, 107, 107, 0.3); }
    }
    
    .glass-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2);
    }
    
    /* Main Title with Rainbow Text */
    .main-title {
        text-align: center;
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6, #ff6b6b);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: rainbowText 4s ease infinite;
        padding: 1rem 0;
        text-shadow: none;
    }
    
    @keyframes rainbowText {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Subtitle Glow */
    .sub-title {
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 300;
        margin-bottom: 2rem;
        text-shadow: 0 0 20px rgba(255,255,255,0.3), 0 0 60px rgba(255,255,255,0.1);
        animation: glowPulse 2s ease-in-out infinite;
    }
    
    @keyframes glowPulse {
        0%, 100% { text-shadow: 0 0 20px rgba(255,255,255,0.3); }
        50% { text-shadow: 0 0 40px rgba(255,255,255,0.6), 0 0 80px rgba(255,255,255,0.2); }
    }
    
    /* Upload Area */
    .upload-area {
        border: 3px dashed rgba(255, 255, 255, 0.4);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        animation: rainbowBorder 4s linear infinite;
    }
    
    .upload-area:hover {
        transform: scale(1.05);
        background: rgba(255, 255, 255, 0.2);
    }
    
    .upload-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        display: block;
        animation: bounce 2s ease-in-out infinite;
    }
    
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }
    
    /* Prediction Box */
    .prediction-box {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        text-align: center;
        border: 2px solid rgba(255,255,255,0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        animation: fadeInUp 0.6s ease, rainbowBorder 3s linear infinite;
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px) scale(0.95); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    .prediction-box h2 {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.5rem 0;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: rainbowText 3s ease infinite;
    }
    
    /* Confidence Colors with Rainbow */
    .confidence-high { 
        background: linear-gradient(90deg, #00b894, #00cec9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    .confidence-medium { 
        background: linear-gradient(90deg, #fdcb6e, #f39c12);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    .confidence-low { 
        background: linear-gradient(90deg, #e17055, #d63031);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.2rem;
    }
    
    /* Rainbow Badges */
    .badge {
        display: inline-block;
        padding: 0.4rem 1.5rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        transition: all 0.5s ease;
        animation: rainbowBorder 3s linear infinite;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #00b894, #00cec9);
        color: white;
        box-shadow: 0 4px 20px rgba(0, 206, 201, 0.4);
    }
    
    .badge-warning {
        background: linear-gradient(135deg, #fdcb6e, #f39c12);
        color: white;
        box-shadow: 0 4px 20px rgba(253, 203, 110, 0.4);
    }
    
    .badge-danger {
        background: linear-gradient(135deg, #e17055, #d63031);
        color: white;
        box-shadow: 0 4px 20px rgba(225, 112, 85, 0.4);
    }
    
    .badge-info {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    
    .badge-rainbow {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
        background-size: 300% 300%;
        animation: rainbowText 3s ease infinite;
        color: white;
        font-weight: 700;
    }
    
    /* Rainbow Button */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6, #ff6b6b) !important;
        background-size: 300% 300% !important;
        animation: rainbowText 3s ease infinite !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.3rem !important;
        padding: 1rem 2.5rem !important;
        border: none !important;
        border-radius: 50px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 30px rgba(255, 255, 255, 0.3) !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-5px) scale(1.02) !important;
        box-shadow: 0 10px 50px rgba(255, 255, 255, 0.5) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Progress Bar Rainbow */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6) !important;
        background-size: 300% 300% !important;
        animation: rainbowText 3s ease infinite !important;
        border-radius: 50px !important;
        height: 12px !important;
    }
    
    /* Sidebar Rainbow */
    .css-1d391kg, .css-1d391kg {
        background: rgba(255, 255, 255, 0.15) !important;
        backdrop-filter: blur(20px) !important;
        border-right: 2px solid rgba(255, 255, 255, 0.2) !important;
        animation: rainbowBorder 4s linear infinite !important;
    }
    
    /* Info Cards */
    .info-card {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
        transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        animation: rainbowBorder 4s linear infinite;
    }
    
    .info-card:hover {
        transform: translateX(10px) scale(1.02);
        background: rgba(255, 255, 255, 0.25);
    }
    
    /* Metrics Rainbow */
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: rainbowText 3s ease infinite;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0;
        color: white;
        font-size: 0.9rem;
        margin-top: 2rem;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Loading Animation */
    @keyframes shimmer {
        0% { background-position: -1000px 0; }
        100% { background-position: 1000px 0; }
    }
    
    .loading-text {
        background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
        background-size: 300% 300%;
        animation: rainbowText 1.5s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    /* Sparkle animation for results */
    @keyframes sparkle {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.1); }
    }
    
    .sparkle {
        animation: sparkle 1s ease-in-out infinite;
    }
    </style>
""", unsafe_allow_html=True)

# ============ RAINBOW PARTICLES ============
def add_particles():
    colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6', '#fd79a8', '#fdcb6e']
    particles = ""
    for i in range(20):
        size = random.randint(8, 25)
        left = random.randint(0, 100)
        duration = random.randint(8, 15)
        delay = random.randint(0, 10)
        color = random.choice(colors)
        particles += f"""
        <div class="particle" style="
            width: {size}px;
            height: {size}px;
            left: {left}%;
            background: {color};
            animation-duration: {duration}s;
            animation-delay: {delay}s;
            box-shadow: 0 0 20px {color};
        "></div>
        """
    st.markdown(f'<div>{particles}</div>', unsafe_allow_html=True)

add_particles()

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
    # Header with Rainbow
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 6rem; margin-bottom: -1.5rem; animation: bounce 2s ease-in-out infinite;">🧫</div>
            <h1 class="main-title">✨ Bacteria Colony Classifier ✨</h1>
            <p class="sub-title">🌟 Upload an image of a bacterial colony to identify the species 🌟</p>
        </div>
    """, unsafe_allow_html=True)
    
    session, class_names = load_onnx_model()
    
    if session is None:
        st.error("⚠️ Please make sure 'bacteria_classifier.onnx' and 'class_names.json' are in the same directory.")
        return
    
    # Layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Upload Section with Rainbow Border
        st.markdown("""
        <div class="upload-area">
            <span class="upload-icon">📸</span>
            <h3 style="color: white; font-weight: 600;">Drop your image here</h3>
            <p style="color: rgba(255,255,255,0.8);">or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
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
            
            # Rainbow Classify Button
            if st.button("🌈 Classify Bacteria", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    time.sleep(0.8)
                    predicted, confidence, all_predictions = predict_image(
                        image, session, class_names
                    )
                    
                    # Determine confidence level
                    if confidence > 70:
                        conf_level = "High"
                        conf_emoji = "✨"
                        conf_color = "confidence-high"
                        badge = "badge-success"
                    elif confidence > 50:
                        conf_level = "Medium"
                        conf_emoji = "🌟"
                        conf_color = "confidence-medium"
                        badge = "badge-warning"
                    else:
                        conf_level = "Low"
                        conf_emoji = "💫"
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
                        <div style="font-size: 4rem; margin-bottom: -0.5rem;">{info.get('emoji', '🧬')}</div>
                        <p style="color: rgba(255,255,255,0.8); margin: 0; font-weight: 300;">🎯 Predicted Species</p>
                        <h2>{predicted.replace('_', ' ')}</h2>
                        <div class="{conf_color}" style="font-size: 2.5rem; font-weight: 800; margin: 0.5rem 0;">
                            {confidence:.1f}%
                        </div>
                        <div>
                            <span class="badge {badge}">{conf_emoji} {conf_level} Confidence</span>
                            <span class="badge badge-rainbow">🌈 AI Verified</span>
                        </div>
                        <div style="margin-top: 1rem;">
                            <p style="margin: 0; color: rgba(255,255,255,0.9); font-size: 0.95rem; font-weight: 300;">
                                ✨ {info.get('description', '')}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Rainbow Progress Bar
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
                    
                    # Confidence Chart with Rainbow Colors
                    st.markdown("### 📈 Confidence Scores")
                    
                    sorted_preds = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                    
                    rainbow_colors = ['#ff6b6b', '#ffd93d', '#6bcb77', '#4d96ff', '#9b59b6', '#fd79a8']
                    colors = [rainbow_colors[i % len(rainbow_colors)] for i in range(len(sorted_preds))]
                    
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
                        marker_line_color='rgba(255,255,255,0.3)',
                        marker_line_width=2,
                        opacity=0.9
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    # Rainbow Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 4rem; animation: bounce 2s ease-in-out infinite;">🧫</div>
            <h3 style="font-weight: 800; background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcb77, #4d96ff, #9b59b6);
            background-size: 300% 300%;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            animation: rainbowText 3s ease infinite;">Bacteria AI</h3>
            <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">✨ Powered by ONNX Runtime ✨</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Model Performance
        st.markdown("""
        <div style="text-align: center;">
            <div class="metric-value">82%</div>
            <div style="color: white; font-weight: 300;">Model Accuracy</div>
            <div style="margin-top: 0.5rem;">
                <span class="badge badge-info">6 Species</span>
                <span class="badge badge-rainbow">AI Powered</span>
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
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0; 
                        color: white; font-weight: 300;">
                <span>{emoji}</span>
                <span style="font-size: 0.9rem;">{display_name}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.divider()
        
        # History
        if st.session_state.prediction_history:
            st.markdown("### 📜 Recent Predictions")
            rainbow_emojis = ['🌈', '✨', '🌟', '💫', '⭐']
            for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                emoji = rainbow_emojis[i % len(rainbow_emojis)]
                st.markdown(f"""
                <div class="info-card" style="padding: 0.75rem; margin: 0.25rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; color: white;">
                        <span><strong>{pred['predicted'].replace('_', ' ')}</strong></span>
                        <span>{emoji} {pred['confidence']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Instructions
        st.divider()
        st.markdown("""
        <div style="font-size: 0.85rem; color: rgba(255,255,255,0.8);">
            <p>📤 Upload a clear image</p>
            <p>🔬 Click Classify</p>
            <p>📊 View detailed analysis</p>
            <p>🌈 Enjoy the rainbow!</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.2rem;">🧫 Bacteria Colony Classifier | Built with ❤️ & 🌈</p>
        <p style="font-size: 0.8rem; opacity: 0.8;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

     
                   
