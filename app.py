import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #d4edda 0%, #fff3cd 25%, #d4edda 50%, #fff3cd 75%, #d4edda 100%);
    background-size: 400% 400%;
    animation: softGradient 15s ease infinite;
}

@keyframes softGradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stMarkdown, .stText, p, li, label { color: #000000 !important; font-weight: 700 !important; }

.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 3px solid #8bc34a;
    box-shadow: 0 8px 32px rgba(139, 195, 74, 0.25);
    padding: 2rem;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
    transform: translateY(-8px) scale(1.01);
    box-shadow: 0 20px 60px rgba(139, 195, 74, 0.35);
}

.main-title {
    text-align: center;
    font-size: 4.2rem;
    font-weight: 900;
    color: #000000 !important;
    text-shadow: 0 4px 20px rgba(139, 195, 74, 0.2);
    padding: 1rem 0;
}

.main-title .green-yellow-text {
    background: linear-gradient(90deg, #558b2f, #f9a825, #558b2f);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: greenYellowText 4s ease infinite;
}

@keyframes greenYellowText {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.upload-area {
    border: 3px dashed #8bc34a;
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(10px);
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    cursor: pointer;
}

.upload-area:hover {
    transform: scale(1.03);
    background: rgba(255, 255, 255, 0.9);
    border-color: #558b2f;
}

.upload-icon {
    font-size: 5rem;
    margin-bottom: 1rem;
    display: block;
    animation: bounce 2s ease-in-out infinite;
}

@keyframes bounce {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-15px); }
}

.prediction-box {
    background: rgba(255, 255, 255, 0.92);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    border: 3px solid #8bc34a;
    box-shadow: 0 8px 32px rgba(139, 195, 74, 0.25);
    animation: fadeInUp 0.6s ease;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px) scale(0.95); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.prediction-box h2 {
    font-size: 3rem;
    font-weight: 900 !important;
    color: #000000 !important;
    margin: 0.5rem 0;
}

.confidence-high { color: #2e7d32 !important; font-weight: 900 !important; font-size: 2.2rem; }
.confidence-medium { color: #f9a825 !important; font-weight: 900 !important; font-size: 2.2rem; }
.confidence-low { color: #c62828 !important; font-weight: 900 !important; font-size: 2.2rem; }

.badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 800 !important;
    font-size: 0.95rem;
    margin: 0.25rem;
    color: #000000 !important;
    border: 2px solid #000000;
}

.badge-success {
    background: linear-gradient(135deg, #a5d6a7, #66bb6a);
    box-shadow: 0 4px 20px rgba(102, 187, 106, 0.3);
}

.badge-warning {
    background: linear-gradient(135deg, #ffe082, #ffd54f);
    box-shadow: 0 4px 20px rgba(255, 213, 79, 0.3);
}

.badge-danger {
    background: linear-gradient(135deg, #ef9a9a, #ef5350);
    box-shadow: 0 4px 20px rgba(239, 83, 80, 0.3);
}

.badge-green-yellow {
    background: linear-gradient(90deg, #a5d6a7, #ffe082, #a5d6a7);
    background-size: 300% 300%;
    animation: greenYellowText 3s ease infinite;
    border-color: #000000;
}

.stButton > button {
    background: linear-gradient(90deg, #66bb6a, #ffd54f, #66bb6a) !important;
    background-size: 300% 300% !important;
    animation: greenYellowText 3s ease infinite !important;
    color: #000000 !important;
    font-weight: 900 !important;
    font-size: 1.3rem !important;
    padding: 1rem 2.5rem !important;
    border: 3px solid #000000 !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 30px rgba(102, 187, 106, 0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-5px) scale(1.02) !important;
    box-shadow: 0 10px 50px rgba(102, 187, 106, 0.4) !important;
}

.stProgress > div > div {
    background: linear-gradient(90deg, #66bb6a, #ffd54f, #66bb6a) !important;
    background-size: 300% 300% !important;
    animation: greenYellowText 3s ease infinite !important;
    border-radius: 50px !important;
    height: 14px !important;
    border: 2px solid #000000 !important;
}

.info-card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 16px;
    padding: 1.5rem;
    border: 2px solid #8bc34a;
    margin: 0.5rem 0;
    transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.info-card:hover {
    transform: translateX(8px) scale(1.01);
    background: rgba(255, 255, 255, 0.95);
}

.info-card h4 { color: #000000 !important; font-weight: 800 !important; }
.info-card p { color: #000000 !important; font-weight: 600 !important; }
.info-card strong { color: #000000 !important; font-weight: 900 !important; }

.footer {
    text-align: center;
    padding: 2rem 0;
    color: #000000 !important;
    font-size: 1rem;
    font-weight: 700 !important;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ============ FLOATING PARTICLES ============
def add_particles():
    colors = ['#a5d6a7', '#66bb6a', '#ffe082', '#ffd54f', '#8bc34a', '#f9a825']
    particles_html = """
    <style>
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: 0;
        animation: floatUp linear infinite;
        opacity: 0.10;
    }
    @keyframes floatUp {
        0% { transform: translateY(100vh) rotate(0deg); opacity: 0; }
        10% { opacity: 0.10; }
        90% { opacity: 0.10; }
        100% { transform: translateY(-10vh) rotate(720deg); opacity: 0; }
    }
    .particle-container {
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none; z-index: 0; overflow: hidden;
    }
    </style>
    <div class="particle-container">
    """
    for i in range(15):
        size = random.randint(5, 15)
        left = random.randint(0, 100)
        duration = random.randint(15, 25)
        delay = random.randint(0, 15)
        color = random.choice(colors)
        particles_html += f"""
        <div class="particle" style="
            width: {size}px; height: {size}px; left: {left}%;
            background: {color};
            animation-duration: {duration}s;
            animation-delay: {delay}s;
            box-shadow: 0 0 15px {color};
        "></div>
        """
    particles_html += "</div>"
    import streamlit.components.v1 as components
    components.html(particles_html, height=0)

add_particles()

# ============ SESSION STATE ============
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============ LOAD TFLITE MODEL ============
@st.cache_resource
def load_tflite_model():
    try:
        interpreter = tf.lite.Interpreter(model_path='bacteria_classifier.tflite')
        interpreter.allocate_tensors()
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        return interpreter, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None

# ============ PREDICTION FUNCTION ============
def predict_image(image, interpreter, class_names):
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
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
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="font-size: 5rem; margin-bottom: -1.5rem; animation: bounce 2s ease-in-out infinite;">🧫</div>
            <h1 class="main-title">
                Bacteria <span class="green-yellow-text">Colony</span> Classifier
            </h1>
            <p class="sub-title">🔬 Upload an image to identify bacteria species with AI-powered precision</p>
        </div>
    """, unsafe_allow_html=True)

    interpreter, class_names = load_tflite_model()

    if interpreter is None:
        st.warning("⚠️ Please make sure 'bacteria_classifier.tflite' and 'class_names.json' are in the same directory.")
        return

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class="upload-area">
            <span class="upload-icon">📸</span>
            <h3>Drop your bacterial colony image here</h3>
            <p>or click to browse files (JPG, PNG, TIFF)</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            " ",
            type=['jpg', 'jpeg', 'png', 'tiff', 'bmp'],
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("🔬 Classify Bacteria 🧫", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    time.sleep(0.5)
                    predicted, confidence, all_predictions = predict_image(
                        image, interpreter, class_names
                    )

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

                    st.session_state.prediction_history.append({
                        'predicted': predicted,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })

                    info = get_bacteria_info(predicted)

                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 3.5rem; margin-bottom: -0.5rem;">{info.get('emoji', '🧬')}</div>
                        <p style="color: #000000 !important; font-weight: 700 !important; margin: 0;">Predicted Species</p>
                        <h2>{predicted.replace('_', ' ')}</h2>
                        <div class="{conf_color}" style="font-size: 2.5rem; font-weight: 900;">
                            {confidence:.1f}%
                        </div>
                        <div>
                            <span class="badge {badge}">{conf_emoji} {conf_level} Confidence</span>
                            <span class="badge badge-green-yellow">🌿 AI Verified</span>
                        </div>
                        <div style="margin-top: 1rem;">
                            <p style="color: #000000 !important; font-weight: 600 !important; margin: 0; font-size: 1rem;">
                                {info.get('description', '')}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(int(confidence))

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

                    st.markdown("### 📈 Confidence Scores")
                    sorted_preds = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))

                    green_yellow_colors = ['#66bb6a', '#8bc34a', '#a5d6a7', '#ffd54f', '#ffe082', '#f9a825']
                    colors = [green_yellow_colors[i % len(green_yellow_colors)] for i in range(len(sorted_preds))]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(sorted_preds.keys()),
                            y=list(sorted_preds.values()),
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in sorted_preds.values()],
                            textposition='outside',
                            textfont=dict(color='#000000', size=14, weight='bold'),
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
                        plot_bgcolor='rgba(255,255,255,0.7)',
                        paper_bgcolor='rgba(255,255,255,0.3)',
                        margin=dict(l=20, r=20, t=20, b=80),
                        hovermode='closest',
                        font=dict(color='#000000', weight='bold')
                    )

                    fig.update_traces(
                        marker_line_color='#000000',
                        marker_line_width=2,
                        opacity=0.9
                    )

                    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3.5rem; animation: bounce 2s ease-in-out infinite;">🧫</div>
            <h3 style="font-weight: 900 !important; color: #000000 !important;">Bacteria AI</h3>
            <p style="color: #000000 !important; font-weight: 700 !important; font-size: 0.9rem;">🌿 Powered by TFLite</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="text-align: center;">
            <div class="metric-value">82%</div>
            <div class="metric-label">Model Accuracy</div>
            <div style="margin-top: 0.5rem;">
                <span class="badge badge-info">6 Species</span>
                <span class="badge badge-green-yellow">🌿 AI Powered</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🦠 Available Species")
        for species in class_names:
            display_name = species.replace('_', ' ')
            info = get_bacteria_info(species)
            emoji = info.get('emoji', '🦠')
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.3rem 0; 
                        color: #000000 !important; font-weight: 700 !important;">
                <span>{emoji}</span>
                <span style="font-size: 0.9rem; font-weight: 700 !important;">{display_name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if st.session_state.prediction_history:
            st.markdown("### 📜 Recent Predictions")
            green_yellow_emojis = ['🌿', '🍀', '🌱', '💚', '💛']
            for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                emoji = green_yellow_emojis[i % len(green_yellow_emojis)]
                st.markdown(f"""
                <div class="info-card" style="padding: 0.75rem; margin: 0.25rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; color: #000000 !important;">
                        <span style="font-weight: 800 !important;">{pred['predicted'].replace('_', ' ')}</span>
                        <span style="font-weight: 700 !important;">{emoji} {pred['confidence']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="font-size: 0.9rem; color: #000000 !important; font-weight: 700 !important;">
            <p>📤 Upload a clear image</p>
            <p>🔬 Click Classify</p>
            <p>📊 View detailed analysis</p>
            <p>🌿 Green & Yellow theme</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; font-weight: 800 !important;">🧫 Bacteria Colony Classifier | Built with ❤️ & 🌿</p>
        <p style="font-size: 0.85rem; font-weight: 600 !important; opacity: 0.8;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
