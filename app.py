import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy import ndimage
import hashlib
import json

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

# ============ CSS ============
st.markdown("""
<style>
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
.glass-card {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 3px solid #8bc34a;
    padding: 2rem;
    box-shadow: 0 8px 32px rgba(139, 195, 74, 0.25);
}
.main-title {
    text-align: center;
    font-size: 3.5rem;
    font-weight: 900;
    color: #000000;
}
.main-title .green-yellow-text {
    background: linear-gradient(90deg, #558b2f, #f9a825);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.upload-area {
    border: 3px dashed #8bc34a;
    border-radius: 24px;
    padding: 3rem 2rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.75);
    cursor: pointer;
}
.upload-area:hover {
    background: rgba(255, 255, 255, 0.9);
    border-color: #558b2f;
}
.prediction-box {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    border: 3px solid #8bc34a;
    box-shadow: 0 8px 32px rgba(139, 195, 74, 0.25);
    animation: fadeInUp 0.6s ease;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.prediction-box h2 {
    font-size: 3rem;
    font-weight: 900;
    color: #000000;
    margin: 0.5rem 0;
}
.confidence-high { color: #2e7d32; font-weight: 900; font-size: 2.2rem; }
.confidence-medium { color: #f9a825; font-weight: 900; font-size: 2.2rem; }
.confidence-low { color: #c62828; font-weight: 900; font-size: 2.2rem; }
.badge {
    display: inline-block;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    font-weight: 800;
    border: 2px solid #000000;
}
.badge-success { background: #a5d6a7; }
.badge-warning { background: #ffe082; }
.badge-danger { background: #ef9a9a; }
.stButton > button {
    background: linear-gradient(90deg, #66bb6a, #ffd54f);
    color: #000000 !important;
    font-weight: 900;
    font-size: 1.3rem;
    padding: 1rem 2.5rem;
    border: 3px solid #000000;
    border-radius: 50px;
    width: 100%;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 30px rgba(102, 187, 106, 0.4);
}
.stProgress > div > div {
    background: linear-gradient(90deg, #66bb6a, #ffd54f) !important;
    border-radius: 50px;
    height: 14px;
    border: 2px solid #000000;
}
.info-card {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 16px;
    padding: 1.5rem;
    border: 2px solid #8bc34a;
    margin: 0.5rem 0;
}
.footer {
    text-align: center;
    padding: 2rem 0;
    color: #000000;
    font-weight: 700;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ============ CLASS NAMES ============
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

# ============ SESSION STATE ============
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# ============ FEATURE EXTRACTION ============
def extract_features_from_image(image):
    """Extract simple color and texture features from image"""
    try:
        # Convert to numpy array
        img_array = np.array(image)
        
        # Resize to standard size
        from PIL import Image
        img = image.resize((128, 128))
        img_array = np.array(img)
        
        # Color features
        features = []
        
        # Mean and std for each channel
        for channel in range(3):
            channel_data = img_array[:, :, channel].flatten()
            features.append(np.mean(channel_data))
            features.append(np.std(channel_data))
        
        # Color histograms (8 bins each)
        for channel in range(3):
            hist, _ = np.histogram(img_array[:, :, channel], bins=8, range=(0, 256))
            features.extend(hist / np.sum(hist))
        
        # Texture features (simple gradient)
        gray = np.mean(img_array, axis=2)
        from scipy import ndimage
        sx = ndimage.sobel(gray, axis=0)
        sy = ndimage.sobel(gray, axis=1)
        gradient_mag = np.sqrt(sx**2 + sy**2)
        features.append(np.mean(gradient_mag))
        features.append(np.std(gradient_mag))
        
        return np.array(features).flatten()
    except Exception as e:
        return None

# ============ LOAD OR CREATE DATABASE ============
@st.cache_resource
def load_reference_database():
    """Load reference features or create from images if available"""
    try:
        # Try to load from pickle file
        if os.path.exists('reference_features.pkl'):
            with open('reference_features.pkl', 'rb') as f:
                data = pickle.load(f)
            
            # Check if data is valid
            if 'features' in data and 'labels' in data:
                return data['features'], data['labels']
    except Exception as e:
        st.warning(f"Could not load reference file: {e}")
    
    # If no reference file, create from local images folder
    if os.path.exists('reference_images'):
        st.info("📁 Loading reference images from 'reference_images' folder...")
        features = []
        labels = []
        
        for species in class_names:
            species_path = os.path.join('reference_images', species)
            if os.path.exists(species_path):
                for img_file in os.listdir(species_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img_path = os.path.join(species_path, img_file)
                            img = Image.open(img_path)
                            feat = extract_features_from_image(img)
                            if feat is not None:
                                features.append(feat)
                                labels.append(species)
                        except:
                            continue
        
        if features:
            # Save for future use
            try:
                data = {'features': np.array(features), 'labels': np.array(labels)}
                with open('reference_features.pkl', 'wb') as f:
                    pickle.dump(data, f)
            except:
                pass
            return np.array(features), np.array(labels)
    
    return None, None

# ============ PREDICT ============
def predict_image(image, ref_features, ref_labels):
    """Predict bacteria by comparing with reference images"""
    try:
        # Extract features from uploaded image
        query_features = extract_features_from_image(image)
        
        if query_features is None:
            return None, 0, {}
        
        # Ensure both are numpy arrays
        query_features = np.array(query_features).reshape(1, -1)
        ref_features = np.array(ref_features)
        
        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_features, ref_features)[0]
        
        # Get predictions per species
        predictions = {}
        for species in class_names:
            indices = [i for i, label in enumerate(ref_labels) if label == species]
            if indices:
                predictions[species] = np.mean(similarities[indices]) * 100
            else:
                predictions[species] = 0
        
        predicted = max(predictions, key=predictions.get)
        confidence = predictions[predicted]
        
        return predicted, confidence, predictions
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, 0, {}

# ============ GET BACTERIA INFO ============
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

# ============ MAIN ============
def main():
    # Load reference database
    ref_features, ref_labels = load_reference_database()
    
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="font-size: 4rem;">🧫</div>
            <h1 class="main-title">
                Bacteria <span class="green-yellow-text">Colony</span> Classifier
            </h1>
            <p style="font-size: 1.2rem; color: #333; font-weight: 700;">🔬 Upload an image to identify bacteria species</p>
        </div>
    """, unsafe_allow_html=True)

    if ref_features is None or len(ref_features) == 0:
        st.warning("⚠️ Reference database not found. Please upload 'reference_features.pkl' to the app directory.")
        st.info("""
        ### How to get the file:
        1. Run the feature extraction code in Colab
        2. Download the `reference_features.pkl` file
        3. Upload it to this app directory on GitHub
        """)
        return

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("""
        <div class="upload-area">
            <span class="upload-icon">📸</span>
            <h3 style="color: #000000;">Drop your bacterial colony image here</h3>
            <p style="color: #333; font-weight: 600;">or click to browse (JPG, PNG, TIFF)</p>
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

            if st.button("🔬 Identify Bacteria", use_container_width=True):
                with st.spinner("🔍 Comparing with reference images..."):
                    time.sleep(0.5)
                    predicted, confidence, all_predictions = predict_image(
                        image, ref_features, ref_labels
                    )

                    if predicted is None:
                        st.error("❌ Could not identify the bacteria. Please try a clearer image.")
                        st.stop()

                    display_name = display_names.get(predicted, predicted.replace('_', ' '))

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
                        <div style="font-size: 3.5rem;">{info.get('emoji', '🧬')}</div>
                        <p style="color: #000000; font-weight: 700; margin: 0;">Identified Species</p>
                        <h2>{display_name}</h2>
                        <div class="{conf_color}">{confidence:.1f}%</div>
                        <div>
                            <span class="badge {badge}">{conf_emoji} {conf_level} Confidence</span>
                            <span class="badge" style="background: linear-gradient(90deg, #a5d6a7, #ffe082); border-color: #000000;">📊 Image Similarity</span>
                        </div>
                        <div style="margin-top: 1rem;">
                            <p style="color: #000000; font-weight: 600; margin: 0; font-size: 1rem;">
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

                    st.markdown("### 📈 Similarity Scores")
                    sorted_preds = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                    display_preds = {display_names.get(k, k.replace('_', ' ')): v for k, v in sorted_preds.items()}

                    colors = ['#2e7d32' if i==0 else '#bdbdbd' for i in range(len(display_preds))]

                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(display_preds.keys()),
                            y=list(display_preds.values()),
                            marker_color=colors,
                            text=[f"{v:.1f}%" for v in display_preds.values()],
                            textposition='outside',
                            textfont=dict(color='#000000', size=12, weight='bold')
                        )
                    ])

                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        showlegend=False,
                        yaxis_range=[0, 100],
                        plot_bgcolor='rgba(255,255,255,0.7)',
                        paper_bgcolor='rgba(255,255,255,0.3)',
                        font=dict(color='#000000', weight='bold')
                    )

                    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🧫</div>
            <h3 style="font-weight: 900; color: #000000;">Bacteria AI</h3>
            <p style="color: #000000; font-weight: 700; font-size: 0.9rem;">🔍 Image Similarity Matching</p>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 900; color: #000000;">6</div>
            <div style="color: #000000; font-weight: 700;">Species</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 🦠 Available Species")
        for species in class_names:
            display_name = display_names.get(species, species.replace('_', ' '))
            info = get_bacteria_info(species)
            emoji = info.get('emoji', '🦠')
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.2rem 0; 
                        color: #000000; font-weight: 700;">
                <span>{emoji}</span>
                <span>{display_name}</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        if st.session_state.prediction_history:
            st.markdown("### 📜 Recent Predictions")
            emojis = ['🌿', '🍀', '🌱', '💚', '💛']
            for i, pred in enumerate(st.session_state.prediction_history[-5:]):
                emoji = emojis[i % len(emojis)]
                display_name = display_names.get(pred['predicted'], pred['predicted'].replace('_', ' '))
                st.markdown(f"""
                <div class="info-card" style="padding: 0.5rem; margin: 0.25rem 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; color: #000000;">
                        <span style="font-weight: 800;">{display_name}</span>
                        <span style="font-weight: 700;">{pred['confidence']:.0f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
        st.markdown("""
        <div style="font-size: 0.9rem; color: #000000; font-weight: 700;">
            <p>📤 Upload a clear image</p>
            <p>🔬 Click Identify</p>
            <p>📊 View similarity scores</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer">
        <p style="font-size: 1.1rem; font-weight: 800;">🧫 Bacteria Colony Classifier | Built with ❤️ & 🌿</p>
        <p style="font-size: 0.85rem; font-weight: 600; opacity: 0.8;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
