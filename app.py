import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import time
import random
import os
import cv2
from sklearn.cluster import KMeans
from scipy import stats
import colorsys

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

# ============ CUSTOM CSS ============
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
.stMarkdown, .stText, p, li, label { color: #000000 !important; font-weight: 700 !important; }
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

# ============ SESSION STATE ============
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

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

# ============ MORPHOLOGY DATABASE ============
morphology_db = {
    "Staphylococcus_aureus": {
        "color": "golden-yellow",
        "hemolysis": "β-hemolytic",
        "colony_description": "Round, smooth, convex, opaque, golden-yellow to cream",
        "size": "2-4 mm",
        "gram_stain": "Gram-positive cocci in clusters",
        "msa": "Yellow colonies with yellow medium (mannitol fermented)",
        "key_id": "Catalase+, Coagulase+, Mannitol fermenter",
        "emoji": "🟡",
        "treatment": "Methicillin (if MSSA), Vancomycin (if MRSA)",
        "virulence": "High"
    },
    "Staphylococcus_saprophyticus": {
        "color": "white-cream",
        "hemolysis": "γ-hemolytic",
        "colony_description": "White to cream, smooth, convex colonies",
        "size": "1-3 mm",
        "gram_stain": "Gram-positive cocci in clusters",
        "msa": "White colonies, medium remains pink (mannitol negative)",
        "key_id": "Catalase+, Coagulase−, Novobiocin resistant",
        "emoji": "⚪",
        "treatment": "Trimethoprim-sulfamethoxazole, Nitrofurantoin",
        "virulence": "Moderate"
    },
    "Staphylococcus_epidermidis": {
        "color": "white-grayish",
        "hemolysis": "γ-hemolytic",
        "colony_description": "Small, white, smooth colonies",
        "size": "Small",
        "gram_stain": "Gram-positive cocci in clusters",
        "msa": "Pink colonies, medium remains pink",
        "key_id": "Catalase+, Coagulase−, Novobiocin sensitive",
        "emoji": "🔘",
        "treatment": "Vancomycin, Rifampin",
        "virulence": "Low (Opportunistic)"
    },
    "Streptococcus_pneumoniae": {
        "color": "gray-translucent",
        "hemolysis": "α-hemolytic",
        "colony_description": "Small, glistening, mucoid; older colonies have central depression (draughtsman)",
        "size": "Small",
        "gram_stain": "Gram-positive lancet-shaped diplococci",
        "msa": "No growth",
        "key_id": "Optochin sensitive, bile soluble",
        "emoji": "🟣",
        "treatment": "Penicillin, Ceftriaxone, Vancomycin",
        "virulence": "High"
    },
    "Streptococcus_pyogenes": {
        "color": "grayish-white-translucent",
        "hemolysis": "β-hemolytic (strong)",
        "colony_description": "Small, translucent, pinpoint colonies",
        "size": "Tiny (0.5-1 mm)",
        "gram_stain": "Gram-positive cocci in chains",
        "msa": "No growth",
        "key_id": "Bacitracin sensitive, PYR positive",
        "emoji": "🔴",
        "treatment": "Penicillin, Amoxicillin",
        "virulence": "High"
    },
    "Streptococcus_agalactiae": {
        "color": "grayish-white-cream",
        "hemolysis": "β-hemolytic (narrow)",
        "colony_description": "Gray-white, smooth colonies",
        "size": "Medium",
        "gram_stain": "Gram-positive cocci in chains",
        "msa": "No growth",
        "key_id": "CAMP positive, Hippurate positive",
        "emoji": "🟢",
        "treatment": "Penicillin, Ampicillin",
        "virulence": "Moderate"
    }
}

# ============ COMPUTER VISION FEATURE EXTRACTION ============
def extract_features(image):
    """Extract features from image using computer vision"""
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Get image dimensions
    height, width = img_array.shape[:2]
    
    # Resize for analysis
    resized = cv2.resize(img_array, (200, 200))
    
    # 1. COLOR ANALYSIS
    # Get dominant colors using K-means
    pixels = resized.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)
    
    # Convert to HSV for better color analysis
    hsv_img = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    
    # Average HSV values
    avg_hue = np.mean(hsv_img[:, :, 0])
    avg_saturation = np.mean(hsv_img[:, :, 1])
    avg_value = np.mean(hsv_img[:, :, 2])
    
    # 2. TEXTURE ANALYSIS
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    
    # Calculate variance (texture measure)
    variance = np.var(gray)
    
    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    # 3. SHAPE ANALYSIS
    # Threshold for colony detection
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
    else:
        area = 0
        circularity = 0
    
    return {
        'dominant_colors': dominant_colors,
        'avg_hue': avg_hue,
        'avg_saturation': avg_saturation,
        'avg_value': avg_value,
        'variance': variance,
        'edge_density': edge_density,
        'circularity': circularity,
        'area': area
    }

# ============ CLASSIFICATION ENGINE ============
def classify_bacteria(features):
    """Classify bacteria based on extracted features"""
    
    scores = {}
    
    for species, data in morphology_db.items():
        score = 0
        reasons = []
        
        # 1. COLOR ANALYSIS
        avg_hue = features['avg_hue']
        avg_saturation = features['avg_saturation']
        avg_value = features['avg_value']
        
        # Check for golden-yellow (S. aureus)
        if data['color'] == 'golden-yellow':
            # Golden-yellow: Hue around 30-60 (yellow), high saturation
            if 20 < avg_hue < 60 and avg_saturation > 100:
                score += 30
                reasons.append("Golden-yellow color detected")
            if avg_value > 150:
                score += 10
                reasons.append("Bright/opaque appearance")
                
        # Check for white/cream (S. saprophyticus)
        elif data['color'] == 'white-cream':
            if avg_saturation < 50 and avg_value > 150:
                score += 25
                reasons.append("White/cream color detected")
                
        # Check for gray/translucent (Streptococcus)
        elif 'gray' in data['color']:
            if avg_saturation < 80 and avg_value < 150:
                score += 25
                reasons.append("Gray/translucent appearance detected")
                
        # 2. SIZE ANALYSIS
        area = features['area']
        if data['size'] == 'Tiny (0.5-1 mm)' and area < 5000:
            score += 15
            reasons.append("Tiny colony size")
        elif data['size'] == 'Small' and area < 10000:
            score += 10
            reasons.append("Small colony size")
        elif data['size'] == 'Medium' and 10000 < area < 30000:
            score += 10
            reasons.append("Medium colony size")
        elif data['size'] == '2-4 mm' and 10000 < area < 30000:
            score += 10
            reasons.append("Medium-large colony size")
            
        # 3. TEXTURE ANALYSIS
        variance = features['variance']
        circularity = features['circularity']
        
        # Smooth vs rough
        if data['colony_description'].find('smooth') != -1:
            if variance < 2000:
                score += 10
                reasons.append("Smooth texture")
        
        # Mucoid (S. pneumoniae)
        if data['colony_description'].find('mucoid') != -1:
            if circularity > 0.7:
                score += 15
                reasons.append("Mucoid/round appearance")
        
        # Pinpoint (S. pyogenes)
        if data['colony_description'].find('pinpoint') != -1:
            if area < 3000:
                score += 15
                reasons.append("Pinpoint colony")
        
        # 4. HEMOLYSIS INDICATORS
        edge_density = features['edge_density']
        
        if data['hemolysis'] == 'β-hemolytic':
            if edge_density > 0.05:
                score += 10
                reasons.append("Beta hemolysis pattern")
        
        if data['hemolysis'] == 'α-hemolytic':
            if edge_density > 0.03:
                score += 10
                reasons.append("Alpha hemolysis pattern")
        
        # 5. MSA GROWTH (inferred from color/colony characteristics)
        if data['msa'] == 'No growth' and avg_value < 150:
            score += 5
            reasons.append("No MSA growth characteristics")
            
        # Store score and reasons
        scores[species] = {
            'score': score,
            'reasons': reasons[:3]  # Keep top 3 reasons
        }
    
    return scores

# ============ PREDICTION FUNCTION ============
def predict_image(image):
    """Main prediction function"""
    
    # Extract features
    features = extract_features(image)
    
    # Classify
    scores = classify_bacteria(features)
    
    # Get best match
    best_species = max(scores, key=lambda x: scores[x]['score'])
    best_score = scores[best_species]['score']
    
    # Calculate confidence (normalized score)
    max_possible_score = 80
    confidence = min((best_score / max_possible_score) * 100, 95)
    
    # Get all scores for display
    all_scores = {species: scores[species]['score'] for species in scores}
    
    # Get reasons
    reasons = scores[best_species]['reasons']
    
    return best_species, confidence, all_scores, reasons

def get_bacteria_info(bacteria_name):
    """Get detailed information about bacteria"""
    data = morphology_db.get(bacteria_name, {})
    return {
        "gram_stain": data.get("gram_stain", "N/A"),
        "hemolysis": data.get("hemolysis", "N/A"),
        "key_id": data.get("key_id", "N/A"),
        "color": data.get("color", "N/A").replace('-', ' '),
        "size": data.get("size", "N/A"),
        "description": data.get("colony_description", "N/A"),
        "virulence": data.get("virulence", "N/A"),
        "treatment": data.get("treatment", "N/A"),
        "emoji": data.get("emoji", "🧬"),
        "msa": data.get("msa", "N/A")
    }

# ============ PARTICLES ============
def add_particles():
    colors = ['#a5d6a7', '#66bb6a', '#ffe082', '#ffd54f', '#8bc34a', '#f9a825']
    particles = []
    for i in range(15):
        size = random.randint(5, 15)
        left = random.randint(0, 100)
        duration = random.randint(15, 25)
        delay = random.randint(0, 15)
        color = random.choice(colors)
        particles.append(f"""
        <div style="
            position: fixed;
            border-radius: 50%;
            pointer-events: none;
            width: {size}px;
            height: {size}px;
            left: {left}%;
            background: {color};
            animation: floatUp {duration}s ease-in-out infinite;
            animation-delay: {delay}s;
            opacity: 0.08;
            z-index: 0;
        "></div>
        """)
    st.markdown(f"""
    <style>
    @keyframes floatUp {{
        0% {{ transform: translateY(100vh) rotate(0deg); opacity: 0; }}
        10% {{ opacity: 0.08; }}
        90% {{ opacity: 0.08; }}
        100% {{ transform: translateY(-10vh) rotate(720deg); opacity: 0; }}
    }}
    </style>
    {''.join(particles)}
    """, unsafe_allow_html=True)

add_particles()

# ============ MAIN ============
def main():
    st.markdown("""
        <div style="text-align: center; padding: 0.5rem 0;">
            <div style="font-size: 4rem;">🧫</div>
            <h1 class="main-title">
                Bacteria <span class="green-yellow-text">Colony</span> Classifier
            </h1>
            <p style="font-size: 1.2rem; color: #333; font-weight: 700;">🔬 Upload an image to identify bacteria species</p>
        </div>
    """, unsafe_allow_html=True)

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
                with st.spinner("🔍 Analyzing colony morphology..."):
                    time.sleep(0.5)
                    predicted, confidence, all_scores, reasons = predict_image(image)
                    
                    display_name = display_names.get(predicted, predicted.replace('_', ' '))
                    info = get_bacteria_info(predicted)
                    
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
                    
                    # Display reasons
                    reason_text = ""
                    if reasons:
                        reason_text = "🔍 " + " • ".join(reasons)
                    
                    st.markdown(f"""
                    <div class="prediction-box">
                        <div style="font-size: 3.5rem;">{info.get('emoji', '🧬')}</div>
                        <p style="color: #000000; font-weight: 700; margin: 0;">Identified Species</p>
                        <h2>{display_name}</h2>
                        <div class="{conf_color}">{confidence:.1f}%</div>
                        <div>
                            <span class="badge {badge}">{conf_emoji} {conf_level} Confidence</span>
                        </div>
                        <div style="margin-top: 1rem; padding: 0.5rem; background: #f0f2f6; border-radius: 8px;">
                            <p style="color: #000000; font-weight: 600; margin: 0; font-size: 0.9rem;">
                                {reason_text}
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.progress(int(confidence))
                    
                    # Display info
                    with st.expander("📊 Morphology & Clinical Characteristics", expanded=True):
                        tab1, tab2, tab3 = st.tabs(["🔬 Morphology", "🦠 Clinical", "💊 Treatment"])
                        
                        with tab1:
                            st.markdown(f"""
                            <div class="info-card">
                                <h4>🔬 Morphological Features</h4>
                                <p><strong>Gram Stain:</strong> {info['gram_stain']}</p>
                                <p><strong>Hemolysis:</strong> {info['hemolysis']}</p>
                                <p><strong>Colony Color:</strong> {info['color']}</p>
                                <p><strong>Colony Size:</strong> {info['size']}</p>
                                <p><strong>Colony Description:</strong> {info['description']}</p>
                                <p><strong>MSA Growth:</strong> {info['msa']}</p>
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
                    
                    # Display all scores
                    st.markdown("### 📊 Identification Scores")
                    
                    # Sort scores
                    sorted_scores = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
                    
                    # Create color mapping
                    colors = ['#2e7d32' if i==0 else '#bdbdbd' for i in range(len(sorted_scores))]
                    
                    # Use display names
                    display_scores = {display_names.get(k, k.replace('_', ' ')): v for k, v in sorted_scores}
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(display_scores.keys()),
                            y=list(display_scores.values()),
                            marker_color=colors,
                            text=[f"{v:.0f}%" for v in display_scores.values()],
                            textposition='outside',
                            textfont=dict(color='#000000', size=12, weight='bold')
                        )
                    ])
                    
                    fig.update_layout(
                        height=400,
                        xaxis_tickangle=-45,
                        showlegend=False,
                        yaxis_range=[0, max(list(display_scores.values())) * 1.2 + 10],
                        plot_bgcolor='rgba(255,255,255,0.5)',
                        paper_bgcolor='rgba(255,255,255,0)',
                        font=dict(color='#000000', weight='bold')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 3rem;">🧫</div>
            <h3 style="font-weight: 900; color: #000000;">Bacteria AI</h3>
            <p style="color: #000000; font-weight: 700; font-size: 0.9rem;">🔬 Computer Vision Analysis</p>
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
            <p>📊 View detailed analysis</p>
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
