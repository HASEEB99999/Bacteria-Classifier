import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import json
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

# ============ PAGE CONFIGURATION ============
st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ CUSTOM CSS ============
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Prediction box */
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        text-align: center;
    }
    
    .prediction-box h2 {
        font-size: 2.5rem;
        margin: 0.5rem 0;
    }
    
    .prediction-box .confidence {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .high-confidence {
        color: #00b894;
    }
    
    .medium-confidence {
        color: #fdcb6e;
    }
    
    .low-confidence {
        color: #e17055;
    }
    
    /* Status badges */
    .badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.25rem;
    }
    
    .badge-success {
        background-color: #00b894;
        color: white;
    }
    
    .badge-warning {
        background-color: #fdcb6e;
        color: #2d3436;
    }
    
    .badge-danger {
        background-color: #e17055;
        color: white;
    }
    
    .badge-info {
        background-color: #74b9ff;
        color: white;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    .info-card h4 {
        color: #2d3436;
        margin-bottom: 0.5rem;
    }
    
    .info-card p {
        color: #636e72;
        margin: 0.25rem 0;
    }
    
    /* Upload area */
    .upload-area {
        border: 3px dashed #dfe6e9;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background: #fafafa;
        transition: all 0.3s;
    }
    
    .upload-area:hover {
        border-color: #667eea;
        background: #f0f2ff;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar styling */
    .sidebar-content {
        padding: 1rem 0;
    }
    
    .sidebar-content h3 {
        color: #2d3436;
        margin-bottom: 1rem;
    }
    
    .sidebar-content .metric {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin: 0.5rem 0;
    }
    
    .sidebar-content .metric .value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .sidebar-content .metric .label {
        color: #636e72;
        font-size: 0.9rem;
    }
    
    /* History table */
    .history-table {
        max-height: 300px;
        overflow-y: auto;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #b2bec3;
        padding: 2rem 0;
        margin-top: 2rem;
        border-top: 1px solid #dfe6e9;
    }
    </style>
""", unsafe_allow_html=True)

# ============ SESSION STATE ============
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'class_names' not in st.session_state:
    st.session_state.class_names = []

# ============ LOAD MODEL ============
@st.cache_resource
def load_bacteria_model():
    try:
        model = load_model('bacteria_classifier.keras')
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
        st.session_state.model_loaded = True
        st.session_state.class_names = class_names
        return model, class_names
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("Please make sure 'bacteria_classifier.keras' and 'class_names.json' are in the same directory.")
        return None, None

# ============ BACTERIA INFORMATION ============
def get_bacteria_info(bacteria_name):
    """Return detailed morphology information for each bacteria"""
    info = {
        "Staphylococcus_aureus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "β-hemolytic",
            "key_id": "Catalase+, Coagulase+, Mannitol fermenter",
            "color": "Golden-yellow to cream",
            "size": "2-4 mm",
            "description": "Round, smooth, convex, opaque colonies with golden-yellow pigment",
            "virulence": "High",
            "common_infections": ["Skin infections", "Pneumonia", "Sepsis", "Endocarditis"],
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
            "common_infections": ["Urinary tract infections", "Cystitis"],
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
            "common_infections": ["Device-associated infections", "Catheter infections"],
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
            "common_infections": ["Pneumonia", "Meningitis", "Otitis media"],
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
            "common_infections": ["Strep throat", "Scarlet fever", "Impetigo"],
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
            "common_infections": ["Neonatal infections", "UTI", "Sepsis in elderly"],
            "treatment": "Penicillin, Ampicillin"
        }
    }
    return info.get(bacteria_name, {})

# ============ PREDICTION FUNCTION ============
def predict_image(image, model, class_names):
    IMG_SIZE = (224, 224)
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    all_predictions = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
    
    return class_names[predicted_class], confidence, all_predictions

# ============ IMAGE PROCESSING FUNCTIONS ============
def get_image_download_link(img, filename="prediction_result.png"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{img_str}" download="{filename}">Download Result</a>'
    return href

# ============ MAIN APP ============
def main():
    # Header
    st.markdown('<h1 class="main-title">🧫 Bacteria Colony Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Upload an image of a bacterial colony to identify the species based on morphology</p>', unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_bacteria_model()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        st.image("https://img.icons8.com/fluency/96/null/bacteria.png", width=80)
        
        st.markdown("### 📋 Instructions")
        st.markdown("""
        1. 📤 Upload a clear image of a bacterial colony
        2. 🔬 Click "Classify Bacteria"
        3. 📊 View detailed analysis and morphology
        4. 📈 Check confidence scores for all species
        """)
        
        st.divider()
        
        if st.session_state.model_loaded:
            st.markdown("### 📊 Model Performance")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric">
                    <div class="value">82%</div>
                    <div class="label">Accuracy</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown("""
                <div class="metric">
                    <div class="value">6</div>
                    <div class="label">Species</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("### 🔬 Available Species")
            for name in class_names:
                display_name = name.replace('_', ' ')
                st.markdown(f"- {display_name}")
        
        st.divider()
        
        # History
        if st.session_state.prediction_history:
            st.markdown("### 📜 Prediction History")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df = history_df.sort_values('timestamp', ascending=False).head(5)
            
            for _, row in history_df.iterrows():
                st.markdown(f"""
                <div class="info-card" style="border-left-color: {'#00b894' if row['confidence'] > 60 else '#fdcb6e'};">
                    <p style="margin:0;"><strong>{row['predicted'].replace('_', ' ')}</strong></p>
                    <p style="margin:0; font-size:0.85rem; color:#636e72;">{row['confidence']:.1f}% confidence</p>
                    <p style="margin:0; font-size:0.75rem; color:#b2bec3;">{row['timestamp'].strftime('%H:%M:%S')}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    if model is None:
        st.warning("⚠️ Please upload the model files to continue.")
        return
    
    # Upload section
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'webp'],
        help="Upload a clear image of a bacterial colony plate"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="info-card">
                <p><strong>📋 Image Info</strong></p>
                <p>Size: {image.size[0]} x {image.size[1]} pixels</p>
                <p>Format: {image.format}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🔬 Classify Bacteria", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    # Make prediction
                    predicted, confidence, all_predictions = predict_image(
                        image, model, class_names
                    )
                    
                    # Determine confidence level
                    if confidence > 70:
                        confidence_level = "High"
                        confidence_color = "high-confidence"
                        emoji = "✅"
                    elif confidence > 50:
                        confidence_level = "Medium"
                        confidence_color = "medium-confidence"
                        emoji = "⚠️"
                    else:
                        confidence_level = "Low"
                        confidence_color = "low-confidence"
                        emoji = "❌"
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-box fade-in">
                        <p style="color:#636e72; margin:0;">Predicted Species</p>
                        <h2 style="color:#2d3436;">{predicted.replace('_', ' ')}</h2>
                        <p class="confidence {confidence_color}">
                            {emoji} {confidence:.1f}% Confidence ({confidence_level})
                        </p>
                        <div>
                            <span class="badge badge-info">{confidence_level} Confidence</span>
                            <span class="badge {'badge-success' if confidence > 50 else 'badge-danger'}">
                                {'' if confidence > 50 else '⚠️ Needs Review'}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Save to history
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now(),
                        'predicted': predicted,
                        'confidence': confidence,
                        'confidence_level': confidence_level
                    })
                    
                    # Display confidence bar
                    st.progress(int(confidence))
                    
                    # Show morphology info
                    info = get_bacteria_info(predicted)
                    if info:
                        with st.expander("📊 Morphology Characteristics", expanded=True):
                            col3, col4 = st.columns(2)
                            with col3:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>🔬 Microscopic Features</h4>
                                    <p><strong>Gram Stain:</strong> {info['gram_stain']}</p>
                                    <p><strong>Hemolysis:</strong> {info['hemolysis']}</p>
                                    <p><strong>Colony Color:</strong> {info['color']}</p>
                                    <p><strong>Colony Size:</strong> {info['size']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            with col4:
                                st.markdown(f"""
                                <div class="info-card">
                                    <h4>🦠 Clinical Information</h4>
                                    <p><strong>Virulence:</strong> {info['virulence']}</p>
                                    <p><strong>Common Infections:</strong> {', '.join(info['common_infections'])}</p>
                                    <p><strong>Treatment:</strong> {info['treatment']}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div class="info-card" style="border-left-color: #00b894;">
                                <h4>📝 Description</h4>
                                <p>{info['description']}</p>
                                <p><strong>Key ID:</strong> {info['key_id']}</p>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Confidence chart for all classes
            if 'all_predictions' in locals():
                st.markdown("### 📈 Confidence Scores for All Species")
                
                # Sort predictions
                sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                
                # Create color mapping
                colors = []
                for name in sorted_predictions.keys():
                    if name == predicted:
                        colors.append('#00b894')  # Green for predicted
                    elif sorted_predictions[name] > 30:
                        colors.append('#74b9ff')  # Blue for moderate
                    else:
                        colors.append('#dfe6e9')  # Gray for low
                
                # Create bar chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(sorted_predictions.keys()),
                        y=list(sorted_predictions.values()),
                        marker_color=colors,
                        text=[f"{v:.1f}%" for v in sorted_predictions.values()],
                        textposition='outside',
                        hovertemplate='<b>%{x}</b><br>Confidence: %{y:.1f}%<extra></extra>'
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
                    paper_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=20, r=20, t=20, b=20),
                    hovermode='closest'
                )
                
                fig.update_traces(
                    marker_line_color='rgb(0,0,0)',
                    marker_line_width=1,
                    opacity=0.8
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download button for results
                st.download_button(
                    label="📥 Download Results",
                    data=json.dumps({
                        'prediction': predicted,
                        'confidence': confidence,
                        'all_predictions': all_predictions,
                        'timestamp': datetime.now().isoformat()
                    }, indent=2),
                    file_name=f"prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>🧫 Bacteria Colony Classifier | Built with Streamlit & TensorFlow</p>
        <p style="font-size:0.8rem;">For educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
