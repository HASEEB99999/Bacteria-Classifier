import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import plotly.graph_objects as go

st.set_page_config(
    page_title="🧫 Bacteria Colony Classifier",
    page_icon="🧫",
    layout="wide"
)

# Load TFLite model
@st.cache_resource
def load_tflite_model():
    try:
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path='bacteria_classifier.tflite')
        interpreter.allocate_tensors()
        
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
            
        return interpreter, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def predict_image(image, interpreter, class_names):
    # Resize and preprocess
    img = image.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output
    predictions = interpreter.get_tensor(output_details[0]['index'])
    
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    all_predictions = {class_names[i]: predictions[0][i] * 100 for i in range(len(class_names))}
    
    return class_names[predicted_class], confidence, all_predictions

def get_bacteria_info(bacteria_name):
    """Return morphology information for each bacteria"""
    info = {
        "Staphylococcus_aureus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "β-hemolytic",
            "key_id": "Catalase+, Coagulase+, Mannitol fermenter",
            "color": "Golden-yellow to cream",
            "size": "2-4 mm",
            "description": "Round, smooth, convex, opaque colonies with golden-yellow pigment"
        },
        "Staphylococcus_saprophyticus": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "Usually γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin resistant",
            "color": "White to cream",
            "size": "1-3 mm",
            "description": "Smooth, convex, opaque colonies; usually non-pigmented"
        },
        "Staphylococcus_epidermidis": {
            "gram_stain": "Gram-positive cocci in clusters",
            "hemolysis": "γ-hemolytic",
            "key_id": "Catalase+, Coagulase−, Novobiocin sensitive",
            "color": "White to grayish-white",
            "size": "Small",
            "description": "Small, smooth, circular, convex, non-pigmented, glossy colonies"
        },
        "Streptococcus_pneumoniae": {
            "gram_stain": "Gram-positive lancet-shaped diplococci",
            "hemolysis": "α-hemolytic",
            "key_id": "Optochin sensitive, bile soluble",
            "color": "Gray, translucent",
            "size": "Small",
            "description": "Small, glistening, mucoid colonies; older colonies have central depression"
        },
        "Streptococcus_pyogenes": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Strong β-hemolytic",
            "key_id": "Bacitracin sensitive, PYR positive",
            "color": "Grayish-white, translucent",
            "size": "Tiny (0.5-1 mm)",
            "description": "Small, translucent, pinpoint colonies with strong beta hemolysis"
        },
        "Streptococcus_agalactiae": {
            "gram_stain": "Gram-positive cocci in chains",
            "hemolysis": "Narrow β-hemolytic",
            "key_id": "CAMP positive, Hippurate positive",
            "color": "Grayish-white to cream",
            "size": "Medium",
            "description": "Medium-sized, smooth, grayish-white to cream, slightly mucoid colonies"
        }
    }
    return info.get(bacteria_name, {})

def main():
    st.markdown("""
        <h1 style='text-align: center; color: #667eea;'>🧫 Bacteria Colony Classifier</h1>
        <p style='text-align: center; color: #666;'>Upload an image of a bacterial colony to identify the species</p>
    """, unsafe_allow_html=True)
    
    # Load model
    interpreter, class_names = load_tflite_model()
    
    if interpreter is None:
        st.warning("⚠️ Please make sure 'bacteria_classifier.tflite' and 'class_names.json' are in the same directory.")
        return
    
    # Upload section
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'tiff', 'bmp']
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_column_width=True)
        
        with col2:
            if st.button("🔬 Classify Bacteria", use_container_width=True):
                with st.spinner("🔍 Analyzing colony morphology..."):
                    # Make prediction
                    predicted, confidence, all_predictions = predict_image(
                        image, interpreter, class_names
                    )
                    
                    # Display prediction
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                                padding: 2rem; border-radius: 15px; text-align: center;'>
                        <h2 style='color: #2d3436;'>{predicted.replace('_', ' ')}</h2>
                        <p style='font-size: 1.5rem; font-weight: 600; color: {"#00b894" if confidence > 60 else "#fdcb6e"};'>
                            {confidence:.1f}% Confidence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(int(confidence))
                    
                    # Show morphology info
                    info = get_bacteria_info(predicted)
                    if info:
                        with st.expander("📊 Morphology Characteristics", expanded=True):
                            st.markdown(f"""
                                - **Gram Stain**: {info['gram_stain']}
                                - **Hemolysis**: {info['hemolysis']}
                                - **Colony Color**: {info['color']}
                                - **Colony Size**: {info['size']}
                                - **Description**: {info['description']}
                                - **Key ID**: {info['key_id']}
                            """)
                    
                    # Confidence chart
                    st.markdown("### 📈 Confidence Scores for All Species")
                    sorted_predictions = dict(sorted(all_predictions.items(), key=lambda x: x[1], reverse=True))
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(sorted_predictions.keys()),
                            y=list(sorted_predictions.values()),
                            marker_color=['#00b894' if x == predicted else '#dfe6e9' 
                                         for x in sorted_predictions.keys()]
                        )
                    ])
                    fig.update_layout(
                        xaxis_title="Bacteria Species",
                        yaxis_title="Confidence (%)",
                        yaxis_range=[0, 100],
                        height=400,
                        xaxis_tickangle=-45
                    )
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
