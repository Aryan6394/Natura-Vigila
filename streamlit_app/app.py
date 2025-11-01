"""
Streamlit app for Natura Vigilia: Endangered Species Classifier
"""

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

from utils import load_model_cached, process_image, create_demo_visualizations
from model_config import CONSERVATION_CLASSES, GEOGRAPHIC_REGIONS

# Page config
st.set_page_config(
    page_title="Natura Vigilia - Endangered Species Classifier",
    page_icon="🌍",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🌍 Natura Vigilia</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Endangered Species Image Classifier with Geographic Regions & Taxonomy</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📤 Upload Image")
st.sidebar.write("Upload an image of an animal to classify its conservation status and view detailed information.")

uploaded_file = st.sidebar.file_uploader(
    "Choose an animal image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of an animal for classification"
)

# Information section
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ About")
    st.write("""
    This app uses deep learning to:
    - Classify species by conservation status
    - Identify geographic regions
    - Display biological taxonomy
    - Show evaluation metrics
    """)
    
    st.markdown("---")
    st.subheader("📊 Conservation Categories")
    for category in CONSERVATION_CLASSES:
        category_names = {
            'CR': 'Critically Endangered',
            'EN': 'Endangered',
            'VU': 'Vulnerable',
            'NT': 'Near Threatened',
            'EW': 'Extinct in the Wild',
            'LC': 'Least Concern'
        }
        st.write(f"**{category}**: {category_names.get(category, category)}")

# Main content
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        # Process image
        with st.spinner("Analyzing image..."):
            try:
                result, highlighted_img = process_image(image)
                
                # Display highlighted image
                st.image(highlighted_img, caption="Highlighted Species", use_column_width=True)
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Using demo mode with placeholder data.")
                
                # Demo data
                result = {
                    'status': 'CR',
                    'status_confidence': 0.87,
                    'regions': ['Africa', 'Asia'],
                    'taxonomy': {
                        'kingdom': 'Animalia',
                        'phylum': 'Chordata',
                        'class': 'Mammalia',
                        'order': 'Primates',
                        'family': 'Hominidae',
                        'genus': 'Pongo',
                        'species': 'abelii',
                        'sciName': 'Pongo abelii'
                    },
                    'accuracy': 0.85,
                    'precision': 0.83,
                    'recall': 0.82,
                    'f1': 0.825
                }
                highlighted_img = image
    
    # Conservation Status
    st.markdown("---")
    st.subheader("🔴 Conservation Status")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        status_color = {
            'CR': '🔴',
            'EN': '🟠', 
            'VU': '🟡',
            'NT': '🟢',
            'EW': '⚫',
            'LC': '🔵'
        }
        st.markdown(f"### {status_color.get(result['status'], '⚪')} {result['status']}")
        
        status_names = {
            'CR': 'Critically Endangered',
            'EN': 'Endangered',
            'VU': 'Vulnerable',
            'NT': 'Near Threatened',
            'EW': 'Extinct in the Wild',
            'LC': 'Least Concern'
        }
        st.write(status_names.get(result['status'], result['status']))
    
    with col2:
        st.metric("Confidence", f"{result['status_confidence']*100:.2f}%")
    
    # Geographic Regions
    st.markdown("---")
    st.subheader("🌍 Geographic Regions")
    
    if result['regions']:
        cols = st.columns(len(result['regions']))
        for i, region in enumerate(result['regions']):
            with cols[i]:
                st.info(f"📍 {region}")
    else:
        st.write("No specific regions identified")
    
    # Taxonomy
    st.markdown("---")
    st.subheader("🧬 Biological Taxonomy")
    
    taxonomy_df = pd.DataFrame([result['taxonomy']]).T
    taxonomy_df.columns = ['Value']
    taxonomy_df.index.name = 'Field'
    st.table(taxonomy_df)
    
    # Model Performance Metrics
    st.markdown("---")
    st.subheader("📊 Model Evaluation Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Accuracy",
            value=f"{result['accuracy']*100:.2f}%",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Precision",
            value=f"{result['precision']*100:.2f}%",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Recall",
            value=f"{result['recall']*100:.2f}%",
            delta=None
        )
    
    with col4:
        st.metric(
            label="F1-Score",
            value=f"{result['f1']*100:.2f}%",
            delta=None
        )
    
    # Visualizations
    st.markdown("---")
    st.subheader("📈 Model Evaluation Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Confusion Matrix", 
        "Training History", 
        "Class Distribution",
        "Geographic Distribution"
    ])
    
    with tab1:
        st.write("### Confusion Matrix - Conservation Status")
        try:
            st.image("../visualizations/confusion_matrix.png", use_column_width=True)
        except:
            st.info("Confusion matrix visualization will appear here after model training.")
    
    with tab2:
        st.write("### Training & Validation Loss")
        try:
            st.image("../visualizations/training_history.png", use_column_width=True)
        except:
            st.info("Training history visualization will appear here after model training.")
    
    with tab3:
        st.write("### Class Distribution in Dataset")
        try:
            st.image("../visualizations/class_distribution.png", use_column_width=True)
        except:
            st.info("Class distribution visualization will appear here after evaluation.")
    
    with tab4:
        st.write("### Geographic Distribution Map")
        try:
            st.image("../visualizations/geographic_distribution_map.png", use_column_width=True)
        except:
            st.info("Geographic distribution map will appear here after data processing.")
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create downloadable prediction data
        prediction_data = {
            'Conservation Status': [result['status']],
            'Confidence': [f"{result['status_confidence']*100:.2f}%"],
            'Regions': [', '.join(result['regions'])],
            'Scientific Name': [result['taxonomy'].get('sciName', 'N/A')]
        }
        prediction_df = pd.DataFrame(prediction_data)
        
        csv = prediction_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Prediction Data (CSV)",
            data=csv,
            file_name="species_prediction.csv",
            mime="text/csv"
        )
    
    with col2:
        # Download highlighted image
        import io
        buf = io.BytesIO()
        highlighted_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="🖼️ Download Highlighted Image",
            data=byte_im,
            file_name="highlighted_species.png",
            mime="image/png"
        )

else:
    # Landing page
    st.info("👆 Please upload an animal image using the sidebar to get started.")
    
    # Example information
    st.markdown("---")
    st.subheader("🎯 What This App Does")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🔍 Classification**
        - Identifies species
        - Determines conservation status
        - Calculates confidence scores
        """)
    
    with col2:
        st.markdown("""
        **🌍 Geographic Analysis**
        - Maps population regions
        - Shows distribution patterns
        - Identifies habitats
        """)
    
    with col3:
        st.markdown("""
        **🧬 Taxonomy Information**
        - Full biological classification
        - Scientific nomenclature
        - Species metadata
        """)
    
    st.markdown("---")
    st.subheader("📚 Conservation Status Categories")
    
    status_info = {
        'CR - Critically Endangered': 'Extremely high risk of extinction in the wild',
        'EN - Endangered': 'High risk of extinction in the wild',
        'VU - Vulnerable': 'High risk of endangerment in the wild',
        'NT - Near Threatened': 'Likely to become endangered in the near future',
        'EW - Extinct in the Wild': 'Only survives in captivity',
        'LC - Least Concern': 'Lowest risk, widespread and abundant'
    }
    
    for status, description in status_info.items():
        st.write(f"**{status}**: {description}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888;'>
        <p>Natura Vigilia - Protecting Endangered Species Through AI</p>
        <p>Powered by Deep Learning & Streamlit</p>
    </div>
""", unsafe_allow_html=True)
