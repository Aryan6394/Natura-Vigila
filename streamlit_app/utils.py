"""
Utility functions for the Streamlit app.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

import streamlit as st
from PIL import Image, ImageDraw
import torch
import numpy as np


@st.cache_resource
def load_model_cached():
    """Load the model and cache it."""
    try:
        from multi_task_model import load_multi_task_model
        from model_config import MODEL_PATHS
        
        model_path = os.path.join('..', MODEL_PATHS['multi_task_model'])
        model = load_multi_task_model(model_path)
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load model: {str(e)}")
        return None


def process_image(image):
    """
    Process an uploaded image and return predictions.
    
    Args:
        image: PIL Image
        
    Returns:
        result: Dictionary with predictions
        highlighted_img: Image with highlighted species
    """
    try:
        model = load_model_cached()
        
        if model is None:
            # Return demo data if model not available
            return get_demo_predictions(image)
        
        from multi_task_model import predict_image_and_highlight
        result, highlighted_img = predict_image_and_highlight(model, image)
        
        return result, highlighted_img
        
    except Exception as e:
        st.error(f"Error in process_image: {str(e)}")
        return get_demo_predictions(image)


def get_demo_predictions(image):
    """Return demo predictions for testing."""
    # Create a simple highlighted version
    highlighted_img = image.copy()
    draw = ImageDraw.Draw(highlighted_img)
    width, height = highlighted_img.size
    
    # Draw bounding box
    box = [width*0.2, height*0.2, width*0.8, height*0.8]
    draw.rectangle(box, outline="red", width=5)
    
    # Demo result
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
    
    return result, highlighted_img


def create_demo_visualizations():
    """Create demo visualizations if real ones don't exist."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create dummy confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    dummy_cm = np.random.randint(0, 50, size=(6, 6))
    sns.heatmap(dummy_cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix (Demo)')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    
    return fig


def format_confidence(confidence):
    """Format confidence score as percentage."""
    return f"{confidence * 100:.2f}%"


def get_status_color(status):
    """Get color for conservation status."""
    colors = {
        'CR': '#D32F2F',  # Red
        'EN': '#F57C00',  # Orange
        'VU': '#FBC02D',  # Yellow
        'NT': '#388E3C',  # Green
        'EW': '#424242',  # Grey
        'LC': '#1976D2'   # Blue
    }
    return colors.get(status, '#9E9E9E')


def get_status_emoji(status):
    """Get emoji for conservation status."""
    emojis = {
        'CR': '🔴',
        'EN': '🟠',
        'VU': '🟡',
        'NT': '🟢',
        'EW': '⚫',
        'LC': '🔵'
    }
    return emojis.get(status, '⚪')


def get_region_emoji(region):
    """Get emoji for geographic region."""
    emojis = {
        'Africa': '🌍',
        'Asia': '🌏',
        'Europe': '🌍',
        'North America': '🌎',
        'South America': '🌎',
        'Oceania': '🌏',
        'Arctic/Antarctic': '❄️',
        'Marine': '🌊'
    }
    return emojis.get(region, '📍')


def create_taxonomy_table(taxonomy):
    """Create a formatted taxonomy table."""
    import pandas as pd
    
    df = pd.DataFrame([taxonomy]).T
    df.columns = ['Value']
    df.index.name = 'Taxonomic Rank'
    
    return df


def highlight_species_in_image(image, bbox=None):
    """
    Draw a bounding box on the image to highlight the species.
    
    Args:
        image: PIL Image
        bbox: Bounding box coordinates [x1, y1, x2, y2], or None for center box
        
    Returns:
        PIL Image with bounding box
    """
    highlighted = image.copy()
    draw = ImageDraw.Draw(highlighted)
    
    if bbox is None:
        # Default to center region
        width, height = image.size
        bbox = [width*0.2, height*0.2, width*0.8, height*0.8]
    
    # Draw rectangle
    draw.rectangle(bbox, outline="red", width=5)
    
    # Add label
    draw.text((bbox[0], bbox[1]-30), "Detected Species", fill="red")
    
    return highlighted


if __name__ == "__main__":
    print("Streamlit utility functions loaded successfully.")
