# 🌍 Natura Vigilia: Endangered Species Image Classifier with Geographic Regions & Taxonomy (CNN Model + Streamlit Demo)

A Convolutional Neural Network (CNN) model for classifying animal images into IUCN Red List conservation categories, predicting their geographic regions, and showing full biological taxonomy. Includes a Streamlit app for interactive use and visualization.

## 📋 Project Overview

This project uses deep learning to automatically classify species images into conservation status categories, identify population regions, and display full taxonomy information for each entry. The Streamlit app provides an easy-to-use web interface to explore predictions and evaluation metrics.

## 🎯 Classification & Metadata Output

### Categories Predicted

- **Conservation Status:** Critically Endangered (CR), Endangered (EN), Vulnerable (VU), Near Threatened (NT), Extinct in the Wild (EW), Least Concern (LC)
- **Geographic Population Region(s):** Africa, Asia, Europe, North America, South America, Oceania, Arctic/Antarctic, Marine
- **Biological Taxonomy:**
  - kingdom: Biological kingdom (all entries are Animalia)
  - phylum: Biological phylum
  - class: Biological class
  - order: Biological order
  - family: Biological family
  - genus: Biological genus
  - species: Specific species name
  - sciName: Scientific name (genus + species)

## 📂 Project Structure

```
endangered-species-classifier/
├── README.md
├── requirements.txt
├── config/
│   └── model_config.py
├── data/
│   ├── raw/
│   ├── processed/
│   ├── geographic/
│   └── taxonomy/
│       └── taxonomy_table.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation.ipynb
│   └── 05_streamlit_demo.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── multi_task_model.py
│   ├── geographic_encoder.py
│   ├── taxonomy_lookup.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── saved_models/
│   │   ├── multi_task_model.h5
│   └── checkpoints/
├── streamlit_app/
│   ├── app.py
│   ├── utils.py
│   ├── assets/
│   └── evaluation_cache.pkl
├── visualizations/
│   ├── accuracy_plots.png
│   ├── loss_plots.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_geographic.png
│   ├── class_distribution.png
│   ├── geographic_distribution_map.png
│   ├── species_density_heatmap.png
│   └── training_history.png
└── results/
    ├── classification_report.txt
    ├── geographic_predictions.txt
    ├── model_metrics.json
    ├── predictions_with_taxonomy.csv
    └── conservation_priority_map.csv
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Google Colab account (for training)
- Hugging Face account
- IUCN Red List API access (optional for enhanced data)

### Required Libraries

```bash
pip install torch torchvision datasets huggingface-hub numpy pandas matplotlib seaborn scikit-learn pillow geopandas folium plotly streamlit
```

### Hugging Face Authentication

```bash
huggingface-cli login
```

## 📊 Dataset Information

- **Primary Dataset:** imageomics/rare-species (Hugging Face)
- **Geographic Data:** IUCN Red List spatial data, mapping regions
- **Taxonomy Table:** Full taxonomy CSV for mapping predictions

## 🧠 Model Architecture

Multi-task CNN: predicts conservation status, population region(s), and taxonomy fields (provided as metadata for output, not predicted).

## 🚀 Training the Model

See previous instructions for Colab-based training.
After training, save model as: `models/saved_models/multi_task_model.h5`

## 🖥️ Streamlit App Features

**Main functionality:**  
- Upload an animal image  
- The trained model will:
  - Classify the species and highlight it in the image
  - Display predicted conservation status (CR, EN, VU, NT, EW, LC)
  - Show predicted population regions (Africa, Asia, etc.)
  - Display biological taxonomy (kingdom, phylum, class, order, family, genus, species, sciName)
  - Show prediction accuracy and confidence scores
  - Present evaluation metrics: overall accuracy, precision, recall, F1-score
  - Display interactive graphs: confusion matrices, accuracy/loss plots, class & region distribution charts, geographic maps
- All information is visible directly in the app interface, including:
  - Highlighted species in the uploaded image (draw a bounding box or overlay on image for focus)
  - Downloadable results and reports

**Example Streamlit workflow:**

```python
import streamlit as st
from PIL import Image
import numpy as np
from src.multi_task_model import load_multi_task_model, predict_image_and_highlight
from src.taxonomy_lookup import get_taxonomy_info
import pandas as pd

st.title("Natura Vigilia: Endangered Species Classifier and Visualizer")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an animal image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Predict and highlight species on image
    model = load_multi_task_model('models/saved_models/multi_task_model.h5')
    result, highlighted_img = predict_image_and_highlight(model, image)
    st.image(highlighted_img, caption="Highlighted Species", use_column_width=True)

    # Prediction results (status, regions, taxonomy, confidence, etc.)
    st.write(f"**Conservation Status:** {result['status']} ({result['status_confidence']*100:.2f}%)")
    st.write(f"**Geographic Regions:** {', '.join(result['regions'])}")
    st.write("**Taxonomy:**")
    st.table(result['taxonomy'])

    # Evaluation metrics
    st.markdown("### Model Evaluation Scores")
    st.write(f"Accuracy: {result['accuracy']*100:.2f}%")
    st.write(f"Precision: {result['precision']*100:.2f}%")
    st.write(f"Recall: {result['recall']*100:.2f}%")
    st.write(f"F1-score: {result['f1']*100:.2f}%")

    # Show evaluation graphs
    st.markdown("### Evaluation Graphs")
    st.image("visualizations/accuracy_plots.png", caption="Accuracy Plot")
    st.image("visualizations/loss_plots.png", caption="Loss Plot")
    st.image("visualizations/confusion_matrix.png", caption="Confusion Matrix")
    st.image("visualizations/geographic_distribution_map.png", caption="Species Distribution Map")

    # Optionally allow downloads
    st.markdown("### Download Results")
    st.download_button("Download Prediction Data", "results/predictions_with_taxonomy.csv")
```

**How to run the app:**
```bash
streamlit run streamlit_app/app.py
```

### Features & Visualizations in App
- Species in the image is visually highlighted (bounding box or overlay)
- Prediction labels, scores, and taxonomy displayed alongside the image
- Evaluation metrics and graphs shown interactively
- Download predictions/report for further analysis

---

**Note:**  
Visual highlighting (e.g., bounding box) assumes optional image localization is implemented.  
For the most accurate scores on uploaded images, make sure your test set is diverse and well-evaluated.

