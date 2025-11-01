"""
Configuration file for the Natura Vigilia endangered species classifier model.
"""

# Model Configuration
MODEL_CONFIG = {
    'input_size': (224, 224),
    'num_channels': 3,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.5,
}

# Conservation Status Categories
CONSERVATION_CLASSES = [
    'CR',  # Critically Endangered
    'EN',  # Endangered
    'VU',  # Vulnerable
    'NT',  # Near Threatened
    'EW',  # Extinct in the Wild
    'LC',  # Least Concern
]

# Geographic Regions
GEOGRAPHIC_REGIONS = [
    'Africa',
    'Asia',
    'Europe',
    'North America',
    'South America',
    'Oceania',
    'Arctic/Antarctic',
    'Marine',
]

# Taxonomy Fields
TAXONOMY_FIELDS = [
    'kingdom',
    'phylum',
    'class',
    'order',
    'family',
    'genus',
    'species',
    'sciName',
]

# Data Paths
DATA_PATHS = {
    'raw': 'data/raw/',
    'processed': 'data/processed/',
    'geographic': 'data/geographic/',
    'taxonomy': 'data/taxonomy/taxonomy_table.csv',
}

# Model Paths
MODEL_PATHS = {
    'saved_models': 'models/saved_models/',
    'checkpoints': 'models/checkpoints/',
    'multi_task_model': 'models/saved_models/multi_task_model.h5',
}

# Dataset Configuration
DATASET_CONFIG = {
    'name': 'imageomics/rare-species',
    'train_split': 0.8,
    'val_split': 0.1,
    'test_split': 0.1,
}

# Training Configuration
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'loss_functions': {
        'conservation': 'categorical_crossentropy',
        'geographic': 'binary_crossentropy',
    },
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
}

# Visualization Configuration
VIZ_CONFIG = {
    'figure_size': (12, 8),
    'dpi': 100,
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': 'viridis',
}
