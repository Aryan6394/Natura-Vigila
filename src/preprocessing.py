"""
Preprocessing utilities for image data and labels.
"""

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms
import sys
sys.path.append('..')
from config.model_config import MODEL_CONFIG, CONSERVATION_CLASSES, GEOGRAPHIC_REGIONS


def preprocess_image(image_path, input_size=None):
    """Preprocess a single image for model input."""
    if input_size is None:
        input_size = MODEL_CONFIG['input_size']
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add batch dimension


def encode_conservation_label(label):
    """Encode conservation status label to integer."""
    if label in CONSERVATION_CLASSES:
        return CONSERVATION_CLASSES.index(label)
    return 0


def decode_conservation_label(index):
    """Decode integer to conservation status label."""
    if 0 <= index < len(CONSERVATION_CLASSES):
        return CONSERVATION_CLASSES[index]
    return CONSERVATION_CLASSES[0]


def encode_geographic_regions(regions):
    """Encode geographic regions to multi-hot vector."""
    vector = np.zeros(len(GEOGRAPHIC_REGIONS))
    for region in regions:
        if region in GEOGRAPHIC_REGIONS:
            idx = GEOGRAPHIC_REGIONS.index(region)
            vector[idx] = 1
    return vector


def decode_geographic_regions(vector, threshold=0.5):
    """Decode multi-hot vector to list of geographic regions."""
    regions = []
    for idx, value in enumerate(vector):
        if value >= threshold:
            regions.append(GEOGRAPHIC_REGIONS[idx])
    return regions


def augment_image(image):
    """Apply data augmentation to an image."""
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ])
    return augmentation(image)


def normalize_image(image_tensor):
    """Normalize image tensor."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (image_tensor - mean) / std


def denormalize_image(image_tensor):
    """Denormalize image tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return image_tensor * std + mean


def prepare_batch(images, conservation_labels, geographic_labels):
    """Prepare a batch of data for training."""
    images = torch.stack(images)
    conservation_labels = torch.tensor([encode_conservation_label(label) 
                                       for label in conservation_labels])
    geographic_labels = torch.tensor([encode_geographic_regions(regions) 
                                     for regions in geographic_labels])
    
    return images, conservation_labels, geographic_labels


if __name__ == "__main__":
    # Test preprocessing
    print("Testing conservation label encoding...")
    label = "CR"
    encoded = encode_conservation_label(label)
    decoded = decode_conservation_label(encoded)
    print(f"{label} -> {encoded} -> {decoded}")
    
    print("\nTesting geographic encoding...")
    regions = ["Africa", "Asia"]
    encoded = encode_geographic_regions(regions)
    decoded = decode_geographic_regions(encoded)
    print(f"{regions} -> {encoded} -> {decoded}")
