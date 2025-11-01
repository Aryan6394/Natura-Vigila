"""
Multi-task model for conservation status, geographic region, and taxonomy prediction.
"""

import torch
import torch.nn as nn
from model import CNNFeatureExtractor
import sys
sys.path.append('..')
from config.model_config import CONSERVATION_CLASSES, GEOGRAPHIC_REGIONS, MODEL_CONFIG


class MultiTaskModel(nn.Module):
    """Multi-task CNN for conservation status and geographic region prediction."""
    
    def __init__(self, pretrained=True):
        super(MultiTaskModel, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(pretrained=pretrained)
        
        # Feature dimension
        feature_dim = 2048  # ResNet50 output
        
        # Conservation status classification head
        self.conservation_head = nn.Sequential(
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(512, len(CONSERVATION_CLASSES))
        )
        
        # Geographic region prediction head (multi-label)
        self.geographic_head = nn.Sequential(
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(256, len(GEOGRAPHIC_REGIONS)),
            nn.Sigmoid()  # Multi-label classification
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions from each head
        conservation_output = self.conservation_head(features)
        geographic_output = self.geographic_head(features)
        
        return conservation_output, geographic_output


def load_multi_task_model(model_path):
    """Load a saved multi-task model."""
    model = MultiTaskModel(pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_multi_task_model(model, model_path):
    """Save multi-task model to disk."""
    torch.save(model.state_dict(), model_path)
    print(f"Multi-task model saved to {model_path}")


def predict_image_and_highlight(model, image):
    """
    Predict conservation status and regions for an image.
    Also highlights the species in the image (placeholder for actual implementation).
    
    Returns:
        result: Dictionary with predictions
        highlighted_img: Image with highlighted species
    """
    from preprocessing import preprocess_image, decode_conservation_label, decode_geographic_regions
    from taxonomy_lookup import get_taxonomy_info
    import numpy as np
    from PIL import Image, ImageDraw
    
    # Preprocess image
    if isinstance(image, str):
        img_tensor = preprocess_image(image)
        pil_image = Image.open(image)
    else:
        # Assume PIL Image
        pil_image = image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(pil_image).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        conservation_output, geographic_output = model(img_tensor)
    
    # Process conservation status
    conservation_probs = torch.softmax(conservation_output, dim=1)[0]
    conservation_idx = torch.argmax(conservation_probs).item()
    conservation_status = decode_conservation_label(conservation_idx)
    conservation_confidence = conservation_probs[conservation_idx].item()
    
    # Process geographic regions
    geographic_probs = geographic_output[0].numpy()
    regions = decode_geographic_regions(geographic_probs, threshold=0.5)
    
    # Get taxonomy (placeholder - would need actual species identification)
    taxonomy = get_taxonomy_info(conservation_status)
    
    # Create highlighted image (simple bounding box as placeholder)
    highlighted_img = pil_image.copy()
    draw = ImageDraw.Draw(highlighted_img)
    width, height = highlighted_img.size
    # Draw a bounding box around center region (placeholder)
    box = [width*0.2, height*0.2, width*0.8, height*0.8]
    draw.rectangle(box, outline="red", width=5)
    
    # Prepare result dictionary
    result = {
        'status': conservation_status,
        'status_confidence': conservation_confidence,
        'regions': regions,
        'taxonomy': taxonomy,
        'accuracy': 0.85,  # Placeholder - would come from evaluation
        'precision': 0.83,
        'recall': 0.82,
        'f1': 0.825,
    }
    
    return result, highlighted_img


if __name__ == "__main__":
    # Test multi-task model
    model = MultiTaskModel()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    conservation_out, geographic_out = model(dummy_input)
    print(f"\nConservation output shape: {conservation_out.shape}")
    print(f"Geographic output shape: {geographic_out.shape}")
