"""
Base CNN model for image classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import sys
sys.path.append('..')
from config.model_config import MODEL_CONFIG, CONSERVATION_CLASSES


class SpeciesClassifier(nn.Module):
    """CNN model for species conservation status classification."""
    
    def __init__(self, num_classes=None, pretrained=True):
        super(SpeciesClassifier, self).__init__()
        
        if num_classes is None:
            num_classes = len(CONSERVATION_CLASSES)
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-30]:
            param.requires_grad = False
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_rate']),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class CNNFeatureExtractor(nn.Module):
    """Feature extractor for multi-task learning."""
    
    def __init__(self, pretrained=True):
        super(CNNFeatureExtractor, self).__init__()
        
        # Load pretrained ResNet50
        backbone = models.resnet50(pretrained=pretrained)
        
        # Remove final FC layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Freeze early layers
        for param in list(self.features.parameters())[:-30]:
            param.requires_grad = False
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x


def create_model(num_classes=None, pretrained=True):
    """Create and return a species classifier model."""
    model = SpeciesClassifier(num_classes=num_classes, pretrained=pretrained)
    return model


def load_model(model_path, num_classes=None):
    """Load a saved model from disk."""
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def save_model(model, model_path):
    """Save model to disk."""
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"\nOutput shape: {output.shape}")
