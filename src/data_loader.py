"""
Data loader for the endangered species classifier.
Handles loading and batching of image data from Hugging Face datasets.
"""

from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms
import sys
sys.path.append('..')
from config.model_config import DATASET_CONFIG, MODEL_CONFIG


class SpeciesDataset(Dataset):
    """Custom dataset for species images."""
    
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        
        if self.transform:
            image = self.transform(image)
        
        label = item.get('label', 0)
        
        return image, label


def get_transforms(train=True):
    """Get image transformations for training or validation."""
    if train:
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(MODEL_CONFIG['input_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def load_species_data(dataset_name=None):
    """Load the species dataset from Hugging Face."""
    if dataset_name is None:
        dataset_name = DATASET_CONFIG['name']
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name)
    
    return dataset


def create_dataloaders(dataset, batch_size=None):
    """Create train, validation, and test dataloaders."""
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    
    # Split dataset
    train_size = int(DATASET_CONFIG['train_split'] * len(dataset['train']))
    val_size = int(DATASET_CONFIG['val_split'] * len(dataset['train']))
    
    # Create datasets with transforms
    train_dataset = SpeciesDataset(dataset['train'][:train_size], 
                                   transform=get_transforms(train=True))
    val_dataset = SpeciesDataset(dataset['train'][train_size:train_size+val_size], 
                                 transform=get_transforms(train=False))
    test_dataset = SpeciesDataset(dataset.get('test', dataset['train'][train_size+val_size:]), 
                                  transform=get_transforms(train=False))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test data loading
    dataset = load_species_data()
    train_loader, val_loader, test_loader = create_dataloaders(dataset)
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
