"""
Training script for the multi-task endangered species classifier.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('..')
from config.model_config import MODEL_CONFIG, TRAINING_CONFIG, MODEL_PATHS
from multi_task_model import MultiTaskModel, save_multi_task_model
from data_loader import create_dataloaders, load_species_data


class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning."""
    
    def __init__(self, conservation_weight=1.0, geographic_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.conservation_weight = conservation_weight
        self.geographic_weight = geographic_weight
        
        self.conservation_loss = nn.CrossEntropyLoss()
        self.geographic_loss = nn.BCELoss()
    
    def forward(self, conservation_pred, conservation_target, 
                geographic_pred, geographic_target):
        
        loss_conservation = self.conservation_loss(conservation_pred, conservation_target)
        loss_geographic = self.geographic_loss(geographic_pred, geographic_target)
        
        total_loss = (self.conservation_weight * loss_conservation + 
                     self.geographic_weight * loss_geographic)
        
        return total_loss, loss_conservation, loss_geographic


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    conservation_loss_sum = 0.0
    geographic_loss_sum = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for images, conservation_labels, geographic_labels in pbar:
        images = images.to(device)
        conservation_labels = conservation_labels.to(device)
        geographic_labels = geographic_labels.to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        conservation_pred, geographic_pred = model(images)
        
        # Calculate loss
        loss, loss_c, loss_g = criterion(
            conservation_pred, conservation_labels,
            geographic_pred, geographic_labels
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        conservation_loss_sum += loss_c.item()
        geographic_loss_sum += loss_g.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    avg_conservation_loss = conservation_loss_sum / len(train_loader)
    avg_geographic_loss = geographic_loss_sum / len(train_loader)
    
    return avg_loss, avg_conservation_loss, avg_geographic_loss


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    conservation_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, conservation_labels, geographic_labels in val_loader:
            images = images.to(device)
            conservation_labels = conservation_labels.to(device)
            geographic_labels = geographic_labels.to(device).float()
            
            # Forward pass
            conservation_pred, geographic_pred = model(images)
            
            # Calculate loss
            loss, _, _ = criterion(
                conservation_pred, conservation_labels,
                geographic_pred, geographic_labels
            )
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(conservation_pred, 1)
            conservation_correct += (predicted == conservation_labels).sum().item()
            total_samples += conservation_labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = conservation_correct / total_samples
    
    return avg_loss, accuracy


def train_model(num_epochs=None, batch_size=None, learning_rate=None):
    """Main training function."""
    
    # Set parameters
    if num_epochs is None:
        num_epochs = MODEL_CONFIG['epochs']
    if batch_size is None:
        batch_size = MODEL_CONFIG['batch_size']
    if learning_rate is None:
        learning_rate = MODEL_CONFIG['learning_rate']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading dataset...")
    dataset = load_species_data()
    train_loader, val_loader, test_loader = create_dataloaders(dataset, batch_size)
    
    # Create model
    print("Creating model...")
    model = MultiTaskModel(pretrained=True)
    model = model.to(device)
    
    # Setup training
    criterion = MultiTaskLoss(conservation_weight=1.0, geographic_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=TRAINING_CONFIG['reduce_lr_patience']
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    print(f"\nStarting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_c_loss, train_g_loss = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_multi_task_model(model, MODEL_PATHS['multi_task_model'])
        else:
            patience_counter += 1
        
        if patience_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\nTraining completed!")
    return model, history


if __name__ == "__main__":
    model, history = train_model()
