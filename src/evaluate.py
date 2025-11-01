"""
Evaluation script for the endangered species classifier.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
sys.path.append('..')
from config.model_config import CONSERVATION_CLASSES, GEOGRAPHIC_REGIONS, MODEL_PATHS, VIZ_CONFIG
from multi_task_model import load_multi_task_model
from data_loader import create_dataloaders, load_species_data
from preprocessing import decode_conservation_label, decode_geographic_regions


def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    
    all_conservation_preds = []
    all_conservation_labels = []
    all_geographic_preds = []
    all_geographic_labels = []
    
    with torch.no_grad():
        for images, conservation_labels, geographic_labels in test_loader:
            images = images.to(device)
            
            # Get predictions
            conservation_pred, geographic_pred = model(images)
            
            # Conservation predictions
            _, predicted = torch.max(conservation_pred, 1)
            all_conservation_preds.extend(predicted.cpu().numpy())
            all_conservation_labels.extend(conservation_labels.numpy())
            
            # Geographic predictions
            all_geographic_preds.extend(geographic_pred.cpu().numpy())
            all_geographic_labels.extend(geographic_labels.numpy())
    
    return {
        'conservation_preds': np.array(all_conservation_preds),
        'conservation_labels': np.array(all_conservation_labels),
        'geographic_preds': np.array(all_geographic_preds),
        'geographic_labels': np.array(all_geographic_labels)
    }


def calculate_metrics(predictions):
    """Calculate evaluation metrics."""
    conservation_preds = predictions['conservation_preds']
    conservation_labels = predictions['conservation_labels']
    
    # Overall metrics
    accuracy = accuracy_score(conservation_labels, conservation_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        conservation_labels, conservation_preds, average='weighted'
    )
    
    # Per-class metrics
    report = classification_report(
        conservation_labels, conservation_preds,
        target_names=CONSERVATION_CLASSES,
        output_dict=True
    )
    
    metrics = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class_metrics': report
    }
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=VIZ_CONFIG['figure_size'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix - Conservation Status')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'])
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'])
        print(f"Training history saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_distribution(labels, save_path=None):
    """Plot class distribution."""
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [CONSERVATION_CLASSES[i] for i in unique]
    
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, counts, color='steelblue')
    plt.xlabel('Conservation Status')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution in Test Set')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=VIZ_CONFIG['dpi'])
        print(f"Class distribution saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_results(metrics, predictions, output_dir='results/'):
    """Save evaluation results."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = os.path.join(output_dir, 'model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}")
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Conservation Status Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall: {metrics['recall']:.4f}\n")
        f.write(f"F1-Score: {metrics['f1']:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 50 + "\n")
        for class_name in CONSERVATION_CLASSES:
            if class_name in metrics['per_class_metrics']:
                class_metrics = metrics['per_class_metrics'][class_name]
                f.write(f"\n{class_name}:\n")
                f.write(f"  Precision: {class_metrics['precision']:.4f}\n")
                f.write(f"  Recall: {class_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score: {class_metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {class_metrics['support']}\n")
    print(f"Classification report saved to {report_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': predictions['conservation_labels'],
        'predicted_label': predictions['conservation_preds'],
        'true_class': [CONSERVATION_CLASSES[i] for i in predictions['conservation_labels']],
        'predicted_class': [CONSERVATION_CLASSES[i] for i in predictions['conservation_preds']]
    })
    predictions_path = os.path.join(output_dir, 'predictions_with_taxonomy.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")


def main():
    """Main evaluation function."""
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_multi_task_model(MODEL_PATHS['multi_task_model'])
    model = model.to(device)
    
    print("Loading test data...")
    dataset = load_species_data()
    _, _, test_loader = create_dataloaders(dataset)
    
    print("Evaluating model...")
    predictions = evaluate_model(model, test_loader, device)
    
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions)
    
    print("\nResults:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        predictions['conservation_labels'],
        predictions['conservation_preds'],
        CONSERVATION_CLASSES,
        save_path='visualizations/confusion_matrix.png'
    )
    
    plot_class_distribution(
        predictions['conservation_labels'],
        save_path='visualizations/class_distribution.png'
    )
    
    print("\nSaving results...")
    save_results(metrics, predictions)
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
