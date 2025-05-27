# -*- coding: utf-8 -*-
"""
Visualization utilities for Brain Tumor Analysis project
Contains functions for plotting and visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix
from config import *

def plot_class_distribution(y, title="Class Distribution"):
    """Plot class distribution as bar chart and pie chart"""
    class_counts = Counter(y)
    
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    
    # Bar chart
    plt.subplot(1, 2, 1)
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Number of Images per Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)
    
    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title('Class Proportion in Dataset')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return class_counts

def plot_sample_images(X, y, classes, title="Sample Images from Each Class"):
    """Plot sample images from each class"""
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    unique_classes = np.unique(y) if isinstance(y[0], str) else classes
    
    for i, class_name in enumerate(unique_classes):
        # Get indices for this class
        if isinstance(y[0], str):
            class_indices = np.where(y == class_name)[0]
        else:
            class_indices = np.where(y == i)[0]
        
        # Select 4 random samples
        sample_indices = np.random.choice(class_indices, 4, replace=False)
        
        for j, idx in enumerate(sample_indices):
            plt.subplot(len(unique_classes), 4, i*4 + j + 1)
            plt.imshow(X[idx], cmap='gray')
            plt.title(f'{class_name}')
            plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_pixel_statistics(X):
    """Plot pixel intensity statistics"""
    print("Dataset Statistics:")
    print(f"Total images: {len(X)}")
    print(f"Image size: {X.shape[1]} x {X.shape[2]}")
    print(f"Min pixel value: {X.min()}")
    print(f"Max pixel value: {X.max()}")
    print(f"Mean pixel value: {X.mean():.2f}")
    print(f"Pixel std: {X.std():.2f}")
    
    # Histogram of pixel intensity distribution
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    plt.hist(X.flatten(), bins=50, alpha=0.7, edgecolor='black')
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_augmentation_demo(sample_img, sample_label, augmentation_functions):
    """Demonstrate augmentation results"""
    plt.figure(figsize=(16, 10))
    
    for i, (name, func) in enumerate(augmentation_functions):
        plt.subplot(2, 4, i+1)
        augmented_img = func(sample_img)
        plt.imshow(augmented_img, cmap='gray')
        plt.title(f'{name}\n({sample_label})')
        plt.axis('off')
    
    plt.suptitle('Image Augmentation Results', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", cmap='Blues'):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    return cm

def plot_incorrect_predictions(X_test, y_test, y_pred, class_names, title="Incorrect Predictions"):
    """Plot examples of incorrect predictions"""
    incorrect_indices = np.where(y_test != y_pred)[0]
    
    if len(incorrect_indices) > 0:
        print(f"Number of incorrect predictions: {len(incorrect_indices)}")
        
        plt.figure(figsize=FIGURE_SIZE_MEDIUM)
        n_samples = min(8, len(incorrect_indices))
        
        for i in range(n_samples):
            idx = incorrect_indices[i]
            if len(X_test.shape) == 4:  # If batch dimension exists
                img = X_test[idx]
            else:
                img = X_test[idx].reshape(IMAGE_SIZE)
            
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]
            
            plt.subplot(2, 4, i+1)
            plt.imshow(img, cmap='gray')
            plt.title(f'True: {true_label}\nPred: {pred_label}', color='red')
            plt.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()
    else:
        print("All predictions are correct!")

def plot_model_comparison(svm_scores, clip_scores, metrics=['Accuracy', 'Precision', 'Recall', 'F1-Score']):
    """Plot comparison between SVM and CLIP models"""
    plt.figure(figsize=FIGURE_SIZE_LARGE)
    
    # Overall comparison
    plt.subplot(2, 3, 1)
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, svm_scores, width, label='SVM', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, clip_scores, width, label='CLIP', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)
    
    # Add values on bars
    for i, (svm_val, clip_val) in enumerate(zip(svm_scores, clip_scores)):
        plt.text(i - width/2, svm_val + 0.01, f'{svm_val:.3f}', ha='center', va='bottom')
        plt.text(i + width/2, clip_val + 0.01, f'{clip_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()

def plot_per_class_comparison(svm_scores_per_class, clip_scores_per_class, class_names, metric_name):
    """Plot per-class comparison between models"""
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    x = np.arange(len(class_names))
    width = 0.35
    
    plt.bar(x - width/2, svm_scores_per_class, width, label='SVM', alpha=0.8, color='skyblue')
    plt.bar(x + width/2, clip_scores_per_class, width, label='CLIP', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Classes')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} Comparison by Class')
    plt.xticks(x, class_names, rotation=45)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def plot_training_history(train_losses, val_accuracies, title="Training History"):
    """Plot training loss and validation accuracy"""
    plt.figure(figsize=FIGURE_SIZE_SMALL)
    
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, val_accuracies, 'r-', label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_grid_search_results(results):
    """Plot grid search results"""
    plt.figure(figsize=FIGURE_SIZE_MEDIUM)
    
    combination_names = [f"C{i+1}" for i in range(len(results))]
    val_accs = [r['best_val_acc'] for r in results]
    training_times = [r['training_time'] for r in results]
    
    # Validation accuracy per combination
    plt.subplot(1, 2, 1)
    plt.bar(combination_names, val_accs)
    plt.title('Best Validation Accuracy per Combination')
    plt.xlabel('Combination')
    plt.ylabel('Validation Accuracy')
    plt.xticks(rotation=45)
    
    # Training time per combination
    plt.subplot(1, 2, 2)
    plt.bar(combination_names, training_times)
    plt.title('Training Time per Combination')
    plt.xlabel('Combination')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def create_results_dataframe(results):
    """Create DataFrame from grid search results"""
    results_df = pd.DataFrame([
        {
            'Combination': f"C{i+1}",
            'Epochs': r['params']['epochs'],
            'Learning_Rate': r['params']['learning_rate'],
            'Optimizer': r['params']['optimizer'],
            'Batch_Size': r['params']['batch_size'],
            'Best_Val_Acc': f"{r['best_val_acc']:.4f}",
            'Final_Val_Acc': f"{r['final_val_acc']:.4f}",
            'Training_Time': f"{r['training_time']:.2f}s"
        }
        for i, r in enumerate(results)
    ])
    
    # Sort by best validation accuracy
    results_df = results_df.sort_values('Best_Val_Acc', ascending=False)
    return results_df