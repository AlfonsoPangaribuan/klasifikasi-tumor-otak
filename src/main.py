#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main execution file for Brain Tumor Analysis project
Orchestrates the complete analysis pipeline comparing SVM and CLIP models
"""

import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader

# Import our modular components
from config import *
from data_utils import *
from visualization_utils import *
from svm_model import *
from clip_model import *
from evaluation_utils import *

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    """Main execution function"""
    print("=" * 80)
    print("BRAIN TUMOR MRI ANALYSIS WITH SVM AND CLIP")
    print("=" * 80)
    print("Comparing traditional machine learning (SVM) vs modern deep learning (CLIP)")
    print()
    
    # 1. Download and Load Dataset
    print("STEP 1: DATASET PREPARATION")
    print("-" * 40)
    
    dataset_path = download_dataset()
    explore_dataset_structure(dataset_path)
    
    print("\nLoading dataset...")
    X, y = load_data(dataset_path)
    print(f"Dataset loaded: {X.shape[0]} images with size {X.shape[1]}x{X.shape[2]}")
    print(f"Classes: {np.unique(y)}")
    
    # 2. Exploratory Data Analysis
    print("\nSTEP 2: EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    # Plot class distribution
    class_counts = plot_class_distribution(y, "Original Dataset Class Distribution")
    
    # Plot sample images
    plot_sample_images(X, y, np.unique(y), "Sample Images from Each Class")
    
    # Plot pixel statistics
    plot_pixel_statistics(X)
    
    # 3. Data Augmentation Demonstration
    print("\nSTEP 3: DATA AUGMENTATION DEMONSTRATION")
    print("-" * 40)
    
    augmentation_functions = get_augmentation_functions()
    sample_idx = 0
    sample_img = X[sample_idx]
    sample_label = y[sample_idx]
    
    plot_augmentation_demo(sample_img, sample_label, augmentation_functions)
    
    # 4. Data Preprocessing and Splitting
    print("\nSTEP 4: DATA PREPROCESSING AND SPLITTING")
    print("-" * 40)
    
    # Preprocess data
    X_normalized, y_encoded, label_encoder = preprocess_data(X, y)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_normalized, y_encoded)
    
    # 5. SVM Model Training and Evaluation
    print("\nSTEP 5: SVM MODEL TRAINING AND EVALUATION")
    print("-" * 40)
    
    # Prepare data for SVM
    X_train_pca, X_val_pca, X_test_pca, scaler, pca = prepare_svm_data(X_train, X_val, X_test)
    
    # Train SVM model
    svm_model = train_svm_model(X_train_pca, y_train)
    
    # Validate SVM model
    svm_val_accuracy, y_val_pred_svm = validate_svm_model(svm_model, X_val_pca, y_val)
    
    # Evaluate SVM model on test set
    svm_results = evaluate_svm_model(svm_model, X_test_pca, y_test, label_encoder.classes_)
    
    # Plot SVM confusion matrix
    plot_confusion_matrix(y_test, svm_results['predictions'], label_encoder.classes_, 
                         "SVM Confusion Matrix", cmap='Blues')
    
    # Plot incorrect predictions for SVM
    plot_incorrect_predictions(X_test, y_test, svm_results['predictions'], 
                              label_encoder.classes_, "SVM Incorrect Predictions")
    
    # 6. CLIP Model Training and Evaluation
    print("\nSTEP 6: CLIP MODEL TRAINING AND EVALUATION")
    print("-" * 40)
    
    # Check for PyTorch and transformers installation
    try:
        device = setup_device()
        
        # Run CLIP grid search
        clip_results_list, best_clip_model, best_clip_params, best_clip_val_acc = run_clip_grid_search(
            X_train, y_train, X_val, y_val, label_encoder.classes_, device
        )
        
        # Create test dataset for best model evaluation
        _, _, test_dataset = create_clip_datasets(
            X_train, y_train, X_val, y_val, X_test, y_test, label_encoder.classes_
        )
        
        # Evaluate best CLIP model on test set
        test_loader = DataLoader(test_dataset, batch_size=best_clip_params['batch_size'], shuffle=False)
        clip_results = evaluate_clip_detailed(best_clip_model, test_loader, device, label_encoder.classes_)
        
        # Plot CLIP confusion matrix
        plot_confusion_matrix(clip_results['true_labels'], clip_results['predictions'], 
                             label_encoder.classes_, "CLIP Confusion Matrix", cmap='Reds')
        
        # Plot CLIP grid search results
        plot_grid_search_results(clip_results_list)
        results_df = create_results_dataframe(clip_results_list)
        print("\nGrid Search Results (sorted by Best Val Accuracy):")
        print("=" * 120)
        print(results_df.to_string(index=False))
        
        print(f"\nBest CLIP Parameters:")
        print("=" * 60)
        for key, value in best_clip_params.items():
            print(f"{key}: {value}")
        print(f"Best Validation Accuracy: {best_clip_val_acc:.4f}")
        print(f"Test Accuracy: {clip_results['accuracy']:.4f}")
        
        clip_available = True
        
    except ImportError as e:
        print(f"CLIP training skipped due to missing dependencies: {e}")
        print("Please install: pip install torch transformers pillow")
        clip_available = False
        clip_results = None
        best_clip_params = None
    
    # 7. Model Comparison and Analysis
    if clip_available and clip_results is not None:
        print("\nSTEP 7: COMPREHENSIVE MODEL COMPARISON")
        print("-" * 40)
        
        # Statistical analysis
        statistical_results = perform_statistical_analysis(
            y_test, svm_results['predictions'], 
            clip_results['true_labels'], clip_results['predictions']
        )
        
        # Detailed model analysis
        print_model_analysis(svm_results, clip_results, label_encoder.classes_)
        
        # Final recommendations
        print_final_recommendations(svm_results, clip_results, statistical_results)
        
        # Visualization comparisons
        print("\nSTEP 8: VISUALIZATION COMPARISONS")
        print("-" * 40)
        
        # Compare overall metrics
        svm_scores = [svm_results['accuracy'], svm_results['precision'], 
                     svm_results['recall'], svm_results['f1_score']]
        clip_scores = [clip_results['accuracy'], clip_results['precision'], 
                      clip_results['recall'], clip_results['f1_score']]
        
        plot_model_comparison(svm_scores, clip_scores)
        
        # Plot per-class comparisons
        metrics_to_compare = ['precision_per_class', 'recall_per_class', 'f1_per_class']
        metric_names = ['Precision', 'Recall', 'F1-Score']
        
        for metric, name in zip(metrics_to_compare, metric_names):
            plot_per_class_comparison(
                svm_results[metric], clip_results[metric], 
                label_encoder.classes_, name
            )
    
    else:
        print("\nSTEP 7: SVM-ONLY ANALYSIS")
        print("-" * 40)
        print("CLIP model evaluation was skipped.")
        print("Install required dependencies to run full comparison.")
        
        print(f"\nSVM Final Results:")
        print(f"Validation Accuracy: {svm_val_accuracy:.4f}")
        print(f"Test Accuracy: {svm_results['accuracy']:.4f}")
        print(f"Precision: {svm_results['precision']:.4f}")
        print(f"Recall: {svm_results['recall']:.4f}")
        print(f"F1-Score: {svm_results['f1_score']:.4f}")
    
    # 8. Summary
    print("\nFINAL SUMMARY")
    print("=" * 60)
    print(f"Dataset: {len(X)} samples with {len(label_encoder.classes_)} classes")
    print(f"Test set: {len(y_test)} samples for objective evaluation")
    print(f"Evaluation method: Stratified train-test split")
    
    if clip_available and clip_results is not None:
        print(f"\nModel Performance Summary:")
        print(f"SVM Test Accuracy: {svm_results['accuracy']:.4f}")
        print(f"CLIP Test Accuracy: {clip_results['accuracy']:.4f}")
        
        if clip_results['accuracy'] > svm_results['accuracy']:
            improvement = ((clip_results['accuracy'] - svm_results['accuracy']) / svm_results['accuracy'] * 100)
            print(f"CLIP improvement: {improvement:+.2f}%")
        else:
            print("SVM performed better than CLIP")
    else:
        print(f"\nSVM Test Accuracy: {svm_results['accuracy']:.4f}")
    
    print("\nAnalysis completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()