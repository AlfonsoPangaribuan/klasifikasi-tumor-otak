# -*- coding: utf-8 -*-
"""
SVM model utilities for Brain Tumor Analysis project
Contains functions for SVM training, evaluation, and analysis
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from config import *

def train_svm_model(X_train, y_train):
    """Train SVM classifier"""
    print("Training SVM Classifier...")
    
    svm_model = SVC(
        kernel=SVM_CONFIG['kernel'],
        C=SVM_CONFIG['C'],
        gamma=SVM_CONFIG['gamma'],
        random_state=SVM_CONFIG['random_state']
    )
    
    # Train model
    svm_model.fit(X_train, y_train)
    print("SVM model training completed!")
    
    return svm_model

def evaluate_svm_model(model, X_test, y_test, class_names):
    """Evaluate SVM model and return detailed metrics"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_test, y_pred, average=None)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    
    print(f"SVM Test Accuracy: {accuracy:.4f}")
    print("\n" + "="*50)
    print("SVM CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Detailed per-class metrics
    cm = confusion_matrix(y_test, y_pred)
    print("\nDetailed per-class metrics:")
    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        print(f"{class_name}:")
        print(f"  Precision: {class_precision:.3f}")
        print(f"  Recall: {class_recall:.3f}")
        print(f"  F1-Score: {class_f1:.3f}")
        print()
    
    results = {
        'predictions': y_pred,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm
    }
    
    return results

def validate_svm_model(model, X_val, y_val):
    """Validate SVM model on validation set"""
    y_val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f"SVM Validation Accuracy: {val_accuracy:.4f}")
    return val_accuracy, y_val_pred