# -*- coding: utf-8 -*-
"""
Evaluation utilities for Brain Tumor Analysis project
Contains functions for statistical analysis, model comparison, and performance evaluation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score
from config import *

def calculate_detailed_metrics(y_true, y_pred, class_names):
    """Calculate detailed metrics for model evaluation"""
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class
    }

def mcnemar_test(y_true, y_pred1, y_pred2):
    """Perform McNemar's test to compare two models"""
    correct1 = (y_true == y_pred1)
    correct2 = (y_true == y_pred2)
    
    # Contingency table
    both_correct = np.sum(correct1 & correct2)
    only_model1_correct = np.sum(correct1 & ~correct2)
    only_model2_correct = np.sum(~correct1 & correct2)
    both_wrong = np.sum(~correct1 & ~correct2)
    
    # McNemar's test statistic
    if only_model1_correct + only_model2_correct > 0:
        chi2 = (abs(only_model1_correct - only_model2_correct) - 1)**2 / (only_model1_correct + only_model2_correct)
        p_value = 1 - stats.chi2.cdf(chi2, 1)
    else:
        chi2 = 0
        p_value = 1
    
    return chi2, p_value, {
        'both_correct': both_correct,
        'only_model1_correct': only_model1_correct,
        'only_model2_correct': only_model2_correct,
        'both_wrong': both_wrong
    }

def wilson_confidence_interval(accuracy, n, confidence=CONFIDENCE_LEVEL):
    """Calculate Wilson confidence interval for accuracy"""
    z = stats.norm.ppf((1 + confidence) / 2)
    p = accuracy
    denominator = 1 + z**2/n
    centre_adjusted = p + z**2/(2*n)
    adjustment = z * np.sqrt((p*(1-p) + z**2/(4*n))/n)
    
    lower = (centre_adjusted - adjustment) / denominator
    upper = (centre_adjusted + adjustment) / denominator
    
    return lower, upper

def cohens_h(p1, p2):
    """Calculate Cohen's h effect size"""
    return 2 * (np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2)))

def interpret_effect_size(effect_size):
    """Interpret Cohen's h effect size"""
    abs_effect = abs(effect_size)
    if abs_effect < 0.2:
        return "Small"
    elif abs_effect < 0.5:
        return "Medium"
    else:
        return "Large"

def create_comparison_table(svm_metrics, clip_metrics, metric_names):
    """Create comparison table between models"""
    comparison_data = []
    
    for i, metric in enumerate(metric_names):
        svm_val = svm_metrics[i]
        clip_val = clip_metrics[i]
        improvement = ((clip_val - svm_val) / svm_val * 100) if svm_val != 0 else 0
        
        comparison_data.append({
            'Metric': metric,
            'SVM': f'{svm_val:.4f}',
            'CLIP': f'{clip_val:.4f}',
            'Improvement (%)': f'{improvement:+.2f}%'
        })
    
    return pd.DataFrame(comparison_data)

def create_per_class_comparison_table(svm_metrics_per_class, clip_metrics_per_class, class_names):
    """Create per-class comparison table"""
    comparison_data = []
    
    for i, class_name in enumerate(class_names):
        comparison_data.append({
            'Class': class_name,
            'SVM_Precision': f'{svm_metrics_per_class["precision_per_class"][i]:.4f}',
            'CLIP_Precision': f'{clip_metrics_per_class["precision_per_class"][i]:.4f}',
            'Precision_Diff': f'{(clip_metrics_per_class["precision_per_class"][i] - svm_metrics_per_class["precision_per_class"][i]):+.4f}',
            'SVM_Recall': f'{svm_metrics_per_class["recall_per_class"][i]:.4f}',
            'CLIP_Recall': f'{clip_metrics_per_class["recall_per_class"][i]:.4f}',
            'Recall_Diff': f'{(clip_metrics_per_class["recall_per_class"][i] - svm_metrics_per_class["recall_per_class"][i]):+.4f}',
            'SVM_F1': f'{svm_metrics_per_class["f1_per_class"][i]:.4f}',
            'CLIP_F1': f'{clip_metrics_per_class["f1_per_class"][i]:.4f}',
            'F1_Diff': f'{(clip_metrics_per_class["f1_per_class"][i] - svm_metrics_per_class["f1_per_class"][i]):+.4f}'
        })
    
    return pd.DataFrame(comparison_data)

def perform_statistical_analysis(y_test_svm, y_pred_svm, y_test_clip, y_pred_clip):
    """Perform comprehensive statistical analysis"""
    print("="*60)
    print("STATISTICAL ANALYSIS AND SIGNIFICANCE")
    print("="*60)
    
    # Calculate accuracies
    svm_accuracy = np.mean(y_test_svm == y_pred_svm)
    clip_accuracy = np.mean(y_test_clip == y_pred_clip)
    
    # McNemar's test
    chi2_stat, p_value, contingency = mcnemar_test(y_test_svm, y_pred_svm, y_pred_clip)
    
    print(f"McNemar's Test Results:")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Significance level (α = {SIGNIFICANCE_LEVEL}): {'Significant' if p_value < SIGNIFICANCE_LEVEL else 'Not Significant'}")
    
    print(f"\nContingency Table:")
    print(f"Both models correct: {contingency['both_correct']}")
    print(f"Only SVM correct: {contingency['only_model1_correct']}")
    print(f"Only CLIP correct: {contingency['only_model2_correct']}")
    print(f"Both models wrong: {contingency['both_wrong']}")
    
    # Confidence intervals
    svm_ci = wilson_confidence_interval(svm_accuracy, len(y_test_svm))
    clip_ci = wilson_confidence_interval(clip_accuracy, len(y_test_clip))
    
    print(f"\n95% Confidence Intervals:")
    print(f"SVM Accuracy: {svm_accuracy:.4f} [{svm_ci[0]:.4f}, {svm_ci[1]:.4f}]")
    print(f"CLIP Accuracy: {clip_accuracy:.4f} [{clip_ci[0]:.4f}, {clip_ci[1]:.4f}]")
    
    # Effect size
    effect_size = cohens_h(clip_accuracy, svm_accuracy)
    effect_interpretation = interpret_effect_size(effect_size)
    
    print(f"\nEffect Size (Cohen's h): {effect_size:.4f}")
    print(f"Effect Size Interpretation: {effect_interpretation}")
    
    return {
        'mcnemar_chi2': chi2_stat,
        'mcnemar_p_value': p_value,
        'contingency': contingency,
        'svm_ci': svm_ci,
        'clip_ci': clip_ci,
        'effect_size': effect_size,
        'effect_interpretation': effect_interpretation
    }

def print_model_analysis(svm_results, clip_results, class_names):
    """Print comprehensive model analysis"""
    print("="*80)
    print("COMPREHENSIVE MODEL ANALYSIS: SVM vs CLIP")
    print("="*80)
    
    # Extract metrics
    svm_accuracy = svm_results['accuracy']
    svm_precision = svm_results['precision']
    svm_recall = svm_results['recall']
    svm_f1 = svm_results['f1_score']
    
    clip_accuracy = clip_results['accuracy']
    clip_precision = clip_results['precision']
    clip_recall = clip_results['recall']
    clip_f1 = clip_results['f1_score']
    
    # Count errors
    svm_errors = len(np.where(svm_results.get('true_labels', []) != svm_results['predictions'])[0]) if 'true_labels' in svm_results else 0
    clip_errors = len(np.where(clip_results['true_labels'] != clip_results['predictions'])[0])
    
    print("\n1. MAIN METRICS COMPARISON")
    print("-" * 40)
    
    # Create comparison table
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    svm_scores = [svm_accuracy, svm_precision, svm_recall, svm_f1]
    clip_scores = [clip_accuracy, clip_precision, clip_recall, clip_f1]
    
    comparison_df = create_comparison_table(svm_scores, clip_scores, metrics)
    print(comparison_df.to_string(index=False))
    
    print("\n2. PER-CLASS ANALYSIS")
    print("-" * 40)
    
    per_class_df = create_per_class_comparison_table(svm_results, clip_results, class_names)
    print(per_class_df.to_string(index=False))
    
    print("\n3. MODEL STRENGTHS AND WEAKNESSES")
    print("-" * 50)
    
    print("\n🔍 SVM ANALYSIS:")
    print("Strengths:")
    print("  ✓ Fast training and efficient")
    print("  ✓ Lower memory requirements")
    print("  ✓ Interpretable and explainable")
    print("  ✓ Robust against overfitting with small datasets")
    print("  ✓ No GPU required for training")
    
    print("\nWeaknesses:")
    print("  ✗ Requires extensive preprocessing (PCA, scaling)")
    print("  ✗ Limited performance on complex visual data")
    print("  ✗ Difficulty capturing high-level features from images")
    print("  ✗ More manual hyperparameter tuning")
    
    print(f"\nSVM Results:")
    print(f"  • Test Accuracy: {svm_accuracy:.4f} ({svm_accuracy*100:.2f}%)")
    print(f"  • Precision: {svm_precision:.4f}")
    print(f"  • Recall: {svm_recall:.4f}")
    print(f"  • F1-Score: {svm_f1:.4f}")
    if svm_errors > 0:
        print(f"  • Errors: {svm_errors} samples")
    
    print("\n🤖 CLIP ANALYSIS:")
    print("Strengths:")
    print("  ✓ Pre-trained knowledge from large datasets")
    print("  ✓ End-to-end learning without manual feature engineering")
    print("  ✓ Multimodal capabilities (vision + text)")
    print("  ✓ Better generalization for visual data")
    print("  ✓ State-of-the-art architecture")
    
    print("\nWeaknesses:")
    print("  ✗ Requires high computational resources")
    print("  ✗ Longer training time")
    print("  ✗ Less interpretable (black box)")
    print("  ✗ Requires GPU for optimal training")
    print("  ✗ Large model size")
    
    print(f"\nCLIP Results:")
    print(f"  • Test Accuracy: {clip_accuracy:.4f} ({clip_accuracy*100:.2f}%)")
    print(f"  • Precision: {clip_precision:.4f}")
    print(f"  • Recall: {clip_recall:.4f}")
    print(f"  • F1-Score: {clip_f1:.4f}")
    print(f"  • Errors: {clip_errors} samples")

def print_final_recommendations(svm_results, clip_results, statistical_results):
    """Print final recommendations based on analysis"""
    print("\n🎯 FINAL CONCLUSIONS:")
    
    svm_accuracy = svm_results['accuracy']
    clip_accuracy = clip_results['accuracy']
    p_value = statistical_results['mcnemar_p_value']
    effect_interpretation = statistical_results['effect_interpretation']
    effect_size = statistical_results['effect_size']
    
    if clip_accuracy > svm_accuracy:
        improvement = ((clip_accuracy - svm_accuracy) / svm_accuracy * 100)
        print(f"✅ CLIP shows better performance with:")
        print(f"   • Accuracy improvement: {improvement:+.2f}%")
    else:
        print(f"✅ SVM shows better performance")
    
    print(f"\n📈 STATISTICAL SIGNIFICANCE:")
    if p_value < SIGNIFICANCE_LEVEL:
        print(f"✅ Performance difference is statistically SIGNIFICANT (p={p_value:.4f})")
    else:
        print(f"❌ Performance difference is statistically NOT SIGNIFICANT (p={p_value:.4f})")
    
    print(f"📏 Effect size: {effect_interpretation} ({effect_size:.4f})")
    
    print(f"\n🏆 RECOMMENDATIONS:")
    print(f"1. FOR PRODUCTION:")
    if clip_accuracy > svm_accuracy and p_value < SIGNIFICANCE_LEVEL:
        print(f"   • Use CLIP for best accuracy")
        print(f"   • Consider computational cost vs performance gain")
    else:
        print(f"   • SVM remains viable for resource-constrained applications")
    
    print(f"\n2. FOR RESEARCH:")
    print(f"   • Explore ensemble methods (SVM + CLIP)")
    print(f"   • Fine-tune CLIP with domain-specific data")
    print(f"   • Investigate failure cases in both models")
    
    print(f"\n3. FOR DEPLOYMENT:")
    print(f"   • SVM: Suitable for edge devices, real-time applications")
    print(f"   • CLIP: Suitable for cloud-based applications requiring high accuracy")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)