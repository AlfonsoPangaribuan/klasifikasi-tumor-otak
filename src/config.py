# -*- coding: utf-8 -*-
"""
Configuration file for Brain Tumor Analysis project
Contains all constants, hyperparameters, and settings
"""

import os

# Dataset Configuration
DATASET_NAME = "masoudnickparvar/brain-tumor-mri-dataset"
IMAGE_SIZE = (256, 256)
CLIP_IMAGE_SIZE = (224, 224)
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Data Split Configuration
TRAIN_SIZE = 0.6
VAL_SIZE = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42

# SVM Configuration
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 1.0,
    'gamma': 'scale',
    'random_state': RANDOM_STATE
}

# PCA Configuration
PCA_VARIANCE_RATIO = 0.95

# CLIP Configuration
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_MAX_LENGTH = 77

# CLIP Text Templates for each class
TEXT_TEMPLATES = {
    'glioma': [
        'a medical brain MRI scan showing glioma brain tumor with irregular borders',
        'brain MRI image displaying glioma tumor with characteristic infiltrative pattern',
        'medical brain scan revealing glioma neoplasm with heterogeneous appearance',
        'MRI brain image showing malignant glioma with surrounding edema'
    ],
    'meningioma': [
        'a medical brain MRI scan showing meningioma tumor with well-defined borders',
        'brain MRI image displaying meningioma with characteristic dural attachment',
        'medical brain scan revealing meningioma with homogeneous enhancement',
        'MRI brain image showing benign meningioma with smooth margins'
    ],
    'notumor': [
        'a normal brain MRI scan without any tumor or abnormality',
        'healthy brain MRI image showing normal brain tissue structure',
        'medical brain scan displaying normal neuroanatomy without lesions',
        'MRI brain image of healthy brain tissue without pathology'
    ],
    'pituitary': [
        'a medical brain MRI scan showing pituitary adenoma in sella turcica',
        'brain MRI image displaying pituitary tumor with sellar expansion',
        'medical brain scan revealing pituitary mass with characteristic location',
        'MRI brain image showing pituitary adenoma with suprasellar extension'
    ]
}

# Grid Search Parameters for CLIP
GRID_SEARCH_PARAMS = {
    'epochs': [25, 50],
    'learning_rate': [1e-2, 1e-3],
    'optimizer': ['adamw'],
    'batch_size': [32, 64],
    'freeze_clip': [True, False],
    'weight_decay': [0.01]
}

# Training Configuration
PATIENCE = 5
LABEL_SMOOTHING = 0.1
GRADIENT_CLIP_MAX_NORM = 1.0

# Visualization Configuration
FIGURE_SIZE_LARGE = (20, 15)
FIGURE_SIZE_MEDIUM = (15, 10)
FIGURE_SIZE_SMALL = (10, 6)

# Statistical Analysis Configuration
CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05