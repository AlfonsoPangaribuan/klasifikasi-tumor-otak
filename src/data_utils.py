# -*- coding: utf-8 -*-
"""
Data utilities for Brain Tumor Analysis project
Contains functions for data loading, preprocessing, augmentation, and visualization
"""

import numpy as np
import pandas as pd
import cv2
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from skimage.transform import rotate, AffineTransform, warp
from skimage.exposure import adjust_gamma
import kagglehub

from config import *

def download_dataset():
    """Download brain tumor dataset from Kaggle"""
    print("Downloading dataset...")
    path = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset downloaded to: {path}")
    return path

def explore_dataset_structure(data_path):
    """Explore and print dataset folder structure"""
    print("Dataset structure:")
    for root, dirs, files in os.walk(data_path):
        level = root.replace(data_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files only
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files)-5} more files")

def load_data(data_path):
    """Load images and labels from dataset"""
    images = []
    labels = []
    
    # Look for Training and Testing folders
    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)
        if os.path.isdir(folder_path) and folder in ['Training', 'Testing']:
            print(f"Processing {folder} folder...")
            
            # Iterate through classes (glioma, meningioma, notumor, pituitary)
            for class_name in os.listdir(folder_path):
                class_path = os.path.join(folder_path, class_name)
                if os.path.isdir(class_path):
                    print(f"  Loading {class_name} images...")
                    
                    # Load all images in class
                    for img_name in os.listdir(class_path):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            # Read image
                            img = cv2.imread(img_path)
                            if img is not None:
                                # Convert to grayscale and resize
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                img = cv2.resize(img, IMAGE_SIZE)
                                images.append(img)
                                labels.append(class_name)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels)

def preprocess_data(X, y):
    """Normalize data and encode labels"""
    # Normalize images
    X_normalized = X.astype('float32') / 255.0
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print("Classes mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"{i}: {class_name}")
    
    return X_normalized, y_encoded, label_encoder

def split_data(X, y):
    """Split data into train, validation, and test sets"""
    # First split: 60% training and 40% remaining
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=VAL_SIZE + TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    # Second split: 20% validation and 20% testing from remaining 40%
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE),
        random_state=RANDOM_STATE,
        stratify=y_temp
    )
    
    print("\nData splitting results:")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Check class distribution in each split
    print("\nClass distribution per split:")
    for split_name, y_split in [('Training', y_train), ('Validation', y_val), ('Testing', y_test)]:
        class_dist = Counter(y_split)
        print(f"{split_name}: {dict(class_dist)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def prepare_svm_data(X_train, X_val, X_test):
    """Prepare data for SVM training (flatten, scale, PCA)"""
    # Flatten images for SVM (SVM needs 1D input)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print(f"Shape after flattening:")
    print(f"Training: {X_train_flat.shape}")
    print(f"Validation: {X_val_flat.shape}")
    print(f"Testing: {X_test_flat.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_val_scaled = scaler.transform(X_val_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    print("\nData successfully normalized with StandardScaler")
    
    # Dimensionality reduction with PCA
    print("\nPerforming PCA for dimensionality reduction...")
    pca = PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print(f"Dimensions after PCA: {X_train_pca.shape[1]}")
    print(f"Variance retained: {pca.explained_variance_ratio_.sum():.3f}")
    
    return X_train_pca, X_val_pca, X_test_pca, scaler, pca

# Data Augmentation Functions
def anticlockwise_rotation(img):
    """Apply anticlockwise rotation augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    angle = random.randint(0, 180)
    return rotate(img, angle)

def clockwise_rotation(img):
    """Apply clockwise rotation augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    angle = random.randint(0, 180)
    return rotate(img, -angle)

def flip_up_down(img):
    """Apply vertical flip augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    return np.flipud(img)

def add_brightness(img):
    """Apply brightness augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    img = adjust_gamma(img, gamma=0.5, gain=1)
    return img

def blur_image(img):
    """Apply blur augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    return cv2.GaussianBlur(img, (9,9), 0)

def sheared(img):
    """Apply shear augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    transform = AffineTransform(shear=0.2)
    shear_image = warp(img, transform, mode='wrap')
    return shear_image

def warp_shift(img):
    """Apply warp shift augmentation"""
    img = cv2.cvtColor(img, 0) if len(img.shape) == 3 else img
    img = cv2.resize(img, CLIP_IMAGE_SIZE)
    transform = AffineTransform(translation=(0,40))
    warp_image = warp(img, transform, mode="wrap")
    return warp_image

def get_augmentation_functions():
    """Get list of available augmentation functions"""
    return [
        ('Original', lambda x: x),
        ('Anticlockwise Rotation', anticlockwise_rotation),
        ('Clockwise Rotation', clockwise_rotation),
        ('Flip Up Down', flip_up_down),
        ('Add Brightness', add_brightness),
        ('Blur', blur_image),
        ('Sheared', sheared),
        ('Warp Shift', warp_shift)
    ]