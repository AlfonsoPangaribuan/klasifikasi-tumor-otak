# -*- coding: utf-8 -*-
"""
CLIP model utilities for Brain Tumor Analysis project
Contains CLIP model classes, training, evaluation, and grid search functions
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import time
from PIL import Image

from config import *

# CLIP Dataset Class
class BrainTumorDataset(Dataset):
    def __init__(self, images, labels, processor, class_names):
        self.images = images
        self.labels = labels
        self.processor = processor
        self.class_names = class_names

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert numpy array to PIL Image
        image = self.images[idx]
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
        image = (image * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)

        # Get class name and random text description for variation
        label = self.labels[idx]
        class_name = self.class_names[label]
        text_options = TEXT_TEMPLATES[class_name]
        text = np.random.choice(text_options)  # Random selection for augmentation

        # Process image and text
        inputs = self.processor(
            text=text,
            images=image_pil,
            return_tensors="pt",
            padding="max_length",
            max_length=CLIP_MAX_LENGTH,
            truncation=True
        )

        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

# CLIP Classifier Class
class CLIPClassifier(nn.Module):
    def __init__(self, num_classes=4, model_name=CLIP_MODEL_NAME, freeze_clip=False):
        super(CLIPClassifier, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)

        # Optional: Freeze CLIP weights for more stable fine-tuning
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False

        # Multi-layer classifier with regularization
        hidden_dim = 256
        self.classifier = nn.Sequential(
            nn.Linear(self.clip_model.config.projection_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.clip_model.config.projection_dim)

    def forward(self, pixel_values, input_ids, attention_mask):
        # Extract features from CLIP
        outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Combine image and text features
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Fusion strategy: weighted average
        combined_features = (image_features + text_features) / 2
        combined_features = self.layer_norm(combined_features)

        # Classification
        logits = self.classifier(combined_features)
        return logits

def setup_device():
    """Setup and return the device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def train_clip_model(model, train_loader, val_loader, optimizer, criterion, epochs, device, scheduler=None):
    """Train CLIP model with early stopping"""
    model.train()
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    patience_counter = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)

            optimizer.step()
            epoch_loss += loss.item()

        # Learning rate scheduling
        if scheduler:
            scheduler.step()

        # Evaluate on validation set
        val_acc = evaluate_clip_model(model, val_loader, device)
        train_losses.append(epoch_loss / len(train_loader))
        val_accuracies.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if epoch % 2 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}')

    return train_losses, val_accuracies

def evaluate_clip_model(model, data_loader, device):
    """Evaluate CLIP model and return accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

def get_detailed_clip_predictions(model, data_loader, device):
    """Get detailed predictions for evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            pixel_values = batch['pixel_values'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(pixel_values, input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)

def evaluate_clip_detailed(model, data_loader, device, class_names):
    """Get detailed evaluation metrics for CLIP model"""
    y_pred, y_true = get_detailed_clip_predictions(model, data_loader, device)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)
    
    print(f"CLIP Test Accuracy: {accuracy:.4f}")
    print("\n" + "="*50)
    print("CLIP CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    results = {
        'predictions': y_pred,
        'true_labels': y_true,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return results

def run_clip_grid_search(X_train, y_train, X_val, y_val, class_names, device):
    """Run grid search for CLIP hyperparameters"""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    
    # Generate all parameter combinations
    all_params = list(ParameterGrid(GRID_SEARCH_PARAMS))
    print(f"Total parameter combinations: {len(all_params)}")
    
    # Display all combinations
    for i, params in enumerate(all_params):
        print(f"Combination {i+1}: {params}")
    
    results = []
    best_val_acc = 0
    best_params = None
    best_model = None
    
    print("\nStarting Grid Search...")
    print("="*60)
    
    for i, params in enumerate(all_params):
        print(f"\nCombination {i+1}/{len(all_params)}: {params}")
        print("-" * 50)
        
        # Recreate datasets for each iteration (for random text selection)
        train_dataset = BrainTumorDataset(X_train, y_train, processor, class_names)
        val_dataset = BrainTumorDataset(X_val, y_val, processor, class_names)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
        
        # Initialize model
        model = CLIPClassifier(
            num_classes=len(class_names),
            freeze_clip=params['freeze_clip']
        )
        model.to(device)
        
        # Initialize optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params['epochs'])
        
        # Label smoothing to reduce overfitting
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        
        # Training
        start_time = time.time()
        train_losses, val_accuracies = train_clip_model(
            model, train_loader, val_loader, optimizer, criterion,
            params['epochs'], device, scheduler
        )
        training_time = time.time() - start_time
        
        # Best validation accuracy for this configuration
        best_val_acc_config = max(val_accuracies) if val_accuracies else 0
        
        # Store results
        results.append({
            'params': params,
            'best_val_acc': best_val_acc_config,
            'final_val_acc': val_accuracies[-1] if val_accuracies else 0,
            'training_time': training_time,
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        })
        
        print(f"Best Val Accuracy: {best_val_acc_config:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Update best model
        if best_val_acc_config > best_val_acc:
            best_val_acc = best_val_acc_config
            best_params = params
            best_model = model
            print("*** New best model! ***")
    
    print("\n" + "="*60)
    print("Grid Search Completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return results, best_model, best_params, best_val_acc

def create_clip_datasets(X_train, y_train, X_val, y_val, X_test, y_test, class_names):
    """Create CLIP datasets for training, validation, and testing"""
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    
    train_dataset = BrainTumorDataset(X_train, y_train, processor, class_names)
    val_dataset = BrainTumorDataset(X_val, y_val, processor, class_names)
    test_dataset = BrainTumorDataset(X_test, y_test, processor, class_names)
    
    print(f"CLIP Dataset sizes:")
    print(f"Train: {len(train_dataset)}")
    print(f"Validation: {len(val_dataset)}")
    print(f"Test: {len(test_dataset)}")
    
    return train_dataset, val_dataset, test_dataset