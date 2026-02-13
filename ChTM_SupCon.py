# -*- coding: utf-8 -*-
'''
Supervised Contrastive Learning Script for ChTM-Based Radar HAR
'''
# Former Author: JoeyBG.
# Improved By: JoeyBG.
# Date: 2026.02.06.
# Platform: Python 3.10, paddlepaddle 3.3.0.
# Affiliation: Beijing Institute of Technology.
# 
# Script Functionality:
#   - This script implements a two-stage training pipeline using Supervised Contrastive Learning.
#   - The first stage performs pretraining using SupConLoss to learn robust feature representations by pulling together representations of the same class.
#   - The second stage performs finetuning using standard Cross Entropy Loss to train a linear classifier on top of the learned features.
#   - The dataset is loaded from a root directory and automatically split into training and validation sets.
#   - It supports strong data augmentation for the contrastive phase and standard augmentation for the evaluation phase.
#   - Key metrics including loss and accuracy are logged, and a confusion matrix is computed during validation.
#
# Key Components:
#   - RadarDataset: Custom Dataset class handling image loading and generating dual views for contrastive learning.
#   - SupConLoss: Implementation of the Supervised Contrastive Loss function.
#   - ConvBackbone: A lightweight Convolutional Neural Network serving as the feature extractor.
#   - ProjectionHead: An MLP used only during the pretraining phase to project features into a lower-dimensional space.
#   - Training Loops: Separate functions for the SupCon pretraining phase and the classification finetuning phase.
#
# Dataset Requirements:
#   - Folder structure must be organized by class names within the root directory.
#   - Images should be in standard formats such as jpg or png.
#
# Usage:
#   - Configure the DATA_ROOT and hyperparameters in the configuration section.
#   - Run the script directly to execute the full pipeline.

'''
Library Importation and Initialization
'''
import os, random, math, json
from pathlib import Path
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from paddle.vision import transforms
from sklearn.metrics import confusion_matrix
import itertools
# === [MODIFIED] Added plotting libraries ===
import matplotlib.pyplot as plt
import seaborn as sns

'''
Parameter Definition
'''
# Configuration of global parameters and hyperparameters.
DATA_ROOT = Path("RW_ChTM_Set")                                                 # Root directory of the dataset.
BATCH_SIZE = 64                                                                 # Batch size for data loaders.
NUM_WORKERS = 4                                                                 # Number of subprocesses to use for data loading.
IMG_SIZE = 64                                                                   # Input image resolution.
EPOCHS_PRETRAIN = 2                                                             # Number of epochs for SupCon pretraining.
EPOCHS_FINETUNE = 2                                                             # Number of epochs for linear or full finetuning.
LEARNING_RATE = 1e-3                                                            # Initial learning rate.
WEIGHT_DECAY = 1e-4                                                             # L2 penalty weight decay.
TEMPERATURE = 0.1                                                               # Temperature parameter for the contrastive loss.
VAL_RATIO = 0.2                                                                 # Ratio of the dataset to be used for validation.
CLASSES = [
    "P1_Gun", "P1_Nogun", "P2_Gun", "P2_Nogun",
    "P3_Gun", "P3_Nogun", "P4_Gun", "P4_Nogun"
]                                                                               # List of class names corresponding to subfolder names.

'''
Dataset Construction and Augmentation
'''
# Define a custom dataset class for loading radar images.
class RadarDataset(Dataset):
    def __init__(self, samples, transform=None, two_views=False):
        """
        Initialization of the dataset.
        samples: list containing path and label tuples
        two_views: boolean flag to return two augmented views when set to True for contrastive learning
        """
        self.samples = samples
        self.transform = transform
        self.two_views = two_views

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = paddle.vision.image_load(path, backend='pil').convert('RGB')
        # Return two augmented versions of the same image if two_views is enabled.
        if self.two_views:
            return self.transform(img), self.transform(img), label
        else:
            return self.transform(img), label

    def __len__(self):
        return len(self.samples)

# Function to traverse the directory and split data into training and validation sets.
def split_dataset():
    samples = []
    # Iterate through each class and collect valid image paths.
    for cidx, cname in enumerate(CLASSES):
        # Added check to ensure directory exists to prevent errors if running without data
        class_dir = DATA_ROOT / cname
        if not class_dir.exists():
            continue
            
        for p in class_dir.glob("*"):
            if p.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"]:
                continue
            samples.append((str(p), cidx))
    
    # Shuffle the samples to ensure random distribution.
    random.shuffle(samples)
    n_val = int(len(samples) * VAL_RATIO)
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]
    return train_samples, val_samples

# Define strong augmentation for the contrastive learning phase.
def build_contrastive_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

# Define standard augmentation or normalization for finetuning and validation.
def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

'''
Model Architecture Definition
'''
# Activation function wrapper for compatibility in Sequential layers.
class HSwish(nn.Layer):
    def forward(self, x):
        return F.hardswish(x)

# Coordinate Attention Module.
class CoordAtt(nn.Layer):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2D((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2D((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2D(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2D(mip)
        self.act = HSwish()
        
        self.conv_h = nn.Conv2D(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2D(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        
        # Decompose the input feature map into two singular coordinate features.
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).transpose([0, 1, 3, 2])

        y = paddle.concat([x_h, x_w], axis=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = paddle.split(y, num_or_sections=[h, w], axis=2)
        x_w = x_w.transpose([0, 1, 3, 2])

        a_h = F.sigmoid(self.conv_h(x_h))
        a_w = F.sigmoid(self.conv_w(x_w))

        out = identity * a_w * a_h
        return out

# Mobile Inverted Residual Bottleneck Block with Coordinate Attention.
class MBConvBlock(nn.Layer):
    def __init__(self, in_dim, out_dim, stride, expand_ratio):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        hidden_dim = int(round(in_dim * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_dim == out_dim

        layers = []
        if expand_ratio != 1:
            # Pointwise convolution (Expansion).
            layers.extend([
                nn.Conv2D(in_dim, hidden_dim, 1, 1, 0, bias_attr=False),
                nn.BatchNorm2D(hidden_dim),
                HSwish()
            ])
        
        layers.extend([
            # Depthwise convolution.
            nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias_attr=False),
            nn.BatchNorm2D(hidden_dim),
            HSwish(),
            # Coordinate Attention mechanism inserted here.
            CoordAtt(hidden_dim),
            # Pointwise convolution (Projection).
            nn.Conv2D(hidden_dim, out_dim, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(out_dim),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Improved Lightweight CNN backbone network for feature extraction.
class ConvBackbone(nn.Layer):
    def __init__(self, out_dim=128):
        super().__init__()
        # Initial stem layer.
        self.stem = nn.Sequential(
            nn.Conv2D(3, 32, 3, 2, 1, bias_attr=False),
            nn.BatchNorm2D(32),
            HSwish()
        )
        
        # Backbone stages using MBConv blocks.
        # Structure aims to be shallow but dense in feature processing.
        self.features = nn.Sequential(
            # Stage 1: Expand and process (32 -> 64)
            MBConvBlock(32, 64, stride=2, expand_ratio=4),
            MBConvBlock(64, 64, stride=1, expand_ratio=4),
            
            # Stage 2: Deepen features (64 -> 128)
            MBConvBlock(64, 128, stride=2, expand_ratio=4),
            MBConvBlock(128, 128, stride=1, expand_ratio=4),
            
            # Stage 3: High-level features (128 -> 256)
            MBConvBlock(128, 256, stride=2, expand_ratio=4),
        )
        
        self.out_dim = out_dim
        # Final projection layer matching the original output structure.
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.normalize(x, axis=1)  # Apply L2 normalization to the features.

# Projection head used to map features to a space where contrastive loss is applied.
class ProjectionHead(nn.Layer):
    def __init__(self, in_dim=128, proj_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        return F.normalize(x, axis=1)

# Simple linear classifier for the downstream classification task.
class Classifier(nn.Layer):
    def __init__(self, feat_dim=128, num_classes=8):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

'''
Loss Function Definition
'''
# Implementation of Supervised Contrastive Loss.
class SupConLoss(nn.Layer):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Compute the Supervised Contrastive Loss.
        features input shape: batch size by number of views by dimension
        labels input shape: batch size
        """
        device = features.place
        bsz, n_views, dim = features.shape
        # Flatten the views dimension for matrix multiplication.
        features = features.reshape([bsz * n_views, dim])

        labels = labels.reshape([bsz, 1])
        # Create a mask to identify positive pairs from the same class.
        mask = (labels == labels.T).astype('float32')
        mask = paddle.repeat_interleave(mask, n_views, axis=0)
        mask = paddle.repeat_interleave(mask, n_views, axis=1)

        # Compute similarity logits.
        anchor_dot_contrast = paddle.matmul(features, features.T) / self.temperature
        logits_max = paddle.max(anchor_dot_contrast, axis=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast to avoid the model cheating by matching the image to itself.
        logits_mask = paddle.ones_like(mask) - paddle.eye(bsz * n_views, dtype='float32')
        mask = mask * logits_mask

        exp_logits = paddle.exp(logits) * logits_mask
        log_prob = logits - paddle.log(exp_logits.sum(axis=1, keepdim=True) + 1e-12)

        # Compute mean of log-likelihood over positive pairs.
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / (mask.sum(axis=1) + 1e-12)
        loss = -mean_log_prob_pos.mean()
        return loss

'''
Training and Validation Loops
'''
# Function to execute one epoch of SupCon pretraining.
def train_supcon(backbone, projection, loader, optimizer, epoch):
    backbone.train(); projection.train()
    supcon = SupConLoss(TEMPERATURE)
    total_loss, steps = 0.0, 0
    total_steps = len(loader)
    
    print(f"\n=== [SupCon Phase] Starting Epoch {epoch}/{EPOCHS_PRETRAIN} ===")
    
    # Iterate through each batch with detailed logging.
    for batch_idx, (view1, view2, labels) in enumerate(loader):
        labels = labels.astype('int64')
        feats1 = backbone(view1)
        feats2 = backbone(view2)
        # Stack features from both views to compute loss.
        feats = paddle.stack([projection(feats1), projection(feats2)], axis=1)
        loss = supcon(feats, labels)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # Update metrics.
        current_loss = loss.item()
        total_loss += current_loss
        steps += 1
        
        # Display detailed metrics for the current batch.
        print(f"[SupCon] Epoch [{epoch}/{EPOCHS_PRETRAIN}] | Batch [{batch_idx+1}/{total_steps}] | Loss: {current_loss:.6f}")

    avg_loss = total_loss / steps
    print(f"--> [SupCon Summary] Epoch {epoch} Completed. Average Loss: {avg_loss:.6f}")

# Function to execute one epoch of classification finetuning.
def finetune(backbone, classifier, loader, optimizer, epoch, freeze_backbone=False):
    if freeze_backbone:
        backbone.eval()
        for p in backbone.parameters():
            p.stop_gradient = True
    else:
        backbone.train()
    classifier.train()

    total_loss, correct, count = 0.0, 0, 0
    total_steps = len(loader)
    
    print(f"\n=== [Finetune Phase] Starting Epoch {epoch}/{EPOCHS_FINETUNE} ===")

    # Iterate through each batch with detailed logging.
    for batch_idx, (imgs, labels) in enumerate(loader):
        labels = labels.astype('int64')
        feats = backbone(imgs)
        logits = classifier(feats)
        loss = F.cross_entropy(logits, labels)

        optimizer.clear_grad()
        loss.backward()
        optimizer.step()

        # Update cumulative metrics.
        current_loss = loss.item()
        total_loss += current_loss
        
        pred = logits.argmax(axis=1)
        batch_correct = (pred == labels).astype('int64').sum().item()
        batch_total = labels.shape[0]
        batch_acc = batch_correct / batch_total
        
        correct += batch_correct
        count += batch_total
        
        # Display detailed metrics for the current batch (Loss and Accuracy).
        print(f"[Finetune] Epoch [{epoch}/{EPOCHS_FINETUNE}] | Batch [{batch_idx+1}/{total_steps}] | Loss: {current_loss:.6f} | Acc: {batch_acc:.4f}")

    avg_loss = total_loss / len(loader)
    avg_acc = correct / count
    print(f"--> [Finetune Summary] Epoch {epoch} Completed. Avg Loss: {avg_loss:.6f} | Avg Acc: {avg_acc:.4f}")

# Function to validate the model and compute the confusion matrix.
@paddle.no_grad()
def validate(backbone, classifier, loader):
    backbone.eval(); classifier.eval()
    correct, count = 0, 0
    all_labels, all_preds = [], []
    for imgs, labels in loader:
        labels = labels.astype('int64')
        feats = backbone(imgs)
        logits = classifier(feats)
        pred = logits.argmax(axis=1)
        correct += (pred == labels).astype('int64').sum().item()
        count += labels.shape[0]
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(pred.numpy().tolist())
    acc = correct / count
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES))))
    return acc, cm

# Plot and save the confusion matrix.
def plot_confusion_matrix(cm, classes, title='Confusion Matrix', save_path='confusion_matrix.png'):
    plt.figure(figsize=(10, 8))
    # Using seaborn for a cleaner heatmap.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    # plt.show() # Uncomment if running in a GUI environment.

def main():
    # Split the dataset into training and validation subsets.
    train_samples, val_samples = split_dataset()
    print(f"Dataset Split -> Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    if len(train_samples) == 0:
        print("Error: No samples found. Please check DATA_ROOT.")
        return

    # Build transformation pipelines.
    contrast_tf = build_contrastive_transform()
    eval_tf = build_eval_transform()

    # Create datasets and dataloaders for different stages.
    train_contrast_ds = RadarDataset(train_samples, transform=contrast_tf, two_views=True)
    train_contrast_loader = DataLoader(train_contrast_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       num_workers=NUM_WORKERS, drop_last=True) # Dataset for contrastive pretraining yielding two views.

    # Dataset for finetuning yielding single standard views.
    train_eval_ds = RadarDataset(train_samples, transform=eval_tf, two_views=False)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, drop_last=False)

    # Dataset for validation.
    val_ds = RadarDataset(val_samples, transform=eval_tf, two_views=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, drop_last=False)

    # Initialize model components.
    backbone = ConvBackbone(out_dim=128)
    projection = ProjectionHead(in_dim=128, proj_dim=128, hidden=256)
    classifier = Classifier(feat_dim=128, num_classes=len(CLASSES))

    # Phase 1: SupCon Pretraining.
    # Optimization focuses on backbone and projection head.
    print("\n" + "="*50)
    print(f"STARTING PHASE 1: SUPERVISED CONTRASTIVE PRETRAINING ({EPOCHS_PRETRAIN} Epochs)")
    print("="*50)
    
    opt_supcon = paddle.optimizer.Adam(parameters=[
        {"params": backbone.parameters()},
        {"params": projection.parameters()}
    ], learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS_PRETRAIN + 1):
        train_supcon(backbone, projection, train_contrast_loader, opt_supcon, epoch)

    # Phase 2: Finetuning.
    print("\n" + "="*50)
    print(f"STARTING PHASE 2: CLASSIFIER FINETUNING ({EPOCHS_FINETUNE} Epochs)")
    print("="*50)
    
    # Option to freeze the backbone is set to False here allowing full fine-tuning.
    freeze_backbone = False
    # Use a lower learning rate for the backbone if freeze_backbone is False.
    opt_finetune = paddle.optimizer.Adam(parameters=[
        {"params": backbone.parameters(), "learning_rate": LEARNING_RATE * (0.3 if freeze_backbone else 1.0)},
        {"params": classifier.parameters()}
    ], learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS_FINETUNE + 1):
        finetune(backbone, classifier, train_eval_loader, opt_finetune, epoch, freeze_backbone=freeze_backbone)
        if epoch % 10 == 0 or epoch == EPOCHS_FINETUNE:
            print(f"\n--- Validation at Epoch {epoch} ---")
            acc, cm = validate(backbone, classifier, val_loader)
            print(f"[Val Result] Epoch {epoch}: Overall Acc={acc:.4f}")
            print("-" * 30)

    # Final validation and reporting.
    print("\n" + "="*50)
    print("TRAINING COMPLETED. FINAL EVALUATION:")
    acc, cm = validate(backbone, classifier, val_loader)
    print(f"Final Val Acc: {acc:.4f}")
    plot_confusion_matrix(cm, CLASSES, save_path='final_confusion_matrix.png')
    print("="*50)

if __name__ == "__main__":
    main()
