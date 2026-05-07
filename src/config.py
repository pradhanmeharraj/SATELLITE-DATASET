"""
Configuration file for Satellite Positioning Training Module
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Dataset configuration
DATASET_CONFIG = {
    "random_samples": 10000,
    "train_ratio": 0.8,
    "test_ratio": 0.2,
    "sequences": ["A", "B", "C"],
    "samples_per_sequence": 3600,
}

# Camera configuration (from Kaggle dataset)
CAMERA_CONFIG = {
    "resolution": (2448, 2048),
    "focal_length": 12,  # mm
    "sensor_size": (8.4456, 7.0656),  # mm
    "fov": (38.8, 32.8),  # degrees
}

# CubeSat model configuration
CUBESAT_CONFIG = {
    "size": 0.1,  # 1U CubeSat: 10cm
    "fiducial_markers": 20,
    "markers_per_side": (2, 4),  # min to max
}

# Training configuration
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "optimizer": "adam",
    "scheduler": "cosine",
    "val_split": 0.1,
    "device": "cuda",  # or "cpu"
}

# Model architecture configuration
MODEL_CONFIG = {
    "backbone": "resnet50",  # resnet18, resnet50, efficientnet_b0, etc.
    "pretrained": False,
    "hidden_dims": [512, 256],
    "dropout": 0.3,
    "use_batch_norm": True,
}

# Loss function weights
LOSS_CONFIG = {
    "translation_weight": 1.0,
    "rotation_weight": 1.0,
    "translation_loss": "l2",  # mse, l1, smooth_l1
    "rotation_loss": "quaternion",  # quaternion, geodesic
}

# Evaluation metrics
EVAL_CONFIG = {
    "position_threshold": 0.1,  # meters
    "angle_threshold": 5.0,  # degrees
    "compute_auc": True,
}

# Data augmentation
AUGMENTATION_CONFIG = {
    "enable": True,
    "random_brightness": 0.2,
    "random_contrast": 0.2,
    "random_rotation": 10,  # degrees
    "random_scale": 0.1,
}

# Inference configuration
INFERENCE_CONFIG = {
    "use_tta": False,  # Test-Time Augmentation
    "tta_iterations": 5,
    "confidence_threshold": 0.5,
}
