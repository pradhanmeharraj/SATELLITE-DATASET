"""
Satellite Positioning Training Module
Package initialization
"""

from .config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR,
    TRAINING_CONFIG, MODEL_CONFIG, CAMERA_CONFIG,
)
from .data_loader import CubeSatDataset, create_data_loaders
from .models import PoseEstimationNetwork, PnPPoseEstimator, HybridPoseEstimator, create_model
from .trainer import Trainer, QuaternionLoss, PoseLoss
from .evaluation import PoseEvaluator, evaluate_model, quaternion_distance
from .visualization import (
    plot_trajectory, plot_error_distribution, plot_training_history,
    visualize_pose_on_image
)

__version__ = "1.0.0"
__author__ = "Satellite Positioning Team"

__all__ = [
    # Config
    "DATA_DIR", "MODEL_DIR", "RESULTS_DIR",
    "TRAINING_CONFIG", "MODEL_CONFIG", "CAMERA_CONFIG",
    
    # Data
    "CubeSatDataset", "create_data_loaders",
    
    # Models
    "PoseEstimationNetwork", "PnPPoseEstimator", "HybridPoseEstimator", "create_model",
    
    # Training
    "Trainer", "QuaternionLoss", "PoseLoss",
    
    # Evaluation
    "PoseEvaluator", "evaluate_model", "quaternion_distance",
    
    # Visualization
    "plot_trajectory", "plot_error_distribution", "plot_training_history",
    "visualize_pose_on_image",
]
