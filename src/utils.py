"""
Utility functions for satellite pose estimation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)


def save_config_json(config_dict: Dict, output_path: Path) -> Path:
    """
    Save configuration to JSON file
    
    Args:
        config_dict: Configuration dictionary
        output_path: Output file path
    
    Returns:
        Path to saved file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(config_dict, f, indent=2, default=str)
    
    logger.info(f"Saved config: {output_path}")
    return output_path


def load_config_json(config_path: Path) -> Dict:
    """Load configuration from JSON file"""
    with open(config_path) as f:
        config = json.load(f)
    logger.info(f"Loaded config: {config_path}")
    return config


def create_experiment_dir(base_dir: Path, experiment_name: str) -> Path:
    """
    Create experiment directory with timestamp
    
    Args:
        base_dir: Base directory
        experiment_name: Name of experiment
    
    Returns:
        Path to experiment directory
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = base_dir / f"{experiment_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir


def print_model_summary(model, input_size: tuple = (1, 3, 256, 256)):
    """
    Print model summary (requires torchsummary)
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
    """
    try:
        from torchsummary import summary
        summary(model, input_size=input_size)
    except ImportError:
        logger.warning("torchsummary not installed. Install with: pip install torchsummary")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


def quaternion_to_euler(quaternion: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to Euler angles (roll, pitch, yaw)
    
    Args:
        quaternion: (w, x, y, z) format
    
    Returns:
        Euler angles in radians
    """
    w, x, y, z = quaternion
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1, 1))
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to quaternion
    
    Args:
        euler: Roll, pitch, yaw in radians
    
    Returns:
        Quaternion (w, x, y, z)
    """
    roll, pitch, yaw = euler
    
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion to unit length"""
    return q / np.linalg.norm(q)


def interpolate_quaternions(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) between quaternions
    
    Args:
        q1: First quaternion
        q2: Second quaternion
        t: Interpolation parameter [0, 1]
    
    Returns:
        Interpolated quaternion
    """
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)
    
    dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
    
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    
    if dot > 0.9995:
        # Linear interpolation for very close quaternions
        result = q1 + t * (q2 - q1)
        return normalize_quaternion(result)
    
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    
    q3 = normalize_quaternion(q2 - q1 * dot)
    
    return q1 * np.cos(theta) + q3 * np.sin(theta)
