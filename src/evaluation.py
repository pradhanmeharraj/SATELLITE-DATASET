"""
Evaluation and metrics module for pose estimation
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, List
import json
import logging
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix
    
    Args:
        q: Quaternion (w, x, y, z) shape (4,)
    
    Returns:
        Rotation matrix (3, 3)
    """
    q = q / np.linalg.norm(q)
    
    w, x, y, z = q
    
    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2],
    ])
    
    return R


def quaternion_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Compute angular distance between two quaternions in degrees
    
    Args:
        q1: First quaternion (4,)
        q2: Second quaternion (4,)
    
    Returns:
        Angular distance in degrees
    """
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Ensure proper sign convention
    if np.dot(q1, q2) < 0:
        q2 = -q2
    
    # Angular distance
    dot_product = np.clip(np.abs(np.dot(q1, q2)), -1.0, 1.0)
    angle_rad = 2.0 * np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    
    return float(angle_deg)


def translation_error(
    pred_trans: np.ndarray,
    target_trans: np.ndarray,
) -> float:
    """
    Compute translation error (Euclidean distance in meters)
    
    Args:
        pred_trans: Predicted translation (3,)
        target_trans: Target translation (3,)
    
    Returns:
        Translation error in meters
    """
    return float(np.linalg.norm(pred_trans - target_trans))


class PoseEvaluator:
    """
    Evaluate pose estimation model on test set
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.predictions = []
        self.ground_truth = []
        self.translation_errors = []
        self.rotation_errors = []
    
    def evaluate_batch(
        self,
        pred_trans: torch.Tensor,
        pred_quat: torch.Tensor,
        target_trans: torch.Tensor,
        target_quat: torch.Tensor,
    ):
        """
        Evaluate a batch of predictions
        
        Args:
            pred_trans: Predicted translations (B, 3)
            pred_quat: Predicted quaternions (B, 4)
            target_trans: Target translations (B, 3)
            target_quat: Target quaternions (B, 4)
        """
        pred_trans = pred_trans.cpu().numpy()
        pred_quat = pred_quat.cpu().numpy()
        target_trans = target_trans.cpu().numpy()
        target_quat = target_quat.cpu().numpy()
        
        batch_size = pred_trans.shape[0]
        
        for i in range(batch_size):
            # Compute translation error
            trans_error = translation_error(
                pred_trans[i], target_trans[i]
            )
            
            # Compute rotation error
            rot_error = quaternion_distance(
                pred_quat[i], target_quat[i]
            )
            
            self.translation_errors.append(trans_error)
            self.rotation_errors.append(rot_error)
            
            self.predictions.append({
                "translation": pred_trans[i].tolist(),
                "quaternion": pred_quat[i].tolist(),
            })
            
            self.ground_truth.append({
                "translation": target_trans[i].tolist(),
                "quaternion": target_quat[i].tolist(),
            })
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get evaluation metrics
        
        Returns:
            Dictionary of metrics
        """
        if not self.translation_errors:
            return {}
        
        trans_errors = np.array(self.translation_errors)
        rot_errors = np.array(self.rotation_errors)
        
        metrics = {
            # Translation metrics (meters)
            "trans_mae": float(np.mean(trans_errors)),
            "trans_std": float(np.std(trans_errors)),
            "trans_min": float(np.min(trans_errors)),
            "trans_max": float(np.max(trans_errors)),
            "trans_median": float(np.median(trans_errors)),
            
            # Rotation metrics (degrees)
            "rot_mae": float(np.mean(rot_errors)),
            "rot_std": float(np.std(rot_errors)),
            "rot_min": float(np.min(rot_errors)),
            "rot_max": float(np.max(rot_errors)),
            "rot_median": float(np.median(rot_errors)),
            
            # Percentage within thresholds
            "trans_within_0_1m": float(np.sum(trans_errors <= 0.1) / len(trans_errors) * 100),
            "trans_within_0_5m": float(np.sum(trans_errors <= 0.5) / len(trans_errors) * 100),
            "rot_within_5deg": float(np.sum(rot_errors <= 5.0) / len(rot_errors) * 100),
            "rot_within_10deg": float(np.sum(rot_errors <= 10.0) / len(rot_errors) * 100),
        }
        
        return metrics
    
    def print_metrics(self):
        """Print evaluation metrics"""
        metrics = self.get_metrics()
        
        logger.info("\n" + "="*60)
        logger.info("POSE ESTIMATION EVALUATION RESULTS")
        logger.info("="*60)
        
        logger.info("\nTranslation Errors (meters):")
        logger.info(f"  MAE: {metrics.get('trans_mae', 0):.6f}")
        logger.info(f"  STD: {metrics.get('trans_std', 0):.6f}")
        logger.info(f"  Min: {metrics.get('trans_min', 0):.6f}")
        logger.info(f"  Max: {metrics.get('trans_max', 0):.6f}")
        logger.info(f"  Median: {metrics.get('trans_median', 0):.6f}")
        
        logger.info("\nRotation Errors (degrees):")
        logger.info(f"  MAE: {metrics.get('rot_mae', 0):.4f}")
        logger.info(f"  STD: {metrics.get('rot_std', 0):.4f}")
        logger.info(f"  Min: {metrics.get('rot_min', 0):.4f}")
        logger.info(f"  Max: {metrics.get('rot_max', 0):.4f}")
        logger.info(f"  Median: {metrics.get('rot_median', 0):.4f}")
        
        logger.info("\nAccuracy Metrics:")
        logger.info(f"  Translation within 0.1m: {metrics.get('trans_within_0_1m', 0):.2f}%")
        logger.info(f"  Translation within 0.5m: {metrics.get('trans_within_0_5m', 0):.2f}%")
        logger.info(f"  Rotation within 5°: {metrics.get('rot_within_5deg', 0):.2f}%")
        logger.info(f"  Rotation within 10°: {metrics.get('rot_within_10deg', 0):.2f}%")
        
        logger.info("="*60 + "\n")
        
        return metrics
    
    def save_results(self, output_path: Path):
        """Save evaluation results to JSON"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics = self.get_metrics()
        
        results = {
            "metrics": metrics,
            "num_samples": len(self.predictions),
            "predictions": self.predictions,
            "ground_truth": self.ground_truth,
        }
        
        results_file = output_path / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results: {results_file}")
        
        return results


def evaluate_model(
    model: torch.nn.Module,
    test_loader,
    device: str = "cuda",
    output_path: Path = None,
) -> Dict[str, float]:
    """
    Evaluate model on test set
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        output_path: Path to save results
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    evaluator = PoseEvaluator(device=device)
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            target_trans = batch["translation"]
            target_quat = batch["quaternion"]
            
            pred_trans, pred_quat = model(images)
            
            evaluator.evaluate_batch(
                pred_trans, pred_quat, target_trans, target_quat
            )
    
    metrics = evaluator.print_metrics()
    
    if output_path:
        evaluator.save_results(output_path)
    
    return metrics
