"""
Visualization utilities for pose estimation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


def plot_trajectory(
    ground_truth: List[Dict],
    predictions: List[Dict],
    output_path: Optional[Path] = None,
    title: str = "3D Trajectory Comparison",
):
    """
    Plot 3D trajectory of predicted vs ground truth positions
    
    Args:
        ground_truth: List of GT poses with 'translation' key
        predictions: List of predicted poses with 'translation' key
        output_path: Where to save figure
        title: Plot title
    """
    gt_trans = np.array([g["translation"] for g in ground_truth])
    pred_trans = np.array([p["translation"] for p in predictions])
    
    fig = plt.figure(figsize=(14, 5))
    
    # 3D plot
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], 
             "b-", linewidth=2, label="Ground Truth")
    ax1.plot(pred_trans[:, 0], pred_trans[:, 1], pred_trans[:, 2], 
             "r--", linewidth=2, label="Predicted")
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title("3D Trajectory")
    ax1.legend()
    
    # XY plane
    ax2 = fig.add_subplot(132)
    ax2.plot(gt_trans[:, 0], gt_trans[:, 1], "b-", linewidth=2, label="Ground Truth")
    ax2.plot(pred_trans[:, 0], pred_trans[:, 1], "r--", linewidth=2, label="Predicted")
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("XY Plane")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # XZ plane
    ax3 = fig.add_subplot(133)
    ax3.plot(gt_trans[:, 0], gt_trans[:, 2], "b-", linewidth=2, label="Ground Truth")
    ax3.plot(pred_trans[:, 0], pred_trans[:, 2], "r--", linewidth=2, label="Predicted")
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Z (m)")
    ax3.set_title("XZ Plane")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved trajectory plot: {output_path}")
    
    return fig


def plot_error_distribution(
    trans_errors: List[float],
    rot_errors: List[float],
    output_path: Optional[Path] = None,
):
    """
    Plot distribution of translation and rotation errors
    
    Args:
        trans_errors: List of translation errors (meters)
        rot_errors: List of rotation errors (degrees)
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Translation error histogram
    axes[0, 0].hist(trans_errors, bins=50, color="blue", alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Translation Error (m)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Translation Error Distribution")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(trans_errors), color="r", linestyle="--", label=f"Mean: {np.mean(trans_errors):.4f}")
    axes[0, 0].legend()
    
    # Rotation error histogram
    axes[0, 1].hist(rot_errors, bins=50, color="green", alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Rotation Error (degrees)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Rotation Error Distribution")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(np.mean(rot_errors), color="r", linestyle="--", label=f"Mean: {np.mean(rot_errors):.2f}°")
    axes[0, 1].legend()
    
    # CDF curves
    sorted_trans = np.sort(trans_errors)
    sorted_rot = np.sort(rot_errors)
    cdf = np.arange(1, len(sorted_trans) + 1) / len(sorted_trans)
    
    axes[1, 0].plot(sorted_trans, cdf, linewidth=2, color="blue")
    axes[1, 0].set_xlabel("Translation Error (m)")
    axes[1, 0].set_ylabel("Cumulative Probability")
    axes[1, 0].set_title("Translation Error CDF")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(np.median(trans_errors), color="r", linestyle="--", label=f"Median: {np.median(trans_errors):.4f}")
    axes[1, 0].legend()
    
    axes[1, 1].plot(sorted_rot, cdf, linewidth=2, color="green")
    axes[1, 1].set_xlabel("Rotation Error (degrees)")
    axes[1, 1].set_ylabel("Cumulative Probability")
    axes[1, 1].set_title("Rotation Error CDF")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(np.median(rot_errors), color="r", linestyle="--", label=f"Median: {np.median(rot_errors):.2f}°")
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved error distribution plot: {output_path}")
    
    return fig


def plot_training_history(
    history: Dict,
    output_path: Optional[Path] = None,
):
    """
    Plot training and validation loss curves
    
    Args:
        history: Dictionary with training history
        output_path: Where to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Total loss
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Total Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Translation loss
    axes[0, 1].plot(epochs, history["train_trans_loss"], "b-", label="Train Trans Loss")
    axes[0, 1].plot(epochs, history["val_trans_loss"], "r-", label="Val Trans Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("Translation Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rotation loss
    axes[1, 0].plot(epochs, history["train_rot_loss"], "b-", label="Train Rot Loss")
    axes[1, 0].plot(epochs, history["val_rot_loss"], "r-", label="Val Rot Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title("Rotation Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if "learning_rate" in history:
        axes[1, 1].semilogy(epochs, history["learning_rate"], "g-")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate (log scale)")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True, alpha=0.3, which="both")
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved training history plot: {output_path}")
    
    return fig


def visualize_pose_on_image(
    image: np.ndarray,
    pred_trans: np.ndarray,
    pred_quat: np.ndarray,
    gt_trans: np.ndarray = None,
    gt_quat: np.ndarray = None,
    camera_matrix: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """
    Visualize pose predictions on image
    
    Args:
        image: Image array
        pred_trans: Predicted translation
        pred_quat: Predicted quaternion
        gt_trans: Ground truth translation
        gt_quat: Ground truth quaternion
        camera_matrix: Camera intrinsic matrix
        output_path: Where to save figure
    
    Returns:
        Annotated image
    """
    img_copy = image.copy()
    
    # Add text annotations
    cv2.putText(
        img_copy,
        f"Pred Trans: ({pred_trans[0]:.2f}, {pred_trans[1]:.2f}, {pred_trans[2]:.2f})",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    
    if gt_trans is not None:
        cv2.putText(
            img_copy,
            f"GT Trans: ({gt_trans[0]:.2f}, {gt_trans[1]:.2f}, {gt_trans[2]:.2f})",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_copy)
        logger.info(f"Saved annotated image: {output_path}")
    
    return img_copy


def create_comparison_report(
    results_dir: Path,
    sequence_type: str = "all",
) -> Path:
    """
    Create a comprehensive comparison report
    
    Args:
        results_dir: Results directory
        sequence_type: Type of sequence ('all', 'random', 'seq_a', 'seq_b', 'seq_c')
    
    Returns:
        Path to generated report
    """
    results_dir = Path(results_dir)
    report_path = results_dir / "comparison_report.html"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Satellite Pose Estimation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            .metric {{ margin: 10px 0; padding: 10px; background-color: #f0f0f0; }}
            img {{ max-width: 100%; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
        </style>
    </head>
    <body>
        <h1>Satellite Pose Estimation Report</h1>
        <p>Generated for: {sequence_type}</p>
        
        <h2>Results Summary</h2>
        <div class="metric">
            <p>This report contains the evaluation results for satellite pose estimation.</p>
        </div>
        
        <h2>Evaluation Plots</h2>
        <h3>Trajectory Comparison</h3>
        <img src="trajectory.png" alt="Trajectory">
        
        <h3>Error Distribution</h3>
        <img src="error_distribution.png" alt="Error Distribution">
        
    </body>
    </html>
    """
    
    with open(report_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Generated report: {report_path}")
    return report_path
