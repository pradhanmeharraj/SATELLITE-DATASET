"""
Inference script for pose estimation
Usage: python inference.py --image image.png --model model.pt
"""

import argparse
import json
import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import create_model
from config import CAMERA_CONFIG
from visualization import visualize_pose_on_image
from evaluation import quaternion_to_rotation_matrix

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PosePredictor:
    """
    Pose estimation inference module
    """
    
    def __init__(
        self,
        model_path: Path,
        model_type: str = "dtf",
        backbone: str = "resnet50",
        device: str = "cuda",
    ):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model
            backbone: Backbone architecture
            device: Device to use
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable, falling back to CPU")
            device = "cpu"
        self.device = device
        self.model_type = model_type
        
        # Load model
        self.model = create_model(
            model_type=model_type,
            backbone=backbone,
            pretrained=False,
        ).to(device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=device)
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        logger.info(f"Loaded model from {model_path}")
    
    def predict(self, image: np.ndarray) -> dict:
        """
        Predict pose from image
        
        Args:
            image: Input image (H, W, 3) in BGR format
        
        Returns:
            Dictionary with 'translation' and 'quaternion'
        """
        # Preprocess image
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Resize to expected size
        img_resized = cv2.resize(image, (256, 256))
        
        # Convert to tensor (H, W, 3) -> (3, H, W)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1)
        
        # Normalize
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        img_tensor = torch.nn.functional.normalize(img_tensor, p=2, dim=1)
        
        # Predict
        with torch.no_grad():
            if self.model_type in ["dtf", "pnp"]:
                if self.model_type == "dtf":
                    translation, quaternion = self.model(img_tensor)
                else:
                    heatmaps = self.model(img_tensor)
                    # For PnP, would extract markers and compute pose
                    translation = torch.zeros(1, 3).to(self.device)
                    quaternion = torch.zeros(1, 4).to(self.device)
            else:  # hybrid
                translation, quaternion, heatmaps = self.model(img_tensor)
        
        # Convert to numpy
        translation = translation.cpu().numpy()[0]
        quaternion = quaternion.cpu().numpy()[0]
        
        return {
            "translation": translation,
            "quaternion": quaternion,
        }


def format_analysis(result: dict, image_path: Path, model_type: str) -> str:
    """
    Build a short human-readable analysis summary for a single image.
    """
    translation = np.asarray(result["translation"], dtype=np.float32)
    quaternion = np.asarray(result["quaternion"], dtype=np.float32)
    translation_norm = float(np.linalg.norm(translation))
    quaternion_norm = float(np.linalg.norm(quaternion))

    return (
        f"Image analysis for {image_path.name}\n"
        f"Model type: {model_type}\n"
        f"Predicted translation (x, y, z): ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})\n"
        f"Predicted position magnitude: {translation_norm:.4f} m\n"
        f"Predicted quaternion (w, x, y, z): ({quaternion[0]:.4f}, {quaternion[1]:.4f}, {quaternion[2]:.4f}, {quaternion[3]:.4f})\n"
        f"Quaternion norm: {quaternion_norm:.4f}\n"
        f"Note: this is pose estimation, so the analysis reports the model's predicted satellite pose for the image."
    )
    
    def predict_batch(self, images: np.ndarray) -> dict:
        """
        Predict poses for batch of images
        
        Args:
            images: Stack of images (B, H, W, 3)
        
        Returns:
            Dictionary with 'translation' and 'quaternion' arrays
        """
        # Preprocess
        if images.dtype != np.float32:
            images = images.astype(np.float32) / 255.0
        
        # Resize
        batch_size = images.shape[0]
        img_batch = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        for i in range(batch_size):
            img_batch[i] = cv2.resize(images[i], (256, 256))
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_batch).permute(0, 3, 1, 2)
        img_tensor = img_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            if self.model_type == "dtf":
                translation, quaternion = self.model(img_tensor)
            else:
                raise NotImplementedError(f"Batch prediction not implemented for {self.model_type}")
        
        # Convert to numpy
        translation = translation.cpu().numpy()
        quaternion = quaternion.cpu().numpy()
        
        return {
            "translation": translation,
            "quaternion": quaternion,
        }


def infer_image(
    image_path: Path,
    model_path: Path,
    model_type: str = "dtf",
    backbone: str = "resnet50",
    device: str = "cuda",
    output_path: Path = None,
    result_path: Path = None,
    visualize: bool = True,
):
    """
    Run inference on a single image
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not load image: {image_path}")
        return
    
    # Create predictor
    predictor = PosePredictor(
        model_path=model_path,
        model_type=model_type,
        backbone=backbone,
        device=device,
    )
    
    # Predict
    logger.info("Running inference...")
    result = predictor.predict(image)
    
    logger.info(f"Translation: {result['translation']}")
    logger.info(f"Quaternion: {result['quaternion']}")

    analysis_text = format_analysis(result, image_path, model_type)
    logger.info("\n" + analysis_text)

    if result_path:
        result_path = Path(result_path)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "image": str(image_path),
            "model": str(model_path),
            "model_type": model_type,
            "translation": result["translation"].tolist(),
            "quaternion": result["quaternion"].tolist(),
            "analysis": analysis_text,
        }
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info(f"Saved analysis JSON: {result_path}")
    
    # Visualize if requested
    if visualize and output_path:
        img_vis = visualize_pose_on_image(
            image,
            result["translation"],
            result["quaternion"],
            output_path=output_path,
        )
        logger.info(f"Saved visualization: {output_path}")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Run pose estimation inference")
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="dtf",
        choices=["dtf", "pnp", "hybrid"],
        help="Type of model"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone architecture"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for visualization"
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        help="Optional path to save the analysis as JSON"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip visualization"
    )
    
    args = parser.parse_args()
    
    infer_image(
        args.image,
        args.model,
        model_type=args.model_type,
        backbone=args.backbone,
        device=args.device,
        output_path=args.output,
        result_path=args.result_json,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
