"""
Main training script for satellite pose estimation model
Usage: python train.py --config config.py --data-dir data/
"""

import argparse
import logging
from pathlib import Path
import sys
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR,
    TRAINING_CONFIG, MODEL_CONFIG, LOSS_CONFIG
)
from data_loader import create_data_loaders
from models import create_model
from trainer import Trainer
from evaluation import evaluate_model
from visualization import (
    plot_training_history, plot_error_distribution, plot_trajectory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(
    data_dir: Path = DATA_DIR,
    model_type: str = "dtf",
    backbone: str = "resnet50",
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = "cuda",
    output_dir: Path = MODEL_DIR,
):
    """
    Train pose estimation model
    
    Args:
        data_dir: Path to dataset
        model_type: Type of model ('dtf', 'pnp', 'hybrid')
        backbone: Backbone architecture
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        device: Device to use ('cuda' or 'cpu')
        output_dir: Output directory for checkpoints
    """
    
    logger.info("=" * 60)
    logger.info("SATELLITE POSE ESTIMATION TRAINING")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"
    
    logger.info(f"Device: {device}")
    logger.info(f"Model: {model_type} with {backbone} backbone")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0,
        sequence_type="random",
        augmentation=True,
    )
    
    # Create model
    logger.info("Creating model...")
    model = create_model(
        model_type=model_type,
        backbone=backbone,
        pretrained=MODEL_CONFIG["pretrained"],
        hidden_dims=MODEL_CONFIG["hidden_dims"],
        dropout=MODEL_CONFIG["dropout"],
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        optimizer_name=TRAINING_CONFIG["optimizer"],
        scheduler_name=TRAINING_CONFIG["scheduler"],
        save_dir=output_dir,
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save history
    trainer.save_history()
    
    # Plot training curves
    logger.info("Plotting training history...")
    plot_training_history(
        history,
        output_path=output_dir / "training_history.png"
    )
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Checkpoints saved to: {output_dir}")
    logger.info("=" * 60)
    
    return trainer, model


def evaluate_on_test_set(
    model,
    data_dir: Path = DATA_DIR,
    device: str = "cuda",
    output_dir: Path = RESULTS_DIR,
):
    """
    Evaluate model on test set
    """
    
    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create test loader
    _, test_loader = create_data_loaders(
        data_dir,
        batch_size=32,
        num_workers=0,
        sequence_type="random",
        augmentation=False,
    )
    
    # Evaluate
    logger.info("Evaluating model on test set...")
    metrics = evaluate_model(
        model,
        test_loader,
        device=device,
        output_path=output_dir,
    )
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train satellite pose estimation model"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Path to dataset"
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
        "--batch-size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=TRAINING_CONFIG["device"],
        choices=["cuda", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MODEL_DIR,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Create and use dummy dataset"
    )
    
    args = parser.parse_args()
    
    # Create dummy data if requested
    if args.dummy_data:
        logger.info("Creating dummy dataset...")
        from data_loader import create_dummy_dataset
        create_dummy_dataset(
            args.data_dir / "dataset" / "train",
            num_samples=100
        )
        create_dummy_dataset(
            args.data_dir / "dataset" / "test",
            num_samples=20
        )
    
    # Train model
    trainer, model = train_model(
        data_dir=args.data_dir,
        model_type=args.model_type,
        backbone=args.backbone,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        output_dir=args.output_dir,
    )
    
    # Evaluate if requested
    if args.evaluate:
        evaluate_on_test_set(
            model,
            data_dir=args.data_dir,
            device=args.device,
            output_dir=RESULTS_DIR,
        )


if __name__ == "__main__":
    main()
