"""
Training pipeline for pose estimation models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
import logging
from tqdm import tqdm
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class QuaternionLoss(nn.Module):
    """
    Quaternion distance loss
    Uses geodesic distance on unit quaternion manifold
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute quaternion loss
        
        Args:
            pred: Predicted quaternions (B, 4)
            target: Target quaternions (B, 4)
        
        Returns:
            Loss value
        """
        # Ensure quaternions are normalized
        pred = torch.nn.functional.normalize(pred, p=2, dim=1)
        target = torch.nn.functional.normalize(target, p=2, dim=1)
        
        # Compute quaternion distance
        # d(q1, q2) = 2 * arccos(|<q1, q2>|)
        dot_product = torch.abs(torch.sum(pred * target, dim=1))
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        
        # Geodesic distance
        distance = 2.0 * torch.acos(dot_product)
        
        if self.reduction == "mean":
            return distance.mean()
        elif self.reduction == "sum":
            return distance.sum()
        else:
            return distance


class PoseLoss(nn.Module):
    """
    Combined loss for pose estimation
    Loss = translation_weight * L_trans + rotation_weight * L_rot
    """
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        translation_loss: str = "l2",
        rotation_loss: str = "quaternion",
    ):
        super().__init__()
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        
        # Translation loss
        if translation_loss == "l2":
            self.trans_loss = nn.MSELoss()
        elif translation_loss == "l1":
            self.trans_loss = nn.L1Loss()
        elif translation_loss == "smooth_l1":
            self.trans_loss = nn.SmoothL1Loss()
        else:
            self.trans_loss = nn.MSELoss()
        
        # Rotation loss
        if rotation_loss == "quaternion":
            self.rot_loss = QuaternionLoss()
        else:
            self.rot_loss = QuaternionLoss()
    
    def forward(
        self,
        pred_trans: torch.Tensor,
        pred_quat: torch.Tensor,
        target_trans: torch.Tensor,
        target_quat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined loss
        
        Returns:
            Tuple of (total_loss, trans_loss, rot_loss)
        """
        trans_loss = self.trans_loss(pred_trans, target_trans)
        rot_loss = self.rot_loss(pred_quat, target_quat)
        
        total_loss = (
            self.translation_weight * trans_loss +
            self.rotation_weight * rot_loss
        )
        
        return total_loss, trans_loss, rot_loss


class Trainer:
    """
    Training manager for pose estimation models
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        learning_rate: float = 1e-3,
        num_epochs: int = 100,
        optimizer_name: str = "adam",
        scheduler_name: str = "cosine",
        save_dir: Path = None,
    ):
        """
        Initialize trainer
        """
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_dir = Path(save_dir) if save_dir else Path("checkpoints")
        self.save_dir.mkdir(exist_ok=True)
        
        # Loss function
        self.criterion = PoseLoss(
            translation_weight=1.0,
            rotation_weight=1.0,
        )
        
        # Optimizer
        if optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        if scheduler_name.lower() == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        elif scheduler_name.lower() == "step":
            self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            self.scheduler = None
        
        # Metrics history
        self.history = {
            "train_loss": [],
            "train_trans_loss": [],
            "train_rot_loss": [],
            "val_loss": [],
            "val_trans_loss": [],
            "val_rot_loss": [],
            "learning_rate": [],
        }
        
        logger.info(f"Trainer initialized with {len(train_loader)} train batches")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        
        for batch in pbar:
            images = batch["image"].to(self.device)
            target_trans = batch["translation"].to(self.device)
            target_quat = batch["quaternion"].to(self.device)
            
            # Forward pass
            pred_trans, pred_quat = self.model(images)
            
            # Compute loss
            loss, trans_loss, rot_loss = self.criterion(
                pred_trans, pred_quat, target_trans, target_quat
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            total_trans_loss += trans_loss.item()
            total_rot_loss += rot_loss.item()
            num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "trans_loss": total_trans_loss / num_batches,
            "rot_loss": total_rot_loss / num_batches,
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        total_trans_loss = 0.0
        total_rot_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validating", leave=False)
            
            for batch in pbar:
                images = batch["image"].to(self.device)
                target_trans = batch["translation"].to(self.device)
                target_quat = batch["quaternion"].to(self.device)
                
                # Forward pass
                pred_trans, pred_quat = self.model(images)
                
                # Compute loss
                loss, trans_loss, rot_loss = self.criterion(
                    pred_trans, pred_quat, target_trans, target_quat
                )
                
                total_loss += loss.item()
                total_trans_loss += trans_loss.item()
                total_rot_loss += rot_loss.item()
                num_batches += 1
        
        return {
            "loss": total_loss / num_batches,
            "trans_loss": total_trans_loss / num_batches,
            "rot_loss": total_rot_loss / num_batches,
        }
    
    def train(self) -> Dict:
        """
        Full training loop
        
        Returns:
            Training history
        """
        logger.info(f"Starting training for {self.num_epochs} epochs...")
        
        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Store history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_trans_loss"].append(train_metrics["trans_loss"])
            self.history["train_rot_loss"].append(train_metrics["rot_loss"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_trans_loss"].append(val_metrics["trans_loss"])
            self.history["val_rot_loss"].append(val_metrics["rot_loss"])
            self.history["learning_rate"].append(
                self.optimizer.param_groups[0]["lr"]
            )
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}"
            )
            
            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "history": self.history,
        }
        
        filename = f"checkpoint_epoch_{epoch}.pt"
        if is_best:
            filename = "best_model.pt"
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def save_history(self):
        """Save training history to JSON"""
        history_path = self.save_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Saved history: {history_path}")
    
    def load_checkpoint(self, path: Path):
        """Load model from checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.history = checkpoint["history"]
        logger.info(f"Loaded checkpoint: {path}")
