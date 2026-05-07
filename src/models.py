"""
Neural network models for satellite pose estimation
Implements Direct-to-Filter (DtF) approach
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class PoseEstimationNetwork(nn.Module):
    """
    Direct-to-Filter (DtF) Pose Estimation Network
    
    Architecture:
    - CNN Backbone (ResNet, EfficientNet, etc.)
    - Feature extraction
    - Two prediction heads:
      * Translation head: (3,) for X, Y, Z
      * Rotation head: (4,) for quaternion
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        hidden_dims: list = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            backbone: Name of backbone architecture
            pretrained: Whether to use pretrained weights
            hidden_dims: List of hidden dimensions for FC layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.hidden_dims = hidden_dims or [512, 256]
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Load backbone
        self._init_backbone(backbone, pretrained)
        
        # Feature dimension from backbone
        if "resnet18" in backbone:
            feature_dim = 512
        elif "resnet" in backbone:
            feature_dim = 2048
        elif "efficientnet" in backbone:
            feature_dim = 1280
        else:
            feature_dim = 2048
        
        # Build feature extraction layers
        self.feature_layers = self._build_fc_layers(
            feature_dim, self.hidden_dims
        )
        
        # Pose prediction heads
        last_hidden = self.hidden_dims[-1]
        
        # Translation head: predict X, Y, Z
        self.translation_head = nn.Sequential(
            nn.Linear(last_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
        )
        
        # Rotation head: predict quaternion (4 values)
        self.rotation_head = nn.Sequential(
            nn.Linear(last_hidden, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 4),
        )
        
        logger.info(f"Initialized {backbone} model with DtF heads")
    
    def _init_backbone(self, backbone: str, pretrained: bool):
        """Initialize backbone network"""
        if "resnet18" in backbone:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif "resnet50" in backbone:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif "resnet101" in backbone:
            weights = models.ResNet101_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet101(weights=weights)
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        elif "efficientnet_b0" in backbone:
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
    
    def _build_fc_layers(self, input_dim: int, hidden_dims: list) -> nn.Sequential:
        """Build fully connected feature extraction layers"""
        layers = []
        
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
            
            prev_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Tuple of (translation, quaternion)
            - translation: (B, 3)
            - quaternion: (B, 4)
        """
        # Extract features from backbone
        features = self.backbone(x)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        
        # Pass through feature layers
        features = self.feature_layers(features)  # (B, hidden_dims[-1])
        
        # Predict translation and rotation
        translation = self.translation_head(features)  # (B, 3)
        quaternion = self.rotation_head(features)  # (B, 4)
        
        # Normalize quaternion to unit length
        quaternion = torch.nn.functional.normalize(quaternion, p=2, dim=1)
        
        return translation, quaternion


class PnPPoseEstimator(nn.Module):
    """
    Perspective-n-Point (PnP) based pose estimator
    Uses neural network to detect fiducial markers, then computes pose via PnP
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_markers: int = 20,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            num_markers: Number of fiducial markers on satellite
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_markers = num_markers
        self.hidden_dims = hidden_dims or [512, 256]
        
        # Initialize backbone
        if "resnet" in backbone:
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Marker detection head (heatmap prediction)
        self.marker_head = nn.Sequential(
            nn.Conv2d(feature_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_markers, kernel_size=1),
        )
        
        logger.info(f"Initialized PnP estimator with {num_markers} markers")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
        
        Returns:
            Marker heatmaps (B, num_markers, H', W')
        """
        features = self.backbone(x)
        heatmaps = self.marker_head(features)
        return heatmaps


class HybridPoseEstimator(nn.Module):
    """
    Hybrid pose estimator combining DtF and PnP approaches
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        num_markers: int = 20,
        hidden_dims: list = None,
        dropout: float = 0.3,
    ):
        """
        Initialize hybrid estimator with both DtF and PnP components
        """
        super().__init__()
        
        # Direct-to-Filter head
        self.dtf_network = PoseEstimationNetwork(
            backbone=backbone,
            pretrained=pretrained,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        # PnP-based head
        self.pnp_network = PnPPoseEstimator(
            backbone=backbone,
            pretrained=pretrained,
            num_markers=num_markers,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        
        logger.info("Initialized Hybrid pose estimator")
    
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            Tuple of (translation, quaternion, heatmaps)
        """
        translation, quaternion = self.dtf_network(x)
        heatmaps = self.pnp_network(x)
        
        return translation, quaternion, heatmaps


def create_model(
    model_type: str = "dtf",
    backbone: str = "resnet50",
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create pose estimation models
    
    Args:
        model_type: 'dtf', 'pnp', or 'hybrid'
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        **kwargs: Additional arguments
    
    Returns:
        Model instance
    """
    if model_type == "dtf":
        return PoseEstimationNetwork(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "pnp":
        return PnPPoseEstimator(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    elif model_type == "hybrid":
        return HybridPoseEstimator(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model
    model = PoseEstimationNetwork(backbone="resnet50", pretrained=False)
    x = torch.randn(4, 3, 256, 256)
    translation, quaternion = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Translation shape: {translation.shape}")
    print(f"Quaternion shape: {quaternion.shape}")
