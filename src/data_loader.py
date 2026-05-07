"""
Data loading and preprocessing module for Satellite Positioning
"""

import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CubeSatDataset(Dataset):
    """
    PyTorch Dataset for CubeSat pose estimation
    Handles both random samples and sequential data
    """
    
    def __init__(
        self,
        data_path: Path,
        sequence_type: str = "random",  # 'random', 'seq_a', 'seq_b', 'seq_c'
        split: str = "train",  # 'train', 'test'
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True,
        image_size: Tuple[int, int] = (256, 256),
    ):
        """
        Args:
            data_path: Path to dataset root
            sequence_type: Type of data to load
            split: Train or test split
            transform: Image transformations
            normalize: Whether to normalize images
        """
        self.data_path = Path(data_path)
        self.sequence_type = sequence_type
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.image_size = image_size
        
        # Default normalization
        if self.normalize:
            self.norm_transform = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        self.samples = []
        self.ground_truth = []
        self._load_data()

    def _find_existing_dir(
        self,
        candidates: List[Path],
        required_subpath: Optional[Path] = None,
    ) -> Optional[Path]:
        """Return first candidate that exists and (optionally) contains required subpath."""
        for candidate in candidates:
            if not candidate.exists():
                continue
            if required_subpath is not None and not (candidate / required_subpath).exists():
                continue
            if candidate.exists():
                return candidate
        return None

    def _list_images(self, image_dir: Path) -> List[Path]:
        """List images across common extensions in stable order."""
        images: List[Path] = []
        for pattern in ("*.png", "*.jpg", "*.jpeg"):
            images.extend(image_dir.glob(pattern))
        return sorted(images)

    def _load_csv_ground_truth(self, csv_path: Path) -> Dict[str, Dict[str, List[float]]]:
        """Load CSV ground truth keyed by image stem."""
        gt_map: Dict[str, Dict[str, List[float]]] = {}
        if not csv_path.exists():
            return gt_map

        df = pd.read_csv(csv_path)
        required_cols = {"IMG_NUM", "X", "Y", "Z", "Q1", "Q2", "Q3", "W"}
        if not required_cols.issubset(set(df.columns)):
            logger.warning(f"CSV GT missing expected columns: {csv_path}")
            return gt_map

        for _, row in df.iterrows():
            img_name = str(row["IMG_NUM"])
            stem = Path(img_name).stem
            gt_map[stem] = {
                "translation": [float(row["X"]), float(row["Y"]), float(row["Z"])],
                # Dataset uses scalar-last Hamilton convention with columns Q1,Q2,Q3,W
                "quaternion": [float(row["Q1"]), float(row["Q2"]), float(row["Q3"]), float(row["W"])],
            }

        return gt_map
    
    def _load_data(self):
        """Load dataset samples and ground truth"""
        logger.info(f"Loading {self.sequence_type} {self.split} data...")
        
        if self.sequence_type == "random":
            self._load_random_samples()
        else:
            self._load_sequence_data()
        
        logger.info(f"Loaded {len(self.samples)} samples")
    
    def _load_random_samples(self):
        """Load random samples from dataset folder"""
        dataset_path = self._find_existing_dir([
            self.data_path / "dataset",
            self.data_path / "synthetic_cubesat" / "dataset",
            self.data_path / "dataset" / "synthetic_cubesat" / "dataset",
        ], required_subpath=Path(self.split) / "images")

        if dataset_path is None:
            logger.warning(
                "Dataset path not found. Expected one of: "
                f"{self.data_path / 'dataset'}, "
                f"{self.data_path / 'synthetic_cubesat' / 'dataset'}, "
                f"{self.data_path / 'dataset' / 'synthetic_cubesat' / 'dataset'}"
            )
            return
        
        # Load image paths and ground truth
        # This assumes the dataset structure from Kaggle
        image_dir = dataset_path / self.split / "images"
        gt_dir = dataset_path / self.split / "ground_truth"

        if image_dir.exists():
            image_files = self._list_images(image_dir)

            # Path A: per-image JSON labels
            used_json = False
            if gt_dir.exists():
                for img_file in image_files:
                    gt_file = gt_dir / f"{img_file.stem}.json"
                    if gt_file.exists():
                        self.samples.append(str(img_file))
                        with open(gt_file) as f:
                            gt_data = json.load(f)
                            self.ground_truth.append(gt_data)
                        used_json = True

            # Path B: split CSV labels (Kaggle dataset)
            if not used_json:
                csv_candidates = [
                    dataset_path / self.split / f"{self.split}_ground_truth.csv",
                    dataset_path / f"{self.split}_ground_truth.csv",
                ]
                csv_gt_map: Dict[str, Dict[str, List[float]]] = {}
                for csv_path in csv_candidates:
                    csv_gt_map = self._load_csv_ground_truth(csv_path)
                    if csv_gt_map:
                        break

                if csv_gt_map:
                    for img_file in image_files:
                        gt_data = csv_gt_map.get(img_file.stem)
                        if gt_data is not None:
                            self.samples.append(str(img_file))
                            self.ground_truth.append(gt_data)
    
    def _load_sequence_data(self):
        """Load sequential data"""
        seq_map = {
            "seq_a": "sequence_a",
            "seq_b": "sequence_b",
            "seq_c": "sequence_c",
        }
        
        seq_name = seq_map.get(self.sequence_type)
        if not seq_name:
            logger.warning(f"Unknown sequence type: {self.sequence_type}")
            return
        
        seq_path = self._find_existing_dir([
            self.data_path / seq_name,
            self.data_path / "synthetic_cubesat" / seq_name,
            self.data_path / "dataset" / "synthetic_cubesat" / seq_name,
        ], required_subpath=Path("images"))

        if seq_path is None:
            logger.warning(
                "Sequence path not found. Expected one of: "
                f"{self.data_path / seq_name}, "
                f"{self.data_path / 'synthetic_cubesat' / seq_name}, "
                f"{self.data_path / 'dataset' / 'synthetic_cubesat' / seq_name}"
            )
            return
        
        # Load sequence images and ground truth
        image_dir = seq_path / "images"
        gt_dir = seq_path / "ground_truth"

        if image_dir.exists():
            image_files = self._list_images(image_dir)

            # Path A: per-image JSON labels
            used_json = False
            if gt_dir.exists():
                for img_file in image_files:
                    gt_file = gt_dir / f"{img_file.stem}.json"
                    if gt_file.exists():
                        self.samples.append(str(img_file))
                        with open(gt_file) as f:
                            gt_data = json.load(f)
                            self.ground_truth.append(gt_data)
                        used_json = True

            # Path B: single sequence CSV labels
            if not used_json:
                csv_gt_map = self._load_csv_ground_truth(seq_path / "ground_truth.csv")
                if csv_gt_map:
                    for img_file in image_files:
                        gt_data = csv_gt_map.get(img_file.stem)
                        if gt_data is not None:
                            self.samples.append(str(img_file))
                            self.ground_truth.append(gt_data)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample with image and ground truth
        
        Returns:
            Dict with keys: 'image', 'translation', 'quaternion'
        """
        # Load image
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            image = F.to_tensor(image)
        
        # Normalize
        if self.normalize:
            image = self.norm_transform(image)
        
        # Load ground truth
        gt = self.ground_truth[idx]
        translation = torch.tensor(gt["translation"], dtype=torch.float32)
        quaternion = torch.tensor(gt["quaternion"], dtype=torch.float32)
        
        return {
            "image": image,
            "translation": translation,
            "quaternion": quaternion,
            "path": img_path,
        }


def create_data_loaders(
    data_path: Path,
    batch_size: int = 32,
    num_workers: int = 0,
    sequence_type: str = "random",
    augmentation: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation data loaders
    
    Args:
        data_path: Path to dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        sequence_type: Type of sequence to load
        augmentation: Whether to apply augmentation
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Define transforms
    transform = None
    if augmentation:
        transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    
    # Create datasets
    train_dataset = CubeSatDataset(
        data_path,
        sequence_type=sequence_type,
        split="train",
        transform=transform,
        normalize=True,
    )
    
    val_dataset = CubeSatDataset(
        data_path,
        sequence_type=sequence_type,
        split="test",
        transform=None,
        normalize=True,
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    logger.info(f"Created loaders: {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    return train_loader, val_loader


def load_ground_truth(gt_file: Path) -> Dict:
    """Load ground truth from JSON file"""
    with open(gt_file) as f:
        return json.load(f)


def create_dummy_dataset(
    output_path: Path,
    num_samples: int = 100,
    img_size: Tuple[int, int] = (256, 256),
):
    """
    Create a dummy dataset for testing purposes
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    img_dir = output_path / "images"
    gt_dir = output_path / "ground_truth"
    
    img_dir.mkdir(exist_ok=True)
    gt_dir.mkdir(exist_ok=True)
    
    logger.info(f"Creating dummy dataset with {num_samples} samples...")
    
    for i in range(num_samples):
        # Create dummy image
        img = np.random.randint(0, 255, (*img_size, 3), dtype=np.uint8)
        img_path = img_dir / f"sample_{i:06d}.png"
        cv2.imwrite(str(img_path), img)
        
        # Create dummy ground truth
        gt = {
            "translation": [
                float(np.random.uniform(-10, 10)),
                float(np.random.uniform(-10, 10)),
                float(np.random.uniform(100, 200)),
            ],
            "quaternion": [
                float(np.random.randn()),
                float(np.random.randn()),
                float(np.random.randn()),
                float(np.random.randn()),
            ],
        }
        # Normalize quaternion
        q_norm = np.linalg.norm(gt["quaternion"])
        gt["quaternion"] = [q / q_norm for q in gt["quaternion"]]
        
        gt_path = gt_dir / f"sample_{i:06d}.json"
        with open(gt_path, "w") as f:
            json.dump(gt, f)
    
    logger.info(f"Dummy dataset created at {output_path}")
