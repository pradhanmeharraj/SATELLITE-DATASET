"""
Minimal dataset for CubeSat pose estimation.
Reads CSV with image names and target poses, returns normalized tensors.
"""
import csv
from pathlib import Path
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class CubeSatPoseDataset(Dataset):
    def __init__(self, csv_path, image_dir, pos_mean=None, pos_std=None, transform=None):
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.samples = []
        self._load_csv()

        positions = np.array([s["translation"] for s in self.samples], dtype=np.float32)
        if pos_mean is None:
            self.pos_mean = positions.mean(axis=0)
        else:
            self.pos_mean = np.array(pos_mean, dtype=np.float32)
        if pos_std is None:
            self.pos_std = positions.std(axis=0) + 1e-8
        else:
            self.pos_std = np.array(pos_std, dtype=np.float32)

    def _load_csv(self):
        with open(self.csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                img_name = r.get("IMG_NUM") or r.get("image") or r.get("img") or r.get("filename")
                if not img_name:
                    continue
                img_path = self.image_dir / img_name
                try:
                    tx = float(r.get("X", 0.0))
                    ty = float(r.get("Y", 0.0))
                    tz = float(r.get("Z", 0.0))
                except Exception:
                    tx = ty = tz = 0.0

                qx = r.get("Q1") or r.get("qx") or r.get("x")
                qy = r.get("Q2") or r.get("qy") or r.get("y")
                qz = r.get("Q3") or r.get("qz") or r.get("z")
                qw = r.get("W") or r.get("qw") or r.get("w")
                try:
                    qx = float(qx); qy = float(qy); qz = float(qz); qw = float(qw)
                except Exception:
                    qx = qy = qz = 0.0
                    qw = 1.0
                quat = np.array([qw, qx, qy, qz], dtype=np.float32)
                self.samples.append({
                    "img": img_path,
                    "translation": np.array([tx, ty, tz], dtype=np.float32),
                    "quaternion": quat,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = cv2.imread(str(s["img"]))
        if img is None:
            img = np.zeros((256,256,3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img).permute(2,0,1).float()
        mean = torch.tensor([0.485,0.456,0.406])[:,None,None]
        std  = torch.tensor([0.229,0.224,0.225])[:,None,None]
        img_tensor = (img_tensor - mean) / std

        t = s["translation"].astype(np.float32)
        t_norm = (t - self.pos_mean) / self.pos_std
        t_tensor = torch.from_numpy(t_norm).float()

        q = s["quaternion"].astype(np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)
        q_tensor = torch.from_numpy(q).float()

        return {"image": img_tensor, "translation": t_tensor, "quaternion": q_tensor}
