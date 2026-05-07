"""
Fixed training loop with:
- normalized position targets
- quaternion normalization
- combined loss: pos_loss + beta * rot_loss

Run example:
  python train_fixed.py --csv data/synthetic_cubesat/sequence_a/ground_truth.csv --img-dir data/synthetic_cubesat/sequence_a/images --epochs 2 --batch-size 32 --max-samples 200

"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset

from dataset import CubeSatPoseDataset
from model import PoseNet


def rotation_geodesic_loss(pred_q, target_q, eps=1e-7):
    # pred_q and target_q are (B,4) normalized
    dot = torch.abs(torch.sum(pred_q * target_q, dim=1))
    dot = torch.clamp(dot, -1.0 + eps, 1.0 - eps)
    angle = 2.0 * torch.acos(dot)  # radians
    return angle.mean()


def train_epoch(model, loader, optimizer, device, beta):
    model.train()
    total_loss = 0.0
    total_pos = 0.0
    total_rot = 0.0
    n = 0
    for batch in loader:
        imgs = batch["image"].to(device)
        t_target = batch["translation"].to(device)
        q_target = batch["quaternion"].to(device)

        optimizer.zero_grad()
        t_pred, q_pred = model(imgs)
        pos_loss = F.mse_loss(t_pred, t_target)
        rot_loss = rotation_geodesic_loss(q_pred, q_target)
        loss = pos_loss + beta * rot_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_pos += pos_loss.item() * bs
        total_rot += rot_loss.item() * bs
        n += bs

    return total_loss / n, total_pos / n, total_rot / n


@torch.no_grad()
def validate(model, loader, device, pos_mean, pos_std):
    model.eval()
    pos_errors = []
    rot_errors = []
    for batch in loader:
        imgs = batch["image"].to(device)
        t_target_norm = batch["translation"].to(device)
        q_target = batch["quaternion"].to(device)

        t_pred_norm, q_pred = model(imgs)
        t_pred = t_pred_norm.cpu().numpy() * pos_std + pos_mean
        t_target = t_target_norm.cpu().numpy() * pos_std + pos_mean
        d = np.linalg.norm(t_pred - t_target, axis=1)
        pos_errors.extend(d.tolist())

        dot = (q_pred * q_target).sum(dim=1).abs().cpu().numpy()
        dot = np.clip(dot, -1.0, 1.0)
        angles = 2.0 * np.degrees(np.arccos(dot))
        rot_errors.extend(angles.tolist())

    metrics = {
        "trans_mae": float(np.mean(pos_errors)),
        "trans_std": float(np.std(pos_errors)),
        "rot_mae_deg": float(np.mean(rot_errors)),
        "rot_std_deg": float(np.std(rot_errors)),
    }
    return metrics


def main(args):
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    ds = CubeSatPoseDataset(csv_path=args.csv, image_dir=args.img_dir)
    if args.max_samples and args.max_samples > 0:
        max_n = min(len(ds), args.max_samples)
        ds = Subset(ds, list(range(max_n)))

    val_len = int(len(ds) * args.val_frac)
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PoseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    best_val = float('inf')
    history = {"train_loss":[], "train_pos":[], "train_rot":[], "val_trans_mae":[], "val_rot_mae":[]}

    # retrieve pos stats (if ds is Subset, underlying dataset still has pos_mean)
    if isinstance(ds, Subset):
        base = ds.dataset
    else:
        base = ds
    pos_mean = base.pos_mean
    pos_std = base.pos_std

    for epoch in range(1, args.epochs + 1):
        train_loss, train_pos, train_rot = train_epoch(model, train_loader, optimizer, device, beta=args.beta)
        val_metrics = validate(model, val_loader, device, pos_mean, pos_std)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_pos"].append(train_pos)
        history["train_rot"].append(train_rot)
        history["val_trans_mae"].append(val_metrics["trans_mae"])
        history["val_rot_mae"].append(val_metrics["rot_mae_deg"])

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} pos={train_pos:.4f} rot={train_rot:.4f}  val_trans_mae={val_metrics['trans_mae']:.3f}m val_rot_mae={val_metrics['rot_mae_deg']:.2f}°")

        if val_metrics["trans_mae"] < best_val:
            best_val = val_metrics["trans_mae"]
            save_path = Path(args.save_dir) / "best_model_fixed.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "pos_mean": pos_mean.tolist(),
                "pos_std": pos_std.tolist(),
                "epoch": epoch,
                "val_trans_mae": val_metrics["trans_mae"],
            }, save_path)
            print("Saved best model to", save_path)

    with open(Path(args.save_dir) / "history_fixed.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/synthetic_cubesat/sequence_a/ground_truth.csv")
    parser.add_argument("--img-dir", type=str, default="data/synthetic_cubesat/sequence_a/images")
    parser.add_argument("--save-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=10.0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=0, help="limit samples for quick runs (0 = all)")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
