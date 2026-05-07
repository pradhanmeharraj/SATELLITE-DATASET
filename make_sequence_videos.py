"""
Generate sequence videos similar to hybrid pose estimation demos.

Creates per-sequence MP4 videos with:
- Frame view + predicted/ground-truth pose text
- Running translation and rotation error plots
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from inference import PosePredictor
from src.evaluation import quaternion_distance, translation_error

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_sequence_dir(data_dir: Path, sequence_name: str) -> Optional[Path]:
    """Resolve sequence directory for common dataset extraction layouts."""
    seq_alias = {
        "seq_a": "sequence_a",
        "seq_b": "sequence_b",
        "seq_c": "sequence_c",
    }
    seq_dir_name = seq_alias.get(sequence_name, sequence_name)

    candidates = [
        data_dir / sequence_name,
        data_dir / seq_dir_name,
        data_dir / "synthetic_cubesat" / sequence_name,
        data_dir / "synthetic_cubesat" / seq_dir_name,
        data_dir / "dataset" / "synthetic_cubesat" / sequence_name,
        data_dir / "dataset" / "synthetic_cubesat" / seq_dir_name,
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "ground_truth").exists():
            return candidate
    return None


def _load_sequence_samples(sequence_dir: Path) -> List[Tuple[Path, Path]]:
    """Pair image and gt json files by stem."""
    image_dir = sequence_dir / "images"
    gt_dir = sequence_dir / "ground_truth"

    image_files = list(image_dir.glob("*.png"))
    if not image_files:
        image_files = list(image_dir.glob("*.jpg"))

    image_files = sorted(image_files)

    pairs: List[Tuple[Path, Path]] = []
    for img_file in image_files:
        gt_file = gt_dir / f"{img_file.stem}.json"
        if gt_file.exists():
            pairs.append((img_file, gt_file))

    return pairs


def _to_plot_points(values: List[float], width: int, height: int, margin: int = 24) -> np.ndarray:
    """Convert 1D values to 2D polyline points in an image coordinate space."""
    if not values:
        return np.empty((0, 2), dtype=np.int32)

    vals = np.array(values, dtype=np.float32)
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-8:
        vmax = vmin + 1.0

    x = np.linspace(margin, width - margin, len(vals))
    y_norm = (vals - vmin) / (vmax - vmin)
    y = (height - margin) - y_norm * (height - 2 * margin)

    points = np.stack([x, y], axis=1).astype(np.int32)
    return points


def _draw_error_panel(
    panel_w: int,
    panel_h: int,
    trans_errors: List[float],
    rot_errors: List[float],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """Draw right-side panel with running error charts."""
    panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (22, 22, 22)

    cv2.putText(panel, "DtF Sequence Metrics", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2)
    cv2.putText(panel, f"Frame: {frame_idx + 1}/{total_frames}", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)

    # Translation plot area
    t_x, t_y, t_w, t_h = 16, 90, panel_w - 32, (panel_h - 130) // 2
    cv2.rectangle(panel, (t_x, t_y), (t_x + t_w, t_y + t_h), (70, 70, 70), 1)
    cv2.putText(panel, "Translation Error (m)", (t_x + 8, t_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    t_points = _to_plot_points(trans_errors, t_w, t_h)
    if len(t_points) > 1:
        t_points[:, 0] += t_x
        t_points[:, 1] += t_y
        cv2.polylines(panel, [t_points], False, (70, 180, 255), 2)

    t_last = trans_errors[-1] if trans_errors else 0.0
    cv2.putText(panel, f"Current: {t_last:.4f} m", (t_x + 8, t_y + t_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 210, 255), 1)

    # Rotation plot area
    r_x, r_y, r_w, r_h = 16, t_y + t_h + 24, panel_w - 32, (panel_h - 130) // 2
    cv2.rectangle(panel, (r_x, r_y), (r_x + r_w, r_y + r_h), (70, 70, 70), 1)
    cv2.putText(panel, "Rotation Error (deg)", (r_x + 8, r_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    r_points = _to_plot_points(rot_errors, r_w, r_h)
    if len(r_points) > 1:
        r_points[:, 0] += r_x
        r_points[:, 1] += r_y
        cv2.polylines(panel, [r_points], False, (120, 255, 120), 2)

    r_last = rot_errors[-1] if rot_errors else 0.0
    cv2.putText(panel, f"Current: {r_last:.3f} deg", (r_x + 8, r_y + r_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 255, 170), 1)

    return panel


def _safe_read_gt(gt_file: Path) -> Dict:
    with open(gt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_sequence_video(
    predictor: PosePredictor,
    data_dir: Path,
    sequence_name: str,
    output_dir: Path,
    fps: int,
    max_frames: Optional[int] = None,
) -> Optional[Path]:
    """Generate a single sequence video and return output path if successful."""
    sequence_dir = _find_sequence_dir(data_dir, sequence_name)
    if sequence_dir is None:
        logger.warning(f"Skipping {sequence_name}: sequence folder not found under {data_dir}")
        return None

    pairs = _load_sequence_samples(sequence_dir)
    if not pairs:
        logger.warning(f"Skipping {sequence_name}: no matched image/json pairs in {sequence_dir}")
        return None

    if max_frames is not None:
        pairs = pairs[:max_frames]

    first_image = cv2.imread(str(pairs[0][0]))
    if first_image is None:
        logger.warning(f"Skipping {sequence_name}: cannot read first image {pairs[0][0]}")
        return None

    h, w = first_image.shape[:2]
    panel_w = max(420, w // 2)
    out_w = w + panel_w
    out_h = h

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{sequence_name}_dtf.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    trans_errors: List[float] = []
    rot_errors: List[float] = []

    total = len(pairs)
    logger.info(f"Rendering {sequence_name}: {total} frames")

    for idx, (img_file, gt_file) in enumerate(pairs):
        frame = cv2.imread(str(img_file))
        if frame is None:
            continue

        gt = _safe_read_gt(gt_file)
        gt_trans = np.array(gt.get("translation", [0.0, 0.0, 0.0]), dtype=np.float32)
        gt_quat = np.array(gt.get("quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32)

        pred = predictor.predict(frame)
        pred_trans = np.array(pred["translation"], dtype=np.float32)
        pred_quat = np.array(pred["quaternion"], dtype=np.float32)

        t_err = translation_error(pred_trans, gt_trans)
        r_err = quaternion_distance(pred_quat, gt_quat)
        trans_errors.append(t_err)
        rot_errors.append(r_err)

        # Left side annotation
        left = frame.copy()
        cv2.rectangle(left, (10, 10), (min(930, w - 10), 180), (0, 0, 0), -1)
        cv2.putText(left, f"{sequence_name.upper()} | DtF", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(left, f"Pred T: [{pred_trans[0]:7.3f}, {pred_trans[1]:7.3f}, {pred_trans[2]:7.3f}]", (20, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (120, 220, 255), 2)
        cv2.putText(left, f"GT   T: [{gt_trans[0]:7.3f}, {gt_trans[1]:7.3f}, {gt_trans[2]:7.3f}]", (20, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 170, 120), 2)
        cv2.putText(left, f"Pred Q: [{pred_quat[0]:6.3f}, {pred_quat[1]:6.3f}, {pred_quat[2]:6.3f}, {pred_quat[3]:6.3f}]", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 255, 170), 1)
        cv2.putText(left, f"GT   Q: [{gt_quat[0]:6.3f}, {gt_quat[1]:6.3f}, {gt_quat[2]:6.3f}, {gt_quat[3]:6.3f}]", (20, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (170, 210, 255), 1)

        cv2.putText(left, f"dT: {t_err:.4f} m", (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 200, 255), 2)
        cv2.putText(left, f"dR: {r_err:.3f} deg", (w - 260, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (130, 255, 130), 2)

        right = _draw_error_panel(panel_w, out_h, trans_errors, rot_errors, idx, total)

        combined = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        combined[:, :w] = left
        combined[:, w:] = right

        writer.write(combined)

    writer.release()
    logger.info(f"Saved video: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sequence MP4 videos for pose estimation results")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Dataset root directory")
    parser.add_argument("--model", type=Path, default=Path("models/best_model.pt"), help="Model checkpoint path")
    parser.add_argument("--output-dir", type=Path, default=Path("results/videos"), help="Output directory for videos")
    parser.add_argument("--sequence", type=str, default="all", choices=["seq_a", "seq_b", "seq_c", "all"], help="Which sequence to render")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick test runs")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {args.model}. Train first with train.py to generate best_model.pt"
        )

    predictor = PosePredictor(model_path=args.model, model_type="dtf", backbone="resnet50", device=args.device)

    if args.sequence == "all":
        sequence_list = ["seq_a", "seq_b", "seq_c"]
    else:
        sequence_list = [args.sequence]

    outputs: List[Path] = []
    for seq in sequence_list:
        out = generate_sequence_video(
            predictor=predictor,
            data_dir=args.data_dir,
            sequence_name=seq,
            output_dir=args.output_dir,
            fps=args.fps,
            max_frames=args.max_frames,
        )
        if out is not None:
            outputs.append(out)

    if outputs:
        logger.info("Video generation complete")
        for out in outputs:
            logger.info(f" - {out}")
    else:
        logger.warning("No videos generated. Check dataset sequence folder structure.")


if __name__ == "__main__":
    main()
