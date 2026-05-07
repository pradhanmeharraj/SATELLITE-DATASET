# Satellite Positioning Training Module

A comprehensive machine learning framework for **CubeSat pose estimation** using synthetic Unreal Engine data. This module implements both traditional (Perspective-n-Point) and deep learning approaches for accurate satellite positioning.

## 📋 Overview

This project focuses on 6-DOF pose estimation (3D translation + 3D rotation) of CubeSats in Low Earth Orbit (LEO) using:
- **Synthetic Dataset**: 10,000 random samples + 3 sequential sequences (3,600 frames each)
- **Deep Learning**: Direct-to-Filter (DtF) neural network approach
- **Traditional CV**: Perspective-n-Point (PnP) marker detection
- **Hybrid Method**: Combining both approaches for robust estimation

### Key Features
✅ ResNet/EfficientNet backbone architectures  
✅ Quaternion-based rotation representation  
✅ Comprehensive evaluation metrics  
✅ Trajectory visualization  
✅ Training history logging  
✅ Modular, extensible design  

## 📊 Dataset Information

**Source**: [Kaggle - Synthetic CubeSat Dataset](https://www.kaggle.com/datasets/eberhardtkorf/synthetic-cubesat/)

### Camera Specs
- Resolution: 2448 × 2048 pixels
- Focal Length: 12 mm
- Sensor Size: 8.4456 × 7.0656 mm
- Field of View: 38.8° × 32.8°

### CubeSat Model
- Size: 1U (10 cm cube)
- Fiducial Markers: 20 markers for pose refinement
- Texture: Based on real-world CubeSat reference photos

### Data Splits
- **Random Samples**: 10,000 total (8,000 train / 2,000 test)
- **Sequence A**: 3,600 sequential frames
- **Sequence B**: 3,600 sequential frames
- **Sequence C**: 3,600 sequential frames

**Ground Truth**: Translation (X, Y, Z in meters) + Attitude (quaternion)

## 🏗️ Project Structure

```
SATELLITE DATASET/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration file
│   ├── data_loader.py         # Data loading & preprocessing
│   ├── models.py              # Neural network models
│   ├── trainer.py             # Training pipeline
│   ├── evaluation.py          # Evaluation metrics
│   └── visualization.py       # Plotting utilities
├── data/                      # Dataset directory (download here)
├── models/                    # Saved checkpoints
├── results/                   # Evaluation results
├── notebooks/                 # Jupyter notebooks
├── train.py                   # Training script
├── inference.py              # Inference script
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the [Kaggle CubeSat dataset](https://www.kaggle.com/datasets/eberhardtkorf/synthetic-cubesat/) and extract to `data/`

### 3. Train Model

```bash
# Basic training with default settings
python train.py

# Training with custom parameters
python train.py \
    --model-type dtf \
    --backbone resnet50 \
    --batch-size 32 \
    --num-epochs 100 \
    --learning-rate 1e-3 \
    --device cuda \
    --evaluate

# Training with dummy data (for testing)
python train.py --dummy-data
```

### 4. Run Inference

```bash
python inference.py \
    --image path/to/image.png \
    --model models/best_model.pt \
    --output results/visualization.png
```

### 5. Generate Sequence Videos (Like Demo-Style Results)

```bash
# Generate all sequence videos (A, B, C)
python make_sequence_videos.py \
    --data-dir data \
    --model models/best_model.pt \
    --output-dir results/videos \
    --sequence all \
    --fps 30

# Quick test on one sequence
python make_sequence_videos.py \
    --sequence seq_a \
    --max-frames 300 \
    --fps 30
```

Generated files:
- `results/videos/seq_a_dtf.mp4`
- `results/videos/seq_b_dtf.mp4`
- `results/videos/seq_c_dtf.mp4`

## 📚 Module Details

### `data_loader.py`
- `CubeSatDataset`: PyTorch Dataset class
  - Handles random and sequential samples
  - Automatic image normalization
  - Configurable augmentation
- `create_data_loaders()`: Creates train/val loaders
- `create_dummy_dataset()`: Generates synthetic data for testing

### `models.py`
Three model architectures:

1. **PoseEstimationNetwork (DtF)**
   - CNN backbone + FC layers
   - Outputs: translation (3D) + quaternion (4D)
   - Best for real-time inference

2. **PnPPoseEstimator**
   - Marker detection via heatmaps
   - Requires explicit marker extraction
   - More interpretable, requires post-processing

3. **HybridPoseEstimator**
   - Combines DtF and PnP heads
   - Leverages strengths of both approaches

### `trainer.py`
- `QuaternionLoss`: Geodesic distance loss on unit quaternion manifold
- `PoseLoss`: Combined translation + rotation loss
- `Trainer`: Full training loop with:
  - Checkpoint management
  - Learning rate scheduling
  - Early stopping
  - History logging

### `evaluation.py`
Comprehensive metrics:
- **Translation Error**: L2 distance in meters
- **Rotation Error**: Angular distance in degrees
- **Accuracy Rates**: % within thresholds (0.1m, 0.5m, 5°, 10°)
- **Statistical Analysis**: MAE, STD, Min/Max, Median

### `visualization.py`
- `plot_trajectory()`: 3D trajectory comparison
- `plot_error_distribution()`: Error histograms and CDFs
- `plot_training_history()`: Loss curves and learning rate schedule
- `visualize_pose_on_image()`: Annotate images with predictions

## 📈 Training Configuration

Key hyperparameters (in `src/config.py`):

```python
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 100,
    "optimizer": "adam",
    "scheduler": "cosine",  # CosineAnnealingLR
    "device": "cuda",
}

MODEL_CONFIG = {
    "backbone": "resnet50",
    "pretrained": True,
    "hidden_dims": [512, 256],
    "dropout": 0.3,
}

LOSS_CONFIG = {
    "translation_weight": 1.0,
    "rotation_weight": 1.0,
}
```

## 📊 Expected Results

Based on reference paper (Korf et al., 2023):

| Method | Seq A | Seq B | Seq C | Average |
|--------|-------|-------|-------|---------|
| **PnP (MAE)** | 0.00182m | 0.02726m | 0.01752m | 0.01553m |
| **DtF (MAE)** | 0.00189m | 0.00469m | 0.00609m | **0.00422m** |

**Rotation Error** (degrees):
- PnP: ~14.03° average
- **DtF: ~2.07° average** ✓

## 🔧 Advanced Usage

### Custom Data Format

To use your own dataset, implement the structure expected by `CubeSatDataset`:

```
your_dataset/
├── train/
│   ├── images/
│   │   ├── sample_000000.png
│   │   └── ...
│   └── ground_truth/
│       ├── sample_000000.json  # {"translation": [x,y,z], "quaternion": [w,x,y,z]}
│       └── ...
└── test/
    ├── images/
    └── ground_truth/
```

### Hyperparameter Tuning

Modify `src/config.py` for your experiments:

```python
TRAINING_CONFIG["learning_rate"] = 5e-4
TRAINING_CONFIG["batch_size"] = 64
MODEL_CONFIG["backbone"] = "resnet101"
```

### Custom Loss Functions

Extend `trainer.py` with custom losses:

```python
class GeodesicQuatLoss(nn.Module):
    # Implement your custom loss
    pass
```

## 🎯 Next Steps

1. **Download Dataset**: Get the Kaggle CubeSat data
2. **Train Model**: `python train.py --evaluate`
3. **Evaluate Results**: Check `results/evaluation_results.json`
4. **Visualize Trajectories**: View `results/*.png` plots
5. **Fine-tune**: Adjust hyperparameters in `config.py`
6. **Deploy**: Use `inference.py` for real-time predictions

## 📖 References

- **Paper**: "Hybrid pose estimation of space debris with neural networks developed using photorealistic synthetic data" - Eberhardt Korf et al.
- **Dataset**: [Kaggle Synthetic CubeSat Dataset](https://www.kaggle.com/datasets/eberhardtkorf/synthetic-cubesat/)
- **Demo**: [Hybrid Pose Estimation Results](https://eberhardtkorf.github.io/pages/hybrid_pose_estimation/page.html)

## 📄 License

This project follows the dataset's CC BY 4.0 license.

## 💡 Tips & Tricks

- **Fast Training**: Use `--dummy-data` flag first to test pipeline
- **GPU Memory**: Reduce `batch_size` if OOM errors occur
- **Better Results**: Increase `num_epochs` or use larger backbone
- **Debugging**: Enable logging with `--verbose` (future feature)
- **Visualization**: Always run with `--evaluate` to see plots

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- [ ] PnP marker detection implementation
- [ ] Real-world data fine-tuning
- [ ] Video-based temporal models
- [ ] Extended Kalman Filter fusion
- [ ] Mobile deployment (ONNX/TFLite)

---

**Happy Training! 🚀** For questions or issues, refer to the paper or dataset documentation.
