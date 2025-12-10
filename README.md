# YOLO Animal Detection - Training Pipeline

A complete pipeline for preparing and training a YOLO model for animal detection using the Animals Detection Images Dataset.

## 📋 Overview

This project provides a robust solution for:
- Converting animal detection datasets to YOLO format
- Validating and normalizing bounding box annotations
- Training a YOLOv11 model for multi-class animal detection

## 🚀 Quick Start

### Prerequisites

```bash
pip install ultralytics opencv-python pillow pyyaml tqdm
```

### Dataset Structure

Expected input format:
```
animals-detection-images-dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── Label/
│   │       ├── image1.txt
│   │       └── image2.txt
│   └── class2/
│       └── ...
└── test/
    └── ...
```

## 📦 Data Processing

### Features

The data processing module handles:

✅ **Automatic label validation and normalization**
- Converts pixel coordinates to normalized YOLO format (0-1 range)
- Validates bounding box coordinates
- Filters invalid annotations

✅ **Smart dataset splitting**
- Configurable train/validation split (default: 80/20)
- Randomized shuffling for better generalization
- Preserves class balance

✅ **Robust error handling**
- Skips corrupted images
- Handles missing labels gracefully
- Detailed processing statistics

### Usage

```python
from yolo_data_prep import prepare_detection_dataset

# Configure paths
source_root = '/path/to/animals-detection-images-dataset'
dataset_path = '/path/to/output/yolo_dataset'
split_ratio = 0.8  # 80% train, 20% validation

# Process dataset
prepare_detection_dataset(source_root, dataset_path, split_ratio)
```

### Output Structure

```
yolo_dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── val/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── img1.txt
│   │   └── img2.txt
│   └── val/
│       └── ...
└── data.yaml
```

### Label Format

Each `.txt` file contains annotations in YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized (0.0 - 1.0).

## 🎯 Model Training

### Training Configuration

```python
from ultralytics import YOLO

# Initialize model
model = YOLO('yolo11s.pt')  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x

# Train
results = model.train(
    data='yolo_dataset/data.yaml',
    epochs=10,              # Number of training epochs
    imgsz=640,             # Input image size
    batch=16,              # Batch size
    name='animal_detector', # Experiment name
    project='runs/detect',  # Save directory
    amp=False,             # Automatic Mixed Precision
    exist_ok=True,         # Overwrite existing experiment
    workers=0              # Data loading workers (0 for Windows)
)
```

### Hyperparameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `epochs` | Training iterations | 10 | 50-300 |
| `imgsz` | Input image size | 640 | 320-1280 |
| `batch` | Batch size | 16 | 8-32 (depends on GPU) |
| `lr0` | Initial learning rate | 0.01 | 0.001-0.01 |
| `patience` | Early stopping patience | 100 | 50-100 |

### Model Variants

| Model | Size | Speed | mAP | Use Case |
|-------|------|-------|-----|----------|
| YOLOv11n | Nano | Fastest | Lower | Mobile/Edge devices |
| YOLOv11s | Small | Fast | Good | General purpose |
| YOLOv11m | Medium | Medium | Better | Balanced |
| YOLOv11l | Large | Slow | Best | High accuracy |
| YOLOv11x | XLarge | Slowest | Highest | Research/Competition |

## 📊 Training Outputs

After training completes, find results in:

```
runs/detect/animal_detector/
├── weights/
│   ├── best.pt         # Best model checkpoint
│   └── last.pt         # Last epoch checkpoint
├── results.png         # Training metrics plot
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
└── val_batch0_pred.jpg # Sample predictions
```

## 🔍 Model Evaluation

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('runs/detect/animal_detector/weights/best.pt')

# Validate
metrics = model.val()

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 🎬 Inference

### Single Image

```python
model = YOLO('runs/detect/animal_detector/weights/best.pt')

# Predict
results = model('path/to/image.jpg')

# Display
results[0].show()

# Save
results[0].save('output.jpg')
```

### Batch Prediction

```python
# Process directory
results = model('path/to/images/')

for r in results:
    r.save(filename=f'prediction_{r.path.stem}.jpg')
```

### Video

```python
results = model('path/to/video.mp4', stream=True)

for r in results:
    r.show()  # Display frame
```

## ⚙️ Advanced Configuration

### Custom Training Arguments

```python
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    
    # Augmentation
    hsv_h=0.015,      # HSV-Hue augmentation
    hsv_s=0.7,        # HSV-Saturation
    hsv_v=0.4,        # HSV-Value
    degrees=0.0,      # Rotation
    translate=0.1,    # Translation
    scale=0.5,        # Scale
    flipud=0.0,       # Flip up-down
    fliplr=0.5,       # Flip left-right
    mosaic=1.0,       # Mosaic augmentation
    
    # Optimization
    optimizer='AdamW', # SGD, Adam, AdamW
    lr0=0.01,         # Initial learning rate
    lrf=0.01,         # Final learning rate
    momentum=0.937,   # SGD momentum
    weight_decay=0.0005,
    
    # Loss weights
    box=7.5,          # Box loss gain
    cls=0.5,          # Class loss gain
    dfl=1.5,          # DFL loss gain
    
    # Other
    cos_lr=True,      # Cosine learning rate scheduler
    close_mosaic=10,  # Disable mosaic last N epochs
    amp=True,         # Automatic Mixed Precision
)
```

## 🐛 Troubleshooting

### OpenCV Version Issues

The script automatically fixes OpenCV compatibility:

```python
# Handled automatically in the script
subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', 
    'opencv-python==4.10.0.84'
])
```

### Common Issues

**Out of Memory (OOM)**
- Reduce `batch` size
- Reduce `imgsz`
- Use a smaller model variant

**Slow Training**
- Enable `amp=True` for mixed precision
- Reduce `workers` if CPU bottleneck
- Use GPU if available

**Poor Performance**
- Increase `epochs`
- Augment data more aggressively
- Use a larger model
- Check label quality

## 📈 Performance Tips

1. **Data Quality**: Ensure accurate bounding boxes
2. **Class Balance**: Balance training samples per class
3. **Augmentation**: Use appropriate data augmentation
4. **Transfer Learning**: Start with pretrained weights
5. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes

## 📝 Statistics & Monitoring

The script provides detailed statistics:

```
Dataset preparation complete!
  Total images processed: 5000
  Successfully created: 4800
  Skipped (no valid labels): 150
  Skipped (bad images): 50

Training: 3840 images, 3840 labels
Validation: 960 images, 960 labels
```

## 🔗 Resources

- [Ultralytics YOLOv11 Docs](https://docs.ultralytics.com/)
- [YOLO Format Specification](https://docs.ultralytics.com/datasets/detect/)
- [Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)

## 📄 License

This pipeline is provided as-is. Please check the license of the YOLO model and dataset you're using.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Note**: For Kaggle environments, the script includes automatic OpenCV version management to ensure compatibility.
