# Animal Detection with YOLO - Project Guide

This project provides a complete pipeline for processing the Animal Detection dataset and training a YOLO model on Kaggle. It handles directory structuring, label normalization, and environment compatibility issues (specifically OpenCV).

---

## 📋 Table of Contents
1. [Overview](#overview)
2. [Step 1: Data Preprocessing Script](#step-1-data-preprocessing-script)
3. [Step 2: Training Script](#step-2-training-script)
4. [How to Run](#how-to-run)

---

## Overview

The workflow consists of two distinct stages:
1.  **Preprocessing:** Cleans the raw dataset, validates coordinates, normalizes labels, and splits data into Train/Validation sets.
2.  **Training:** Sets up the correct Python environment (downgrades OpenCV) and trains a YOLOv11s model.

---

## Step 1: Data Preprocessing Script

Create a file named `data_preprocessing.py` and paste the following code:

```python
import os
import shutil
import random
import yaml
from tqdm import tqdm
from PIL import Image

# --- Configuration ---
SOURCE_ROOT = '/kaggle/input/animals-detection-images-dataset'
DATASET_PATH = '/kaggle/working/yolo_dataset'
SPLIT_RATIO = 0.8  # 80% train, 20% val

def validate_and_normalize_label(label_path, img_width, img_height, class_id):
    """
    Read a label file, validate it, and normalize coordinates if needed.
    Returns list of properly formatted YOLO annotation lines.
    """
    try:
        valid_lines = []
        if not os.path.exists(label_path):
            return []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Need at least 5 values: class, x_center, y_center, width, height
                if len(parts) < 5:
                    continue
                
                try:
                    # Parse coordinates (skip the old class ID at parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Normalize pixel coordinates if necessary
                    if x_center > 1.0 or y_center > 1.0 or width > 1.0 or height > 1.0:
                        x_center = x_center / img_width
                        y_center = y_center / img_height
                        width = width / img_width
                        height = height / img_height
                    
                    # Validate normalized coordinates
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                            0 < width <= 1 and 0 < height <= 1):
                        continue
                    
                    valid_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                except (ValueError, ZeroDivisionError):
                    continue
        
        return valid_lines
        
    except Exception as e:
        return []

def prepare_detection_dataset(source_root, dest_root, ratio):
    print(f"Preparing dataset from: {source_root}")
    
    # 1. Cleanup Destination
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    
    # 2. Create YOLO Directory Structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dest_root, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dest_root, 'labels', split), exist_ok=True)

    # 3. Identify Classes
    train_source = os.path.join(source_root, 'train')
    if not os.path.exists(train_source):
        raise FileNotFoundError(f"Source folder not found: {train_source}")
        
    classes = sorted([d for d in os.listdir(train_source) 
                      if os.path.isdir(os.path.join(train_source, d))])
    class_to_id = {name: i for i, name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes: {classes}")

    # 4. Process Each Class
    total_processed = 0
    total_valid = 0
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_id = class_to_id[class_name]
        all_valid_pairs = []
        
        # Gather from both train and test folders
        for subset in ['train', 'test']:
            class_dir = os.path.join(source_root, subset, class_name)
            label_dir = os.path.join(class_dir, 'Label')
            
            if not os.path.exists(class_dir) or not os.path.exists(label_dir):
                continue
            
            images = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in images:
                total_processed += 1
                src_img_path = os.path.join(class_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label_path = os.path.join(label_dir, label_file)
                
                if not os.path.exists(src_label_path):
                    continue
                
                try:
                    with Image.open(src_img_path) as img:
                        img_width, img_height = img.size
                except:
                    continue
                
                valid_lines = validate_and_normalize_label(src_label_path, img_width, img_height, class_id)
                if valid_lines:
                    all_valid_pairs.append((src_img_path, valid_lines, img_file))

        # 5. Shuffle and Split
        random.shuffle(all_valid_pairs)
        split_point = int(len(all_valid_pairs) * ratio)
        
        train_pairs = all_valid_pairs[:split_point]
        val_pairs = all_valid_pairs[split_point:]
        
        def process_batch(pairs, split_name):
            nonlocal total_valid
            for src_img, label_lines, filename in pairs:
                dest_img_path = os.path.join(dest_root, 'images', split_name, filename)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(dest_root, 'labels', split_name, label_filename)
                
                shutil.copy2(src_img, dest_img_path)
                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines))
                total_valid += 1

        process_batch(train_pairs, 'train')
        process_batch(val_pairs, 'val')
    
    # 6. Generate data.yaml
    yaml_content = {
        'path': dest_root,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(dest_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"\nPreprocessing Complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total valid dataset size: {total_valid}")
    print(f"Config saved to: {yaml_path}")

if __name__ == "__main__":
    prepare_detection_dataset(SOURCE_ROOT, DATASET_PATH, SPLIT_RATIO)
