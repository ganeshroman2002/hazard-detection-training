# CRITICAL FIX: Downgrade OpenCV before importing it
import subprocess
import sys

print("="*60)
print("Fixing OpenCV version...")
print("="*60)

# Uninstall current OpenCV and install compatible version
subprocess.check_call([
    sys.executable, '-m', 'pip', 'uninstall', 
    'opencv-python', 'opencv-python-headless', '-y', '-q'
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

subprocess.check_call([
    sys.executable, '-m', 'pip', 'install', 
    'opencv-python==4.10.0.84', 
    'opencv-python-headless==4.10.0.84',
    '-q'
], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✓ OpenCV 4.10.0 installed\n")

# Now import everything else
import os
import shutil
import random
import yaml
import cv2
from ultralytics import YOLO
from tqdm import tqdm
from PIL import Image

print(f"Using OpenCV version: {cv2.__version__}\n")

# --- Configuration ---
source_root = '/kaggle/input/animals-detection-images-dataset'
dataset_path = '/kaggle/working/yolo_dataset'
split_ratio = 0.8  # 80% train, 20% val

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
                    
                    # Check if coordinates need normalization (values > 1 indicate pixel coordinates)
                    if x_center > 1.0 or y_center > 1.0 or width > 1.0 or height > 1.0:
                        # Normalize pixel coordinates
                        x_center = x_center / img_width
                        y_center = y_center / img_height
                        width = width / img_width
                        height = height / img_height
                    
                    # Validate normalized coordinates are in valid range
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 < width <= 1 and 0 < height <= 1):
                        continue
                    
                    # Format with new class ID
                    valid_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    
                except (ValueError, ZeroDivisionError):
                    continue
        
        return valid_lines
        
    except Exception as e:
        return []

def prepare_detection_dataset(source_root, dest_root, ratio):
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
    
    print(f"Found {len(classes)} classes")
    print(f"Starting dataset preparation with {ratio*100:.0f}/{(1-ratio)*100:.0f} train/val split...\n")

    # 4. Process Each Class
    total_processed = 0
    total_valid = 0
    skipped_no_labels = 0
    skipped_bad_images = 0
    
    for class_name in tqdm(classes, desc="Processing classes"):
        class_id = class_to_id[class_name]
        all_valid_pairs = []
        
        # Gather from both train and test folders
        for subset in ['train', 'test']:
            class_dir = os.path.join(source_root, subset, class_name)
            label_dir = os.path.join(class_dir, 'Label')
            
            if not os.path.exists(class_dir) or not os.path.exists(label_dir):
                continue
            
            # Find all images
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            
            for img_file in images:
                total_processed += 1
                src_img_path = os.path.join(class_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                src_label_path = os.path.join(label_dir, label_file)
                
                # Check if label exists
                if not os.path.exists(src_label_path):
                    skipped_no_labels += 1
                    continue
                
                # Get image dimensions
                try:
                    with Image.open(src_img_path) as img:
                        img_width, img_height = img.size
                except:
                    skipped_bad_images += 1
                    continue
                
                # Validate labels
                valid_lines = validate_and_normalize_label(src_label_path, img_width, img_height, class_id)
                if not valid_lines:
                    skipped_no_labels += 1
                    continue
                
                all_valid_pairs.append((src_img_path, valid_lines, img_file))

        # 5. Shuffle and Split
        random.shuffle(all_valid_pairs)
        split_point = int(len(all_valid_pairs) * ratio)
        
        train_pairs = all_valid_pairs[:split_point]
        val_pairs = all_valid_pairs[split_point:]
        
        # Process batches
        def process_batch(pairs, split_name):
            nonlocal total_valid
            
            for src_img, label_lines, filename in pairs:
                # Destination paths
                dest_img_path = os.path.join(dest_root, 'images', split_name, filename)
                label_filename = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(dest_root, 'labels', split_name, label_filename)
                
                # Copy image
                shutil.copy2(src_img, dest_img_path)
                
                # Write labels
                with open(label_path, 'w') as f:
                    f.write("\n".join(label_lines))
                
                total_valid += 1

        process_batch(train_pairs, 'train')
        process_batch(val_pairs, 'val')

    print(f"\nDataset preparation complete!")
    print(f"  Total images processed: {total_processed}")
    print(f"  Successfully created: {total_valid}")
    print(f"  Skipped (no valid labels): {skipped_no_labels}")
    print(f"  Skipped (bad images): {skipped_bad_images}")
    
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
    
    print(f"\ndata.yaml created at: {yaml_path}")
    
    # Print sample
    train_images = os.listdir(os.path.join(dest_root, 'images', 'train'))[:3]
    if train_images:
        print(f"\nSample training images: {train_images}")
        sample_label = os.path.splitext(train_images[0])[0] + '.txt'
        sample_label_path = os.path.join(dest_root, 'labels', 'train', sample_label)
        if os.path.exists(sample_label_path):
            with open(sample_label_path, 'r') as f:
                content = f.read()
                print(f"Sample label ({sample_label}): {content[:100]}")

# --- EXECUTION ---
print("="*60)
print("YOLO Dataset Preparation & Training")
print("="*60 + "\n")

# Run the Data Prep
prepare_detection_dataset(source_root, dataset_path, split_ratio)

# Verify before training
train_img_count = len(os.listdir(os.path.join(dataset_path, 'images', 'train')))
train_label_count = len(os.listdir(os.path.join(dataset_path, 'labels', 'train')))
val_img_count = len(os.listdir(os.path.join(dataset_path, 'images', 'val')))
val_label_count = len(os.listdir(os.path.join(dataset_path, 'labels', 'val')))

print(f"\nFinal verification:")
print(f"  Training: {train_img_count} images, {train_label_count} labels")
print(f"  Validation: {val_img_count} images, {val_label_count} labels")

if train_img_count > 0 and train_label_count > 0:
    print("\n" + "="*60)
    print("Starting YOLO Training...")
    print("="*60 + "\n")
    
    model = YOLO('yolo11s.pt')
    
    try:
        results = model.train(
            data=os.path.join(dataset_path, 'data.yaml'),
            epochs=10,
            imgsz=640,
            batch=16,
            name='animal_detector',
            project='/kaggle/working/runs/detect',
            amp=False,
            exist_ok=True,
            workers=0
        )
        print("\n✓ Training completed successfully!")
        print(f"Results saved to: {results.save_dir}")
    except Exception as e:
        print(f"\n❌ Training failed:")
        print(f"{type(e).__name__}: {str(e)[:200]}")
else:
    print("\n⚠️ No valid training data found!")
