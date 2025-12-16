#!/usr/bin/env python3
"""
Create augmented training set with MORE augmentation on sparse images (Approach B).

Strategy:
- Empty images (0 rocks): Keep as-is
- Sparse images (1-3 rocks): 5x augmentation
- Medium images (4-10 rocks): 2x augmentation  
- Dense images (11+ rocks): Keep as-is
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm


def count_rocks(label_file):
    """Count number of rocks in a label file."""
    try:
        with open(label_file, 'r') as f:
            return len([l for l in f.readlines() if l.strip()])
    except:
        return 0


def get_augmentation_pipeline():
    """Define safe augmentation pipeline using Albumentations with clipping."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.2,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,  # Important: don't reflect or wrap
            value=0,  # pad with black
            p=0.5
        ),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    ],
    bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.1,   # Discard boxes where <10% remains visible
        min_area=0.0,         # Keep all boxes (even tiny) unless filtered by visibility
        clip=True             # ! THIS IS CRITICAL: clip boxes to [0,1]
    ))


def augment_image_and_label(image_path, label_path, output_img_path, output_label_path, transform):
    """Apply augmentation to image and update labels accordingly."""
    
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"⚠️  Failed to read {image_path}")
        return False
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Read labels
    bboxes = []
    class_labels = []
    
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_labels.append(int(parts[0]))
                    bboxes.append([float(x) for x in parts[1:5]])
    
    # Apply augmentation
    try:
        transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        aug_image = transformed['image']
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']
        
        # Save augmented image
        aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_img_path), aug_image_bgr)
        
        # Save augmented labels
        with open(output_label_path, 'w') as f:
            for bbox, label in zip(aug_bboxes, aug_labels):
                f.write(f"{label} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return True
    except Exception as e:
        print(f"⚠️  Augmentation failed for {image_path.name}: {e}")
        return False


def main():
    import os
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    os.chdir(project_root)
    
    print("=" * 80)
    print("CREATING TARGETED AUGMENTATION DATASET (APPROACH B)")
    print("=" * 80)
    print()
    
    # Check if albumentations is installed
    try:
        import albumentations
        print(f"✅ Using albumentations v{albumentations.__version__}")
    except ImportError:
        print("❌ albumentations not installed")
        print("   Install with: pip install albumentations")
        return
    
    # Paths
    src_images = Path("data/swisstopo_data/images/train")
    src_labels = Path("data/swisstopo_data/labels/train")
    
    dst_base = Path("data/augmented_swisstopo_data")
    dst_images_train = dst_base / "images" / "train"
    dst_labels_train = dst_base / "labels" / "train"
    
    dst_images_train.mkdir(parents=True, exist_ok=True)
    dst_labels_train.mkdir(parents=True, exist_ok=True)
    
    # Get augmentation pipeline
    transform = get_augmentation_pipeline()
    
    # Process each image
    image_files = sorted(src_images.glob("*.tif"))
    
    print(f"Processing {len(image_files)} training images...")
    print()
    
    stats = {
        'empty': {'original': 0, 'augmented': 0},
        'sparse': {'original': 0, 'augmented': 0},
        'medium': {'original': 0, 'augmented': 0},
        'dense': {'original': 0, 'augmented': 0}
    }
    
    for img_file in tqdm(image_files, desc="Augmenting"):
        label_file = src_labels / f"{img_file.stem}.txt"
        
        rock_count = count_rocks(label_file)
        
        # Determine category and augmentation factor
        if rock_count == 0:
            category = 'empty'
            aug_factor = 3  # Augment negatives
        elif rock_count <= 3:
            category = 'sparse'
            aug_factor = 5  # 5x augmentation
        elif rock_count <= 10:
            category = 'medium'
            aug_factor = 2  # 2x augmentation
        else:
            category = 'dense'
            aug_factor = 0  # No augmentation
        
        stats[category]['original'] += 1
        
        # Copy original
        shutil.copy(img_file, dst_images_train / img_file.name)
        if label_file.exists():
            shutil.copy(label_file, dst_labels_train / f"{img_file.stem}.txt")
        else:
            (dst_labels_train / f"{img_file.stem}.txt").touch()
        
        # Create augmentations
        for aug_idx in range(aug_factor):
            aug_img_name = f"{img_file.stem}_aug{aug_idx+1}{img_file.suffix}"
            aug_label_name = f"{img_file.stem}_aug{aug_idx+1}.txt"
            
            success = augment_image_and_label(
                img_file, label_file,
                dst_images_train / aug_img_name,
                dst_labels_train / aug_label_name,
                transform
            )
            
            if success:
                stats[category]['augmented'] += 1
    
    # Copy val and test sets (no augmentation)
    print("\nCopying validation and test sets...")
    
    for split in ['val', 'test']:
        src_img_split = Path(f"data/swisstopo_data/images/{split}")
        src_lbl_split = Path(f"data/swisstopo_data/labels/{split}")
        dst_img_split = dst_base / "images" / split
        dst_lbl_split = dst_base / "labels" / split
        
        dst_img_split.mkdir(parents=True, exist_ok=True)
        dst_lbl_split.mkdir(parents=True, exist_ok=True)
        
        if src_img_split.exists():
            count = 0
            for img in src_img_split.glob("*.tif"):
                shutil.copy(img, dst_img_split / img.name)
                
                label = src_lbl_split / f"{img.stem}.txt"
                if label.exists():
                    shutil.copy(label, dst_lbl_split / label.name)
                else:
                    (dst_lbl_split / f"{img.stem}.txt").touch()
                count += 1
            print(f"  {split}: {count} images copied")
    
    # Create data.yaml
    data_yaml = dst_base / "data.yaml"
    with open(data_yaml, 'w') as f:
        f.write(f"path: {dst_base.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write("nc: 1\n")
        f.write("names: ['rock']\n")
    
    # Summary
    print()
    print("=" * 80)
    print("AUGMENTATION SUMMARY")
    print("=" * 80)
    print()
    
    for category, counts in stats.items():
        total = counts['original'] + counts['augmented']
        print(f"{category.capitalize():10s}: {counts['original']:3d} original + "
              f"{counts['augmented']:3d} augmented = {total:4d} total")
    
    total_original = sum(s['original'] for s in stats.values())
    total_augmented = sum(s['augmented'] for s in stats.values())
    total_all = total_original + total_augmented
    
    print()
    print(f"{'TOTAL':10s}: {total_original:3d} original + "
          f"{total_augmented:3d} augmented = {total_all:4d} total")
    print()
    print(f"Dataset saved to: {dst_base}")
    print(f"Data config: {data_yaml}")
    
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review augmented images to verify quality")
    print("2. Train Approach B model:")
    print(f"   yolo train model=models/baseline_best.pt data={data_yaml} \\")
    print("            epochs=50 batch=8 imgsz=640 device=0 \\")
    print("            project=outputs/training name=approach_b_targeted")
    print()
    print("3. Compare to Approach A (standard augmentation):")
    print("   yolo train model=models/baseline_best.pt data=data/swisstopo_data/data.yaml \\")
    print("            epochs=50 batch=8 imgsz=640 device=0 \\")
    print("            project=outputs/training name=approach_a_uniform")


if __name__ == "__main__":
    main()