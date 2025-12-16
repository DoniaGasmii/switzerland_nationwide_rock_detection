#!/usr/bin/env python3
"""
Visualize the effect of each augmentation IN ISOLATION (not mixed).
Useful for understanding failure modes and data bias.
"""

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# Define individual augmentations (same params as your pipeline)
AUGMENTATIONS = {
    "Original": None,
    "HorizontalFlip": A.HorizontalFlip(p=1.0),
    "VerticalFlip": A.VerticalFlip(p=1.0),
    "RandomRotate90": A.RandomRotate90(p=1.0),
    "ShiftScaleRotate": A.ShiftScaleRotate(
        shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=1.0,
        border_mode=cv2.BORDER_CONSTANT, value=0
    ),
    "GaussNoise": A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
    "GaussianBlur": A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    "BrightnessContrast": A.RandomBrightnessContrast(
        brightness_limit=0.2, contrast_limit=0.2, p=1.0
    ),
    "HueSaturation": A.HueSaturationValue(
        hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
    ),
}

def load_yolo_image_and_boxes(img_path, label_path):
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    bboxes, labels = [], []
    if label_path.exists():
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    labels.append(int(float(parts[0])))  # safe parse
                    bboxes.append([float(x) for x in parts[1:5]])
    return image, bboxes, labels

def apply_aug_and_draw(image, bboxes, labels, aug_name, transform):
    if transform is None:
        aug_img, aug_boxes = image.copy(), bboxes
    else:
        try:
            # Use same bbox params as your training
            composed = A.Compose([transform], 
                bbox_params=A.BboxParams(format='yolo', min_visibility=0.1)
            )
            result = composed(image=image, bboxes=bboxes, class_labels=labels)
            aug_img = result['image']
            aug_boxes = result['bboxes']
        except Exception as e:
            print(f"⚠️ Failed {aug_name}: {e}")
            aug_img, aug_boxes = image.copy(), []
    return aug_img, aug_boxes

def draw_boxes(image, bboxes):
    h, w = image.shape[:2]
    out = image.copy()
    for cx, cy, bw, bh in bboxes:
        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return out

def main():
    import os
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    os.chdir(project_root)

    # Pick one image to analyze (change as needed)
    img_name = "2598_1131_3_3"  # or choose dynamically
    img_path = Path(f"data/swisstopo_data/images/train/{img_name}.tif")
    label_path = Path(f"data/swisstopo_data/labels/train/{img_name}.txt")

    image, bboxes, labels = load_yolo_image_and_boxes(img_path, label_path)

    n = len(AUGMENTATIONS)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (name, aug) in enumerate(AUGMENTATIONS.items()):
        aug_img, aug_boxes = apply_aug_and_draw(image, bboxes, labels, name, aug)
        img_with_boxes = draw_boxes(aug_img, aug_boxes)
        axes[idx].imshow(img_with_boxes)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    output_dir = Path("outputs/data_augmentation")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"aug_isolated_{img_name}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved isolated augmentations to {out_path}")

if __name__ == "__main__":
    main()