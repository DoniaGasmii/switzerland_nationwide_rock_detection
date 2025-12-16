#!/usr/bin/env python3
"""
Analyze the distribution of rocks per image in the training set.
This helps us understand class imbalance and design augmentation strategy.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter


def count_rocks_per_image(labels_dir):
    """Count number of rocks in each label file."""
    rock_counts = {}
    
    for label_file in sorted(labels_dir.glob("*.txt")):
        with open(label_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
            rock_counts[label_file.stem] = len(lines)
    
    return rock_counts


def categorize_images(rock_counts):
    """Categorize images by number of rocks."""
    categories = {
        'empty': [],      # 0 rocks
        'sparse': [],     # 1-3 rocks
        'medium': [],     # 4-10 rocks
        'dense': []       # 11+ rocks
    }
    
    for img_name, count in rock_counts.items():
        if count == 0:
            categories['empty'].append(img_name)
        elif count <= 3:
            categories['sparse'].append(img_name)
        elif count <= 10:
            categories['medium'].append(img_name)
        else:
            categories['dense'].append(img_name)
    
    return categories

def visualize_category_samples(categories, images_dir, labels_dir, output_dir):
    """
    Visualize one sample image from each category with bounding boxes.
    """
    import rasterio
    import cv2
    from matplotlib.patches import Rectangle
    
    # Select one sample from each category
    samples = {}
    for category, images in categories.items():
        if images:
            # Pick the first image from each category
            samples[category] = images[0]
    
    if not samples:
        print("⚠️  No samples to visualize")
        return
    
    # Create visualization
    n_samples = len(samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(6*n_samples, 6))
    
    if n_samples == 1:
        axes = [axes]
    
    for idx, (category, img_name) in enumerate(samples.items()):
        ax = axes[idx]
        
        # Load image
        img_path = images_dir / f"{img_name}.tif"
        
        if not img_path.exists():
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            ax.set_title(f'{category.capitalize()}\n(not found)')
            continue
        
        try:
            # Read image with rasterio (handles georeferenced TIFs)
            with rasterio.open(img_path) as src:
                image = src.read()
                
                # Convert to displayable format
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[0] == 1:
                    image = image[0]
                    image = np.stack([image]*3, axis=-1)  # Convert to RGB
                
                # Normalize for display
                if image.dtype == np.uint16:
                    image = (image / 65535.0 * 255).astype(np.uint8)
                elif image.dtype in [np.float32, np.float64]:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            ax.text(0.5, 0.5, f'Error loading\n{e}', ha='center', va='center')
            ax.set_title(f'{category.capitalize()}\n(error)')
            continue
        
        # Display image
        ax.imshow(image)
        
        # Load and draw bounding boxes
        label_path = labels_dir / f"{img_name}.txt"
        n_boxes = 0
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class cx cy w h
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert normalized coords to pixels
                        img_h, img_w = image.shape[:2]
                        
                        x_center = cx * img_w
                        y_center = cy * img_h
                        box_w = w * img_w
                        box_h = h * img_h
                        
                        x_min = x_center - box_w / 2
                        y_min = y_center - box_h / 2
                        
                        # Draw bounding box
                        rect = Rectangle(
                            (x_min, y_min), box_w, box_h,
                            linewidth=2, edgecolor='red', facecolor='none'
                        )
                        ax.add_patch(rect)
                        n_boxes += 1
        
        # Color coding by category
        colors = {
            'empty': '#ff6b6b',
            'sparse': '#feca57',
            'medium': '#48dbfb',
            'dense': '#1dd1a1'
        }
        
        ax.set_title(
            f'{category.capitalize()}\n{n_boxes} rocks',
            fontsize=12,
            fontweight='bold',
            color=colors.get(category, 'black')
        )
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "category_samples.png", dpi=150, bbox_inches='tight')
    print(f"📊 Saved category samples to {output_dir}/category_samples.png")
    
    # Save individual images with more detail
    print("\nSaving detailed individual samples...")
    
    for category, img_name in samples.items():
        fig_single, ax_single = plt.subplots(1, 1, figsize=(10, 10))
        
        img_path = images_dir / f"{img_name}.tif"
        
        if not img_path.exists():
            continue
        
        try:
            with rasterio.open(img_path) as src:
                image = src.read()
                
                if image.shape[0] == 3:
                    image = np.transpose(image, (1, 2, 0))
                elif image.shape[0] == 1:
                    image = image[0]
                    image = np.stack([image]*3, axis=-1)
                
                if image.dtype == np.uint16:
                    image = (image / 65535.0 * 255).astype(np.uint8)
                elif image.dtype in [np.float32, np.float64]:
                    image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        except Exception as e:
            print(f"⚠️  Error loading {img_path}: {e}")
            plt.close(fig_single)
            continue
        
        ax_single.imshow(image)
        
        label_path = labels_dir / f"{img_name}.txt"
        n_boxes = 0
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        img_h, img_w = image.shape[:2]
                        x_center = cx * img_w
                        y_center = cy * img_h
                        box_w = w * img_w
                        box_h = h * img_h
                        
                        x_min = x_center - box_w / 2
                        y_min = y_center - box_h / 2
                        
                        rect = Rectangle(
                            (x_min, y_min), box_w, box_h,
                            linewidth=3, edgecolor='red', facecolor='none'
                        )
                        ax_single.add_patch(rect)
                        n_boxes += 1
        
        ax_single.set_title(
            f'Sample: {category.capitalize()} Category\n'
            f'Image: {img_name}.tif | Rocks: {n_boxes}',
            fontsize=14,
            fontweight='bold'
        )
        ax_single.axis('off')
        
        plt.tight_layout()
        output_path = output_dir / f"sample_{category}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ Saved {output_path.name}")
        plt.close(fig_single)
        
def main():
    print("=" * 80)
    print("TRAINING SET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    # Use relative path from current directory (no chdir)
    train_labels = Path("data/swisstopo_data/labels/train")
    
    if not train_labels.exists():
        print(f"❌ Training labels not found: {train_labels.absolute()}")
        print(f"\nPlease run from project root:")
        print(f"   cd ~/Switzerland_Expansion/switzerland_nationwide_rock_detection")
        return
    
    # Count rocks per image
    rock_counts = count_rocks_per_image(train_labels)
    
    print(f"Total training images: {len(rock_counts)}")
    print(f"Total rocks: {sum(rock_counts.values())}")
    print(f"Average rocks per image: {np.mean(list(rock_counts.values())):.2f}")
    print(f"Median rocks per image: {np.median(list(rock_counts.values())):.0f}")
    print()
    
    # Categorize
    categories = categorize_images(rock_counts)
    
    print("=" * 80)
    print("DISTRIBUTION BY CATEGORY")
    print("=" * 80)
    print()
    
    for category, images in categories.items():
        pct = len(images) / len(rock_counts) * 100
        print(f"{category.capitalize():10s}: {len(images):4d} images ({pct:5.1f}%)")
    
    print()
    
    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1 = axes[0]
    counts = list(rock_counts.values())
    ax1.hist(counts, bins=range(0, max(counts)+2), edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Number of Rocks per Image')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Rocks per Image')
    ax1.grid(True, alpha=0.3)
    
    # Bar plot by category
    ax2 = axes[1]
    category_counts = [len(categories[cat]) for cat in ['empty', 'sparse', 'medium', 'dense']]
    
    ax2.bar(range(4), category_counts, 
            tick_label=['Empty\n(0)', 'Sparse\n(1-3)', 'Medium\n(4-10)', 'Dense\n(11+)'],
            edgecolor='black', alpha=0.7, color=['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1'])
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Images by Rock Density Category')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path("outputs/data_augmentation/training_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "training_distribution.png", dpi=150, bbox_inches='tight')
    print(f"📊 Saved plot to {output_dir}/training_distribution.png")
    
    # Save detailed stats
    with open(output_dir / "training_distribution.txt", 'w') as f:
        f.write("TRAINING SET DISTRIBUTION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total images: {len(rock_counts)}\n")
        f.write(f"Total rocks: {sum(rock_counts.values())}\n")
        f.write(f"Mean rocks/image: {np.mean(list(rock_counts.values())):.2f}\n")
        f.write(f"Median rocks/image: {np.median(list(rock_counts.values())):.0f}\n")
        f.write(f"Min rocks/image: {min(rock_counts.values())}\n")
        f.write(f"Max rocks/image: {max(rock_counts.values())}\n\n")
        
        f.write("DISTRIBUTION BY CATEGORY:\n")
        f.write("-" * 80 + "\n")
        for category, images in categories.items():
            pct = len(images) / len(rock_counts) * 100
            f.write(f"{category.capitalize():10s}: {len(images):4d} images ({pct:5.1f}%)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("AUGMENTATION RECOMMENDATIONS:\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Approach A (Uniform Augmentation):\n")
        f.write("  - Apply same augmentation to all images\n")
        f.write("  - Standard YOLO augmentations during training\n\n")
        
        sparse_pct = len(categories['sparse']) / len(rock_counts) * 100
        empty_pct = len(categories['empty']) / len(rock_counts) * 100
        
        f.write("Approach B (Targeted Augmentation):\n")
        f.write(f"  - Empty images ({empty_pct:.1f}%): 3x augmentation (reduce false positives)\n")
        f.write(f"  - Sparse images ({sparse_pct:.1f}%): 5x augmentation (hard cases)\n")
        f.write(f"  - Medium images: 2x augmentation\n")
        f.write(f"  - Dense images: No extra augmentation\n")
        f.write("  - Helps balance learning across difficulty levels\n")
    
    print(f"📄 Saved detailed stats to {output_dir}/training_distribution.txt")
    
    print()
    
    # Visualize samples from each category
    print("=" * 80)
    print("VISUALIZING CATEGORY SAMPLES")
    print("=" * 80)
    print()
    
    images_dir = Path("data/swisstopo_data/images/train")
    
    if images_dir.exists():
        visualize_category_samples(categories, images_dir, train_labels, output_dir)
    else:
        print(f"⚠️  Images directory not found: {images_dir}")
    
    print()

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review distribution plot and stats")
    print("2. Choose augmentation approach:")
    print("   A) Uniform: Standard YOLO augmentation on all images")
    print("   B) Targeted: Pre-augment sparse images (run create_augmented_dataset.py)")
    print("3. Train and compare both approaches")


if __name__ == "__main__":
    main()
