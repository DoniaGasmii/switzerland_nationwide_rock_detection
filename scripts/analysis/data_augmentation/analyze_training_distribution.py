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


def main():
    import os
    
    # Navigate to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    os.chdir(project_root)
    
    print("=" * 80)
    print("TRAINING SET DISTRIBUTION ANALYSIS")
    print("=" * 80)
    print()
    
    train_labels = Path("data/swisstopo_data/labels/train")
    
    if not train_labels.exists():
        print(f"Training labels not found: {train_labels}")
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
    
    # Box plot
    ax2 = axes[1]
    category_data = [
        [rock_counts[img] for img in categories['empty']],
        [rock_counts[img] for img in categories['sparse']],
        [rock_counts[img] for img in categories['medium']],
        [rock_counts[img] for img in categories['dense']]
    ]
    
    ax2.bar(range(4), [len(cat) for cat in category_data], 
            tick_label=['Empty\n(0)', 'Sparse\n(1-3)', 'Medium\n(4-10)', 'Dense\n(11+)'],
            edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Number of Images')
    ax2.set_title('Images by Rock Density Category')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_dir = Path("outputs/training_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "training_distribution.png", dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_dir}/training_distribution.png")
    
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
        
        sparse_pct = len(categories['sparse']) / len(rock_counts) * 100
        empty_pct = len(categories['empty']) / len(rock_counts) * 100
        
        f.write("Approach A (Uniform Augmentation):\n")
        f.write("  - Apply same augmentation to all images\n")
        f.write("  - Augmentation factor: 3-5x\n\n")
        
        f.write("Approach B (Targeted Augmentation):\n")
        f.write(f"  - Empty images ({empty_pct:.1f}%): 0x (no augmentation)\n")
        f.write(f"  - Sparse images ({sparse_pct:.1f}%): 5-7x augmentation\n")
        f.write(f"  - Medium images: 3x augmentation\n")
        f.write(f"  - Dense images: 1x (no augmentation)\n")
        f.write("  - Helps balance learning across difficulty levels\n")
    
    print(f"Saved detailed stats to {output_dir}/training_distribution.txt")
    
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Review distribution plot and stats")
    print("2. Choose augmentation approach (A or B)")
    print("3. Run create_augmented_configs.py to generate training configs")
    print("4. Train models with both approaches and compare")


if __name__ == "__main__":
    main()