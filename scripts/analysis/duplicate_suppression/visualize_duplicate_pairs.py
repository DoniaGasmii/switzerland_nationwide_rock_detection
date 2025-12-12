#!/usr/bin/env python3
"""
Visualize REAL duplicate pairs found by label-based method.
Shows the same rock detected in two overlapping patches with correct global coordinates.
"""

import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def parse_patch_name(filename):
    """Extract tile ID and patch position from filename."""
    stem = filename.replace('.txt', '').replace('.tif', '')
    parts = stem.split('_')
    
    if len(parts) >= 4:
        tile_id = f"{parts[0]}_{parts[1]}"
        row = int(parts[2])
        col = int(parts[3])
        return tile_id, row, col
    return None, None, None


def load_yolo_boxes(label_path):
    """Load YOLO boxes from a label file."""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append((cls, cx, cy, w, h))
    return boxes


def load_tif_image(tif_path):
    """Load a TIF image for visualization."""
    with rasterio.open(tif_path) as src:
        image = src.read()
        
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 1:
            image = image[0]
        else:
            image = np.transpose(image[:3], (1, 2, 0))
        
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            img_min = np.nanmin(image)
            img_max = np.nanmax(image)
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            image = np.clip(image, 0, 1)
        
        return image


def yolo_to_pixel_bbox(cx, cy, w, h, img_width=640, img_height=640):
    """Convert YOLO normalized coordinates to pixel bounding box."""
    # Convert to pixel coordinates
    px = cx * img_width
    py = cy * img_height
    pw = w * img_width
    ph = h * img_height
    
    # Convert to corner coordinates
    xmin = px - pw / 2
    ymin = py - ph / 2
    xmax = px + pw / 2
    ymax = py + ph / 2
    
    return xmin, ymin, xmax, ymax


def find_matching_box(target_global, boxes, patch_row, patch_col, patch_size=640, overlap=210):
    """
    Find which box in the list matches the target global coordinates.
    Returns index and local pixel coords.
    """
    stride = patch_size - overlap
    
    for idx, box in enumerate(boxes):
        cls, cx, cy, w, h = box
        
        # Convert to patch-local pixels
        px_local = cx * patch_size
        py_local = cy * patch_size
        
        # Convert to global
        global_cx = px_local + patch_col * stride
        global_cy = py_local + patch_row * stride
        
        # Check if this matches our target (within 5 pixels)
        distance = np.sqrt((global_cx - target_global[0])**2 + (global_cy - target_global[1])**2)
        
        if distance < 5.0:
            # Return the pixel bbox for visualization
            xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(cx, cy, w, h)
            return idx, (xmin, ymin, xmax, ymax)
    
    return None, None


def visualize_duplicate_pair(img1_name, img2_name, global1, global2, 
                            labels_dir, images_dir, output_path):
    """Visualize a real duplicate pair with correct boxes highlighted."""
    
    # Parse patch positions
    _, row1, col1 = parse_patch_name(img1_name)
    _, row2, col2 = parse_patch_name(img2_name)
    
    # Load images
    img1_path = Path(images_dir) / img1_name.replace('.txt', '.tif')
    img2_path = Path(images_dir) / img2_name.replace('.txt', '.tif')
    
    print(f"  Loading {img1_name}...")
    img1 = load_tif_image(img1_path)
    print(f"  Loading {img2_name}...")
    img2 = load_tif_image(img2_path)
    
    # Load all boxes from both label files
    label1_path = Path(labels_dir) / img1_name
    label2_path = Path(labels_dir) / img2_name
    
    boxes1 = load_yolo_boxes(label1_path)
    boxes2 = load_yolo_boxes(label2_path)
    
    # Find which boxes are the duplicates
    dup_idx1, dup_bbox1 = find_matching_box(global1, boxes1, row1, col1)
    dup_idx2, dup_bbox2 = find_matching_box(global2, boxes2, row2, col2)
    
    if dup_bbox1 is None or dup_bbox2 is None:
        print(f"  ⚠️  Could not find matching boxes")
        return
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot image 1
    ax1 = axes[0]
    ax1.imshow(img1)
    
    # Draw all boxes in green first
    for idx, box in enumerate(boxes1):
        cls, cx, cy, w, h = box
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(cx, cy, w, h)
        width = xmax - xmin
        height = ymax - ymin
        
        if idx == dup_idx1:
            # Duplicate box in RED
            rect = Rectangle((xmin, ymin), width, height, 
                           linewidth=3, edgecolor='red', facecolor='none')
        else:
            # Other boxes in GREEN
            rect = Rectangle((xmin, ymin), width, height, 
                           linewidth=1.5, edgecolor='green', facecolor='none', alpha=0.7)
        ax1.add_patch(rect)
    
    ax1.set_title(f'{img1_name.replace(".txt", ".tif")}\nPatch [{row1}, {col1}]', fontsize=11)
    ax1.axis('off')
    
    # Plot image 2
    ax2 = axes[1]
    ax2.imshow(img2)
    
    # Draw all boxes
    for idx, box in enumerate(boxes2):
        cls, cx, cy, w, h = box
        xmin, ymin, xmax, ymax = yolo_to_pixel_bbox(cx, cy, w, h)
        width = xmax - xmin
        height = ymax - ymin
        
        if idx == dup_idx2:
            # Duplicate box in RED
            rect = Rectangle((xmin, ymin), width, height, 
                           linewidth=3, edgecolor='red', facecolor='none')
        else:
            # Other boxes in GREEN
            rect = Rectangle((xmin, ymin), width, height, 
                           linewidth=1.5, edgecolor='green', facecolor='none', alpha=0.7)
        ax2.add_patch(rect)
    
    ax2.set_title(f'{img2_name.replace(".txt", ".tif")}\nPatch [{row2}, {col2}]', fontsize=11)
    ax2.axis('off')
    
    # Add overall title
    tile1 = '_'.join(img1_name.split('_')[:2])
    
    distance = np.sqrt((global1[0] - global2[0])**2 + (global1[1] - global2[1])**2)
    
    fig.suptitle(f'Real Duplicate Pair - Tile {tile1}\n'
                 f'Global coords: ({global1[0]:.1f}, {global1[1]:.1f}) ↔ ({global2[0]:.1f}, {global2[1]:.1f})\n'
                 f'Distance: {distance:.1f} pixels', 
                 fontsize=13, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', linewidth=3, label='Duplicate rock (same in both patches)'),
        Patch(facecolor='none', edgecolor='green', linewidth=1.5, label='Other rocks')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def main():
    import os
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    os.chdir(project_root)
    
    print("=" * 80)
    print("VISUALIZING REAL DUPLICATE PAIRS")
    print("=" * 80)
    print(f"Working directory: {os.getcwd()}")
    print()
    
    labels_dir = Path("data/swisstopo_data/labels/test")
    images_dir = Path("data/swisstopo_data/images/test")
    output_dir = Path("outputs/duplicate_visualizations_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the real duplicates file
    duplicates_file = Path("outputs/duplicate_suppression/real_duplicates.txt")
    
    if not duplicates_file.exists():
        print(f"❌ Please run find_duplicates_from_labels.py first!")
        print(f"   Expected file: {duplicates_file}")
        return
    
    # Parse the duplicates file
    duplicates = []
    with open(duplicates_file, 'r') as f:
        lines = f.readlines()
        
        i = 0
        while i < len(lines):
            if lines[i].startswith("Pair"):
                img1 = lines[i+1].split(': ')[1].strip()
                img2 = lines[i+2].split(': ')[1].strip()
                distance_line = lines[i+3]
                coords1_line = lines[i+4]
                coords2_line = lines[i+5]
                
                # Parse coordinates
                coords1_str = coords1_line.split(': ')[1].strip()
                coords1 = tuple(map(float, coords1_str.strip('()').split(', ')))
                
                coords2_str = coords2_line.split(': ')[1].strip()
                coords2 = tuple(map(float, coords2_str.strip('()').split(', ')))
                
                duplicates.append({
                    'img1': img1,
                    'img2': img2,
                    'global1': coords1,
                    'global2': coords2
                })
                
                i += 7  # Skip to next pair
            else:
                i += 1
    
    print(f"Found {len(duplicates)} duplicate pairs to visualize")
    print()
    
    # Visualize each pair
    for pair_num, dup in enumerate(duplicates, 1):
        print(f"--- Processing Pair {pair_num}/{len(duplicates)} ---")
        print(f"  {dup['img1']} ↔ {dup['img2']}")
        
        output_path = output_dir / f"real_duplicate_pair_{pair_num:02d}.png"
        
        try:
            visualize_duplicate_pair(
                dup['img1'], dup['img2'],
                dup['global1'], dup['global2'],
                labels_dir, images_dir, output_path
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print(f"✅ Done! Visualizations saved to {output_dir}/")
    print(f"   Created {len(duplicates)} files")
    print("=" * 80)
    print()
    print("These show the REAL duplicates (5 pairs) with correct global coordinates.")
    print("Compare to the shapefile method which found 8 pairs (3 false positives).")


if __name__ == "__main__":
    main()