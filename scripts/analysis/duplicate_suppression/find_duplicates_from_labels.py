#!/usr/bin/env python3
"""
Find duplicates by comparing YOLO labels directly, using patch naming to identify overlaps.
This correctly handles patch-local to tile-global coordinate conversion.
"""

from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_patch_name(filename):
    """
    Extract tile ID and patch position from filename.
    e.g., '2587_1133_0_3.txt' -> tile='2587_1133', row=0, col=3
    """
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


def box_to_global_coords(box, patch_row, patch_col, patch_size=640, overlap=210):
    """
    Convert patch-local normalized coordinates to global tile pixel coordinates.
    
    Args:
        box: (class, cx, cy, w, h) in normalized [0-1] coordinates
        patch_row, patch_col: Position of patch in tile grid
        patch_size: Size of each patch in pixels (default 640)
        overlap: Overlap between patches in pixels (default 210)
    
    Returns:
        (global_cx, global_cy, w_pixels, h_pixels)
    """
    cls, cx, cy, w, h = box
    
    # Convert normalized [0-1] to patch-local pixels
    px_local = cx * patch_size
    py_local = cy * patch_size
    w_pixels = w * patch_size
    h_pixels = h * patch_size
    
    # Calculate stride (distance between patch origins)
    stride = patch_size - overlap
    
    # Convert to global tile coordinates
    global_cx = px_local + patch_col * stride
    global_cy = py_local + patch_row * stride
    
    return global_cx, global_cy, w_pixels, h_pixels


def boxes_are_duplicates(box1_global, box2_global, distance_threshold=15.0):
    """
    Check if two boxes in global coordinates represent the same rock.
    
    Args:
        box1_global, box2_global: (cx, cy, w, h) in global pixel coordinates
        distance_threshold: Maximum distance between centers to consider duplicate (pixels)
    
    Returns:
        (is_duplicate, distance)
    """
    cx1, cy1, w1, h1 = box1_global
    cx2, cy2, w2, h2 = box2_global
    
    # Calculate Euclidean distance between centers
    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    
    return distance < distance_threshold, distance


def main():
    import os
    
    # Navigate to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    os.chdir(project_root)
    
    print("=" * 80)
    print("FINDING REAL DUPLICATES FROM YOLO LABELS")
    print("=" * 80)
    print(f"Working directory: {os.getcwd()}")
    print()
    
    labels_dir = Path("data/swisstopo_data/labels/test")
    
    if not labels_dir.exists():
        print(f"❌ Labels directory not found: {labels_dir}")
        return
    
    # Group labels by tile
    tiles = defaultdict(list)
    
    for label_file in sorted(labels_dir.glob("*.txt")):
        tile_id, row, col = parse_patch_name(label_file.name)
        if tile_id:
            boxes = load_yolo_boxes(label_file)
            tiles[tile_id].append((label_file.name, row, col, boxes))
    
    print(f"Found {len(tiles)} unique tiles")
    print(f"Total patches: {sum(len(patches) for patches in tiles.values())}")
    print()
    
    # Find duplicates within each tile
    total_duplicates = 0
    all_duplicate_pairs = []
    
    for tile_id, patches in sorted(tiles.items()):
        if len(patches) < 2:
            continue  # No overlaps possible with single patch
        
        print(f"Tile {tile_id}: {len(patches)} patches")
        
        duplicates_in_tile = []
        
        # Compare all patch pairs
        for i in range(len(patches)):
            img1, row1, col1, boxes1 = patches[i]
            
            for j in range(i + 1, len(patches)):
                img2, row2, col2, boxes2 = patches[j]
                
                # Compare all box pairs
                for box_idx1, box1 in enumerate(boxes1):
                    # Convert box1 to global coordinates
                    global1 = box_to_global_coords(box1, row1, col1)
                    
                    for box_idx2, box2 in enumerate(boxes2):
                        # Convert box2 to global coordinates
                        global2 = box_to_global_coords(box2, row2, col2)
                        
                        # Check if they're duplicates
                        is_dup, distance = boxes_are_duplicates(global1, global2)
                        
                        if is_dup:
                            duplicates_in_tile.append({
                                'img1': img1,
                                'img2': img2,
                                'box1': box1,
                                'box2': box2,
                                'global1': global1,
                                'global2': global2,
                                'distance': distance
                            })
        
        if duplicates_in_tile:
            print(f"  Found {len(duplicates_in_tile)} duplicate pair(s):")
            for dup in duplicates_in_tile:
                print(f"    {dup['img1']} ↔ {dup['img2']}")
                print(f"      Distance: {dup['distance']:.2f} pixels")
                print(f"      Global coords 1: ({dup['global1'][0]:.1f}, {dup['global1'][1]:.1f})")
                print(f"      Global coords 2: ({dup['global2'][0]:.1f}, {dup['global2'][1]:.1f})")
            total_duplicates += len(duplicates_in_tile)
            all_duplicate_pairs.extend(duplicates_in_tile)
        else:
            print(f"  No duplicates found")
        
        print()
    
    print("=" * 80)
    print(f"SUMMARY")
    print("=" * 80)
    print(f"Total real duplicates found: {total_duplicates}")
    print()
    
    # Save results to file
    output_file = Path("outputs/duplicate_suppression/real_duplicates.txt")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("REAL DUPLICATES FOUND (Label-Based Method)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, dup in enumerate(all_duplicate_pairs, 1):
            f.write(f"Pair {i}:\n")
            f.write(f"  Image 1: {dup['img1']}\n")
            f.write(f"  Image 2: {dup['img2']}\n")
            f.write(f"  Distance: {dup['distance']:.2f} pixels\n")
            f.write(f"  Global coords 1: ({dup['global1'][0]:.1f}, {dup['global1'][1]:.1f})\n")
            f.write(f"  Global coords 2: ({dup['global2'][0]:.1f}, {dup['global2'][1]:.1f})\n\n")
    
    print(f"✅ Detailed results saved to {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()