#!/usr/bin/env python3
"""
Visualize duplicate bounding box pairs on TIF images side-by-side.
Shows the same rock detected in two overlapping patches.
"""

import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path


def compute_iou_shapely(box1, box2):
    """Compute IoU between two shapely geometries."""
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union > 0 else 0.0


def find_duplicates(gdf, iou_threshold=0.9):
    """Find all pairs of boxes that are likely duplicates."""
    duplicates = []
    
    for i in range(len(gdf)):
        box1 = gdf.geometry.iloc[i]
        
        for j in range(i + 1, len(gdf)):
            box2 = gdf.geometry.iloc[j]
            iou = compute_iou_shapely(box1, box2)
            
            if iou >= iou_threshold:
                duplicates.append((i, j, iou))
    
    return duplicates


def load_tif_image(tif_path):
    """Load a TIF image for visualization."""
    with rasterio.open(tif_path) as src:
        image = src.read()  # Shape: (bands, height, width)
        transform = src.transform
        
        # Convert to (height, width, bands) for matplotlib
        if image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
        elif image.shape[0] == 1:
            image = image[0]  # Grayscale
        else:
            # Take first 3 bands if more than 3
            image = np.transpose(image[:3], (1, 2, 0))
        
        # Handle data type and scaling
        if image.dtype == np.uint16:
            image = image.astype(np.float32) / 65535.0
        elif image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.dtype in [np.float32, np.float64]:
            # Normalize to [0, 1] range
            img_min = np.nanmin(image)
            img_max = np.nanmax(image)
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min)
            image = np.clip(image, 0, 1)
        
        return image, transform


def global_to_pixel_coords(global_box, transform):
    """
    Convert global coordinates to pixel coordinates using the transform.
    
    Args:
        global_box: Shapely box with global coordinates
        transform: Rasterio affine transform
    
    Returns:
        (xmin, ymin, xmax, ymax) in pixel coordinates
    """
    minx, miny, maxx, maxy = global_box.bounds
    
    # If no geotransform (identity matrix), assume pixel coords = global coords
    if transform == rasterio.Affine.identity():
        return minx, miny, maxx, maxy
    
    # Convert global to pixel using inverse transform
    inv_transform = ~transform
    
    # Top-left corner (minx, maxy because y is flipped in images)
    col_min, row_min = inv_transform * (minx, maxy)
    # Bottom-right corner
    col_max, row_max = inv_transform * (maxx, miny)
    
    # Ensure correct order
    if col_min > col_max:
        col_min, col_max = col_max, col_min
    if row_min > row_max:
        row_min, row_max = row_max, row_min
    
    return col_min, row_min, col_max, row_max


def visualize_duplicate_pair(gdf, idx1, idx2, iou, image_dir, output_path):
    """
    Visualize a duplicate pair side-by-side.
    
    Args:
        gdf: GeoDataFrame with all boxes
        idx1, idx2: Indices of the duplicate pair
        iou: IoU value
        image_dir: Directory containing TIF images
        output_path: Where to save the visualization
    """
    # Get box info
    img1_name = gdf.iloc[idx1]['img']
    img2_name = gdf.iloc[idx2]['img']
    box1 = gdf.geometry.iloc[idx1]
    box2 = gdf.geometry.iloc[idx2]
    
    # Load images
    img1_path = Path(image_dir) / img1_name
    img2_path = Path(image_dir) / img2_name
    
    print(f"  Loading {img1_name}...")
    img1, transform1 = load_tif_image(img1_path)
    print(f"  Loading {img2_name}...")
    img2, transform2 = load_tif_image(img2_path)
    
    # Convert global boxes to pixel coordinates
    x1min, y1min, x1max, y1max = global_to_pixel_coords(box1, transform1)
    x2min, y2min, x2max, y2max = global_to_pixel_coords(box2, transform2)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot image 1
    ax1 = axes[0]
    ax1.imshow(img1)
    
    # Draw the duplicate box in RED
    width1 = x1max - x1min
    height1 = y1max - y1min
    rect1 = Rectangle((x1min, y1min), width1, height1, 
                      linewidth=3, edgecolor='red', facecolor='none')
    ax1.add_patch(rect1)
    
    # Draw all other boxes from this image in GREEN
    for i, row in gdf[gdf['img'] == img1_name].iterrows():
        if i == idx1:
            continue
        other_box = row.geometry
        xmin, ymin, xmax, ymax = global_to_pixel_coords(other_box, transform1)
        width = xmax - xmin
        height = ymax - ymin
        rect = Rectangle((xmin, ymin), width, height, 
                        linewidth=1.5, edgecolor='green', facecolor='none', 
                        alpha=0.7)
        ax1.add_patch(rect)
    
    ax1.set_title(f'{img1_name}\n(Red = Duplicate rock)', fontsize=11)
    ax1.axis('off')
    
    # Plot image 2
    ax2 = axes[1]
    ax2.imshow(img2)
    
    # Draw the duplicate box in RED
    width2 = x2max - x2min
    height2 = y2max - y2min
    rect2 = Rectangle((x2min, y2min), width2, height2, 
                      linewidth=3, edgecolor='red', facecolor='none')
    ax2.add_patch(rect2)
    
    # Draw all other boxes from this image in GREEN
    for i, row in gdf[gdf['img'] == img2_name].iterrows():
        if i == idx2:
            continue
        other_box = row.geometry
        xmin, ymin, xmax, ymax = global_to_pixel_coords(other_box, transform2)
        width = xmax - xmin
        height = ymax - ymin
        rect = Rectangle((xmin, ymin), width, height, 
                        linewidth=1.5, edgecolor='green', facecolor='none', 
                        alpha=0.7)
        ax2.add_patch(rect)
    
    ax2.set_title(f'{img2_name}\n(Red = Duplicate rock)', fontsize=11)
    ax2.axis('off')
    
    # Add overall title
    tile1 = '_'.join(img1_name.split('_')[:2])
    tile2 = '_'.join(img2_name.split('_')[:2])
    same_tile = "Same tile" if tile1 == tile2 else "Different tiles"
    
    minx, miny, maxx, maxy = box1.bounds
    fig.suptitle(f'Duplicate Pair (IoU={iou:.4f}, {same_tile})\n'
                 f'Global coords: ({minx:.1f}, {miny:.1f}) → ({maxx:.1f}, {maxy:.1f})', 
                 fontsize=13, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='none', edgecolor='red', linewidth=3, label='Duplicate rock (same in both images)'),
        Patch(facecolor='none', edgecolor='green', linewidth=1.5, label='Other rocks')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, 
               bbox_to_anchor=(0.5, -0.01), fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved to {output_path}")
    plt.close()


def main():
    print("=" * 80)
    print("VISUALIZING DUPLICATE PAIRS")
    print("=" * 80)
    print()
    
    # Configuration
    shapefile_path = "outputs/shapefiles/test_GT_with_duplicates.shp"
    image_dir = "data/swisstopo_data/images/test"
    output_dir = Path("outputs/duplicate_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load shapefile
    print(f"Loading {shapefile_path}...")
    gdf = gpd.read_file(shapefile_path)
    print(f"Total boxes: {len(gdf)}")
    print()
    
    # Find duplicates
    print("Finding duplicates (IoU ≥ 0.9)...")
    duplicates = find_duplicates(gdf, iou_threshold=0.9)
    print(f"Found {len(duplicates)} duplicate pairs")
    print()
    
    if not duplicates:
        print("No duplicates found!")
        return
    
    # Print summary
    print("Duplicate pairs found:")
    for pair_num, (idx1, idx2, iou) in enumerate(duplicates, 1):
        img1 = gdf.iloc[idx1]['img']
        img2 = gdf.iloc[idx2]['img']
        print(f"  Pair {pair_num}: {img1} ↔ {img2} (IoU={iou:.4f})")
    print()
    
    # Visualize each pair
    for pair_num, (idx1, idx2, iou) in enumerate(duplicates, 1):
        print(f"--- Processing Pair {pair_num}/{len(duplicates)} ---")
        
        output_path = output_dir / f"duplicate_pair_{pair_num:02d}.png"
        
        try:
            visualize_duplicate_pair(gdf, idx1, idx2, iou, image_dir, output_path)
        except Exception as e:
            print(f"  ❌ Error visualizing pair {pair_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print()
    
    print("=" * 80)
    print(f"✅ Done! All visualizations saved to {output_dir}/")
    print(f"   Created {len(duplicates)} visualization files")
    print("=" * 80)


if __name__ == "__main__":
    main()