#!/usr/bin/env python3
"""
List duplicate pairs for manual inspection in QGIS.
"""

import geopandas as gpd


def compute_iou_shapely(box1, box2):
    """Compute IoU between two shapely geometries."""
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union > 0 else 0.0


def find_and_list_duplicates(gdf, iou_threshold=0.9):
    """
    Find all pairs of boxes that are likely duplicates.
    """
    duplicates = []
    
    for i in range(len(gdf)):
        box1 = gdf.geometry.iloc[i]
        
        for j in range(i + 1, len(gdf)):
            box2 = gdf.geometry.iloc[j]
            iou = compute_iou_shapely(box1, box2)
            
            if iou >= iou_threshold:
                duplicates.append((i, j, iou))
    
    return duplicates


def main():
    print("=" * 80)
    print("DUPLICATE PAIRS FOR QGIS INSPECTION")
    print("=" * 80)
    print()
    
    # Load shapefile with duplicates
    gdf = gpd.read_file("../outputs/shapefiles/test_GT_with_duplicates.shp")
    
    print(f"Total boxes: {len(gdf)}")
    print()
    
    # Find duplicate pairs
    duplicates = find_and_list_duplicates(gdf, iou_threshold=0.9)
    
    print(f"Found {len(duplicates)} duplicate pairs")
    print()
    
    if not duplicates:
        print("No duplicates found!")
        return
    
    # List all duplicate pairs
    print("=" * 80)
    print("DUPLICATE PAIRS:")
    print("=" * 80)
    print()
    
    for pair_num, (i, j, iou) in enumerate(duplicates, 1):
        img1 = gdf.iloc[i]['img']
        img2 = gdf.iloc[j]['img']
        
        box1 = gdf.geometry.iloc[i]
        box2 = gdf.geometry.iloc[j]
        
        minx1, miny1, maxx1, maxy1 = box1.bounds
        minx2, miny2, maxx2, maxy2 = box2.bounds
        
        # Extract tile IDs
        tile1 = '_'.join(img1.split('_')[:2])
        tile2 = '_'.join(img2.split('_')[:2])
        same_tile = "✓" if tile1 == tile2 else "✗"
        
        # Calculate center distance
        center1_x = (minx1 + maxx1) / 2
        center1_y = (miny1 + maxy1) / 2
        center2_x = (minx2 + maxx2) / 2
        center2_y = (miny2 + maxy2) / 2
        distance = ((center1_x - center2_x)**2 + (center1_y - center2_y)**2)**0.5
        
        print(f"Pair {pair_num}: IoU = {iou:.4f} | Same tile: {same_tile}")
        print(f"  Image 1: {img1}")
        print(f"  Image 2: {img2}")
        print(f"  Box 1 coords: ({minx1:.1f}, {miny1:.1f}) → ({maxx1:.1f}, {maxy1:.1f})")
        print(f"  Box 2 coords: ({minx2:.1f}, {miny2:.1f}) → ({maxx2:.1f}, {maxy2:.1f})")
        print(f"  Center distance: {distance:.1f}m")
        print()
    
    # Summary by tile
    print("=" * 80)
    print("SUMMARY BY TILE:")
    print("=" * 80)
    print()
    
    tile_pairs = {}
    for i, j, iou in duplicates:
        img1 = gdf.iloc[i]['img']
        img2 = gdf.iloc[j]['img']
        tile1 = '_'.join(img1.split('_')[:2])
        tile2 = '_'.join(img2.split('_')[:2])
        
        if tile1 == tile2:
            if tile1 not in tile_pairs:
                tile_pairs[tile1] = []
            tile_pairs[tile1].append((img1, img2))
    
    for tile, pairs in sorted(tile_pairs.items()):
        print(f"Tile {tile}: {len(pairs)} duplicate pair(s)")
        for img1, img2 in pairs:
            print(f"  - {img1} ↔ {img2}")
        print()
    
    # Create a text file for easy reference
    with open('../outputs/shapefiles/duplicate_pairs.txt', 'w') as f:
        f.write("DUPLICATE PAIRS - FOR QGIS INSPECTION\n")
        f.write("=" * 80 + "\n\n")
        
        for pair_num, (i, j, iou) in enumerate(duplicates, 1):
            img1 = gdf.iloc[i]['img']
            img2 = gdf.iloc[j]['img']
            
            f.write(f"Pair {pair_num}:\n")
            f.write(f"  Image 1: {img1}\n")
            f.write(f"  Image 2: {img2}\n")
            f.write(f"  IoU: {iou:.4f}\n\n")
    
    print("=" * 80)
    print(f"✅ Saved list to outputs/shapefiles/duplicate_pairs.txt")
    print()
    print("TO INSPECT IN QGIS:")
    print("1. Load data/swisstopo_data/images/test/ folder")
    print("2. Load outputs/shapefiles/test_GT_with_duplicates.shp")
    print("3. Open the attribute table and filter by 'img' field")
    print("4. For each pair listed above, zoom to the boxes")
    print("=" * 80)


if __name__ == "__main__":
    main()