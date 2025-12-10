#!/usr/bin/env python3
import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import box


def yolo_to_pixel(cx, cy, w, h, width, height):
    """Convert YOLO-normalized bbox to pixel coords."""
    xmin = (cx - w / 2.0) * width
    xmax = (cx + w / 2.0) * width
    ymin = (cy - h / 2.0) * height
    ymax = (cy + h / 2.0) * height
    return xmin, ymin, xmax, ymax


def compute_iou_shapely(box1, box2):
    """Compute IoU between two shapely box geometries."""
    intersection = box1.intersection(box2).area
    union = box1.union(box2).area
    return intersection / union if union > 0 else 0.0


def nms_geospatial(gdf, iou_threshold=0.5):
    """
    Apply NMS on a GeoDataFrame of detections.
    
    Args:
        gdf: GeoDataFrame with 'geometry' and 'score' columns
        iou_threshold: IoU threshold for duplicate removal
    
    Returns:
        GeoDataFrame with duplicates removed
    """
    if len(gdf) == 0:
        return gdf
    
    # Sort by score (descending)
    gdf = gdf.sort_values('score', ascending=False).reset_index(drop=True)
    
    keep_indices = []
    suppressed = set()
    
    for i in range(len(gdf)):
        if i in suppressed:
            continue
        
        keep_indices.append(i)
        box1 = gdf.geometry.iloc[i]
        
        # Suppress overlapping lower-confidence boxes
        for j in range(i + 1, len(gdf)):
            if j in suppressed:
                continue
            
            box2 = gdf.geometry.iloc[j]
            iou = compute_iou_shapely(box1, box2)
            
            if iou > iou_threshold:
                suppressed.add(j)
    
    return gdf.iloc[keep_indices].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(
        description="Convert YOLO-normalized bboxes (with optional score) to a Shapefile using georeferenced TIFF patches."
    )
    ap.add_argument("--tif_dir", required=True, help="Folder with .tif/.tiff patches")
    ap.add_argument("--labels_dir", required=True, help="Folder with YOLO .txt labels")
    ap.add_argument("--out", required=True, help="Output Shapefile or GeoPackage (e.g. detections.shp or detections.gpkg)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subfolders")
    ap.add_argument("--fallback_crs", default="EPSG:2056", help="CRS to use if a TIFF lacks CRS (default: EPSG:2056)")
    ap.add_argument("--nms_iou", type=float, default=0.0, help="IoU threshold for cross-patch NMS (0=disabled, 0.3-0.7 recommended)")
    args = ap.parse_args()

    # Collect TIFFs
    tif_glob = "**/*.tif" if args.recursive else "*.tif"
    tiff_glob = "**/*.tiff" if args.recursive else "*.tiff"
    tif_paths = sorted(list(Path(args.tif_dir).glob(tif_glob)) + list(Path(args.tif_dir).glob(tiff_glob)))

    records, geoms = [], []
    last_crs = None  # keep track of a valid CRS to attach to the GeoDataFrame

    for tif in tif_paths:
        stem = tif.stem
        # Match any label file that starts with the image stem (handles _0, _1, etc.)
        label_pattern = f"{stem}*.txt"
        if args.recursive:
            label_files = sorted(Path(args.labels_dir).rglob(label_pattern))
        else:
            label_files = sorted(Path(args.labels_dir).glob(label_pattern))

        if not label_files:
            continue

        with rasterio.open(tif) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs if src.crs else args.fallback_crs
            last_crs = crs

            for lf in label_files:
                with open(lf, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        # YOLO normalized: class cx cy w h [score]
                        try:
                            cls = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            score = float(parts[5]) if len(parts) >= 6 else np.nan
                        except Exception:
                            continue

                        # 1) YOLO → pixel bbox
                        xmin, ymin, xmax, ymax = yolo_to_pixel(cx, cy, w, h, width, height)

                        # 2) Pixel → map coords via affine (col=x, row=y)
                        x1, y1 = transform * (xmin, ymin)   # top-left
                        x2, y2 = transform * (xmax, ymax)   # bottom-right

                        # 3) Ensure proper ordering
                        minx, maxx = sorted([x1, x2])
                        miny, maxy = sorted([y1, y2])

                        geoms.append(box(minx, miny, maxx, maxy))
                        records.append({
                            "img": tif.name,
                            "label": lf.name,
                            "class": cls,
                            "score": score,
                        })

    if not geoms:
        raise SystemExit("No detections found! (Check filename matching and that labels are YOLO-normalized)")

    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs=last_crs or args.fallback_crs)
    
    print(f"Total detections before NMS: {len(gdf)}")
    
    # Apply cross-patch NMS if requested
    if args.nms_iou > 0:
        gdf = nms_geospatial(gdf, iou_threshold=args.nms_iou)
        print(f"✂️  Detections after NMS (IoU={args.nms_iou}): {len(gdf)}")
        print(f"🗑️  Removed duplicates: {len(geoms) - len(gdf)}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write (support .gpkg too—nice in QGIS)
    if out_path.suffix.lower() == ".gpkg":
        gdf.to_file(out_path, driver="GPKG", layer="detections")
    else:
        gdf.to_file(out_path, driver="ESRI Shapefile")

    print(f"✅ Wrote {len(gdf)} boxes to {out_path}")
    print(f"CRS: {gdf.crs}")


if __name__ == "__main__":
    main()