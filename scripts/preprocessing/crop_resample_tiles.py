#!/usr/bin/env python3
"""
Crop GeoTIFF tiles into smaller patches with uniform naming based on geographic coordinates.
Uses subprocess to call gdalinfo instead of importing osgeo.
"""

import argparse
from pathlib import Path
import subprocess
import json
import sys

def get_tile_bounds(tif_path):
    """Extract geographic bounds from GeoTIFF using gdalinfo."""
    try:
        cmd = f'gdalinfo -json "{tif_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        # Extract bounds from cornerCoordinates
        coords = info['cornerCoordinates']
        x_min = coords['lowerLeft'][0]
        y_min = coords['lowerLeft'][1]
        x_max = coords['upperRight'][0]
        y_max = coords['upperRight'][1]
        
        return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"⚠️ Error reading {tif_path}: {e}")
        return None

def get_tile_size(tif_path):
    """Get raster dimensions using gdalinfo."""
    try:
        cmd = f'gdalinfo -json "{tif_path}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)
        
        width = info['size'][0]
        height = info['size'][1]
        
        return (width, height)
    except Exception as e:
        print(f"⚠️ Error reading size of {tif_path}: {e}")
        return None

def run_cmd(cmd: str, quiet: bool):
    """Run shell command safely."""
    try:
        if quiet:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error while executing:\n{cmd}\n{e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Crop GeoTIFF tiles into smaller patches with uniform naming.")
    parser.add_argument("--src", required=True, help="Source folder with .tif files.")
    parser.add_argument("--out", required=True, help="Output folder for cropped/resampled tiles.")
    parser.add_argument("--src_res", type=float, required=True, help="Source ground resolution (m/pixel).")
    parser.add_argument("--dst_res", type=float, required=True, help="Target ground resolution (m/pixel).")
    parser.add_argument("--tilesize", type=int, default=640, help="Tile size in pixels after resampling.")
    parser.add_argument("--overlap", type=int, default=210, help="Overlap in pixels between tiles.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress and print output.")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Calculate ground size per output tile
    tile_ground_size = args.tilesize * args.dst_res  # in meters
    stride_ground = (args.tilesize - args.overlap) * args.dst_res  # stride in meters

    # Input pixel sizes
    src_tile_size = int(tile_ground_size / args.src_res)  # size in source pixels
    src_stride = int(stride_ground / args.src_res)  # stride in source pixels

    if not args.quiet:
        print(f"➡️ Cutting tiles from {src_dir}")
        print(f"   Source res: {args.src_res} m/px → Target res: {args.dst_res} m/px")
        print(f"   Output tile: {args.tilesize}px ({tile_ground_size}m ground)")
        print(f"   Overlap: {args.overlap}px ({args.overlap * args.dst_res}m ground)")
        print(f"   Source tile size: {src_tile_size}px")
        print(f"   Source stride: {src_stride}px")

    tif_files = sorted(src_dir.glob("*.tif"))
    if not tif_files:
        print(f"❌ No .tif files found in {src_dir}")
        sys.exit(1)

    total_created = 0
    total_skipped = 0

    for tif_path in tif_files:
        bounds = get_tile_bounds(tif_path)
        if bounds is None:
            print(f"⚠️ Skipping {tif_path.name} (could not read bounds)")
            continue
        
        size = get_tile_size(tif_path)
        if size is None:
            print(f"⚠️ Skipping {tif_path.name} (could not read size)")
            continue
        
        x_min, y_min, x_max, y_max = bounds
        src_width, src_height = size

        # Calculate how many tiles fit in this source image
        idx_i = 0
        y_offset = 0
        while y_offset + src_tile_size <= src_height:
            idx_j = 0
            x_offset = 0
            while x_offset + src_tile_size <= src_width:
                # Calculate the geographic center of this tile
                tile_x_min = x_min + (x_offset * args.src_res)
                tile_y_max = y_max - (y_offset * args.src_res)
                tile_x_center = tile_x_min + (tile_ground_size / 2)
                tile_y_center = tile_y_max - (tile_ground_size / 2)

                # Convert center to grid coordinates (LV95 km grid)
                grid_x = int(tile_x_center / 1000)
                grid_y = int(tile_y_center / 1000)

                # Create uniform name based on grid coordinates
                out_name = f"{grid_x}_{grid_y}_{idx_i}_{idx_j}.tif"
                out_path = out_dir / out_name

                if out_path.exists():
                    total_skipped += 1
                    idx_j += 1
                    x_offset += src_stride
                    continue

                # Use gdal_translate with source pixel coordinates
                cmd = (
                    f"gdal_translate {'-q ' if args.quiet else ''}-of GTiff "
                    f"-srcwin {x_offset} {y_offset} {src_tile_size} {src_tile_size} "
                    f"-outsize {args.tilesize} {args.tilesize} "
                    f"-r cubic -co COMPRESS=LZW -co TILED=YES "
                    f'"{tif_path}" "{out_path}"'
                )
                
                if run_cmd(cmd, args.quiet):
                    total_created += 1
                
                idx_j += 1
                x_offset += src_stride
            
            idx_i += 1
            y_offset += src_stride

    if not args.quiet:
        print(f"✅ Cropping complete!")
        print(f"   Created: {total_created} tiles")
        print(f"   Skipped (already exist): {total_skipped} tiles")
        print(f"   Output: {out_dir}")

if __name__ == "__main__":
    main()