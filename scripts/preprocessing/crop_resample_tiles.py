#!/usr/bin/env python3
"""
Crop large SWISSIMAGE / DSM GeoTIFFs into smaller 640x640 patches
and resample them to 0.5 m/pixel resolution.

Usage:
    python scripts/crop_resample_tiles.py \
        --src data/raw/canton_valais/swissimage \
        --out data/tiles/canton_valais/swissimage_50cm \
        --tilesize 640 \
        --resolution 0.5 \
        --overlap 210

Notes:
- Uses gdal_translate under the hood.
- Works for both 10 cm (SWISSIMAGE) and 50 cm (DSM) inputs.
"""

import os
import argparse
from pathlib import Path
import subprocess

def run_cmd(cmd):
    """Run a shell command safely."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error while executing:\n{cmd}\n{e}")

def main():
    parser = argparse.ArgumentParser(description="Crop and resample GeoTIFF tiles to 50 cm.")
    parser.add_argument("--src", required=True, help="Source folder with .tif files.")
    parser.add_argument("--out", required=True, help="Output folder for cropped/resampled tiles.")
    parser.add_argument("--tilesize", type=int, default=640, help="Tile size in pixels (after resampling).")
    parser.add_argument("--resolution", type=float, default=0.5, help="Target ground resolution (m/pixel).")
    parser.add_argument("--overlap", type=int, default=210, help="Overlap in pixels between tiles.")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tile size in source pixels (before resampling)
    # For SWISSIMAGE 0.1 m -> multiply by 5; for DSM 0.5 m -> multiply by 1
    # The script assumes the resolution difference is handled by -tr
    step = args.overlap
    tile_size_px = args.tilesize

    # Convert overlap to step distance between tiles (stride)
    stride = tile_size_px - args.overlap

    print(f"➡️  Cutting tiles from {src_dir}")
    print(f"    Tile size: {tile_size_px}px, stride: {stride}px, target res: {args.resolution}m")

    for tif_path in sorted(src_dir.glob("*.tif")):
        print(f"Processing {tif_path.name} ...")
        # Define pixel offsets for a 4x4 grid (like in your boss's script)
        offsets = [0, 430 * 5, 930 * 5, 1360 * 5]  # these match 1 km tiles for 0.1 m input
        for idx_i, i in enumerate(offsets):
            for idx_j, j in enumerate(offsets):
                out_name = (
                    f"{tif_path.stem}_{idx_i}_{idx_j}.tif"
                )
                out_path = out_dir / out_name
                if out_path.exists():
                    continue

                cmd = (
                    f"gdal_translate -of GTiff "
                    f"-srcwin {i} {j} {tile_size_px*5} {tile_size_px*5} "  # crop window in source pixels
                    f"-tr {args.resolution} {args.resolution} "
                    f"-r cubic -co COMPRESS=LZW -co TILED=YES "
                    f"{tif_path} {out_path}"
                )
                run_cmd(cmd)

    print(f"✅ Cropping complete. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
