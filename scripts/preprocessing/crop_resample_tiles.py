#!/usr/bin/env python3
"""
Crop GeoTIFF tiles (SWISSIMAGE, DSM, or Hillshade) into smaller 640x640 patches,
optionally resampling to a target ground resolution.

Usage examples:
---------------
# SWISSIMAGE (0.1 m → 0.5 m)
python scripts/preprocessing/crop_resample_tiles.py \
  --src data/raw/canton_valais/swissimage \
  --out data/tiles/canton_valais/swissimage_50cm \
  --src_res 0.1 --dst_res 0.5 --tilesize 640 --overlap 210

# Hillshade (already 0.5 m)
python scripts/preprocessing/crop_resample_tiles.py \
  --src data/raw/canton_valais/hillshade \
  --out data/tiles/canton_valais/hillshade_patches \
  --src_res 0.5 --dst_res 0.5 --tilesize 640 --overlap 210

# Quiet mode (no GDAL progress or prints)
python scripts/preprocessing/crop_resample_tiles.py \
  --src ... --out ... --src_res 0.5 --dst_res 0.5 --quiet
"""

import argparse
from pathlib import Path
import subprocess

def run_cmd(cmd: str, quiet: bool):
    """Run shell command safely."""
    try:
        if quiet:
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error while executing:\n{cmd}\n{e}")

def main():
    parser = argparse.ArgumentParser(description="Crop GeoTIFF tiles into smaller patches.")
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

    # scaling factor between input and output resolution
    scale = args.dst_res / args.src_res
    stride = args.tilesize - args.overlap
    offsets = [0, int(stride / scale), int((stride * 2) / scale), int((stride * 3) / scale)]

    if not args.quiet:
        print(f"➡️ Cutting tiles from {src_dir}")
        print(f"   Source res: {args.src_res} m | Target res: {args.dst_res} m | Scale: {scale:.1f}×")
        print(f"   Tile size: {args.tilesize}px | Overlap: {args.overlap}px | Stride: {stride}px")

    for tif_path in sorted(src_dir.glob("*.tif")):
        for idx_i, i in enumerate(offsets):
            for idx_j, j in enumerate(offsets):
                out_name = f"{tif_path.stem}_{idx_i}_{idx_j}.tif"
                out_path = out_dir / out_name
                if out_path.exists():
                    continue
                # add -q to disable GDAL’s verbose output
                cmd = (
                    f"gdal_translate {'-q ' if args.quiet else ''}-of GTiff "
                    f"-srcwin {i} {j} {int(args.tilesize/scale)} {int(args.tilesize/scale)} "
                    f"-tr {args.dst_res} {args.dst_res} "
                    f"-r cubic -co COMPRESS=LZW -co TILED=YES "
                    f"{tif_path} {out_path}"
                )
                run_cmd(cmd, args.quiet)

    if not args.quiet:
        print(f"✅ Cropping complete. Results saved to: {out_dir}")

if __name__ == "__main__":
    main()
