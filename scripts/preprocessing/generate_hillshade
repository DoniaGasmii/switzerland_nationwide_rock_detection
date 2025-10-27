#!/usr/bin/env python3
"""
Generate hillshade GeoTIFFs from DSM tiles.

Usage:
    python scripts/generate_hillshade.py \
        --src data/raw/canton_valais/dsm \
        --out data/raw/canton_valais/hillshade \
        --az 315 --alt 45
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import subprocess


def run_cmd(cmd):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Error while running:\n{cmd}\n{e}")


def main():
    parser = argparse.ArgumentParser(description="Generate hillshades from DSM GeoTIFF tiles using gdaldem.")
    parser.add_argument("--src", required=True, help="Source folder containing DSM .tif tiles")
    parser.add_argument("--out", required=True, help="Output folder for hillshade .tif files")
    parser.add_argument("--az", type=int, default=315, help="Sun azimuth in degrees (default: 315)")
    parser.add_argument("--alt", type=int, default=45, help="Sun altitude in degrees (default: 45)")
    parser.add_argument("--z", type=float, default=1.0, help="Vertical exaggeration factor (default: 1.0)")
    args = parser.parse_args()

    src = Path(args.src)
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    dsm_files = [f for f in src.glob("*.tif")]

    print(f"➡️ Generating hillshades from {len(dsm_files)} DSM tiles...")
    for tif in tqdm(dsm_files, desc="Hillshade tiles"):
        out_tif = dest / tif.name
        if out_tif.exists():
            continue
        cmd = (
            f"gdaldem hillshade {tif} {out_tif} "
            f"-az {args.az} -alt {args.alt} -z {args.z} "
            f"-compute_edges -combined "
            f"-of GTiff -co COMPRESS=LZW -co TILED=YES"
        )
        run_cmd(cmd)

    print(f"✅ Done. Saved hillshades to: {dest}")


if __name__ == "__main__":
    main()
