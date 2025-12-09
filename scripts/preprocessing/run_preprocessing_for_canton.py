#!/usr/bin/env python3
"""
Automate preprocessing pipeline for a single Swiss canton.

Steps:
1. Download SWISSIMAGE 10 cm tiles
2. Download swissSURFACE3D Raster (DSM) 0.5 m tiles
3. Generate hillshades from DSM
4. Crop/resample SWISSIMAGE and hillshades to 640x640 @ 0.5 m
5. Fuse RGB + Hillshade patches

Usage:
python scripts/preprocessing/run_preprocessing_for_canton.py --canton valais
"""

import argparse
import subprocess
from pathlib import Path
import sys

# auto-detect project root even if run from nested folders (scripts/preprocessing/)
ROOT = Path(__file__).resolve()
while ROOT.name not in ("switzerland_nationwide_rock_detection", "") and not (ROOT / "data").exists():
    ROOT = ROOT.parent

def run(cmd, quiet=False):
    """Run shell command with clear logging."""
    print(f"\n {cmd}")
    try:
        if quiet:
            subprocess.run(cmd, shell=True, check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error while running:\n{cmd}\n{e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline for a given canton.")
    parser.add_argument("--canton", required=True, help="Canton name, e.g. valais, geneva, bern")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed logs")
    args = parser.parse_args()

    canton = args.canton.lower()
    quiet = args.quiet

    # ---- Define paths ----
    urls_dir = ROOT / "data" / "URLs" / f"canton_{canton}"
    raw_dir = ROOT / "data" / "raw" / f"canton_{canton}"
    tiles_dir = ROOT / "data" / "tiles" / f"canton_{canton}"
    processed_dir = ROOT / "data" / "processed" / f"canton_{canton}"

    def find_csv(folder: Path, keyword: str):
        """Find CSV file in folder whose name contains the keyword (case-insensitive)."""
        matches = [f for f in folder.glob("*.csv") if keyword.lower() in f.name.lower()]
        if not matches:
            print(f"No CSV containing '{keyword}' found in {folder}")
            return None
        if len(matches) > 1:
            print(f"Multiple CSVs containing '{keyword}' found, using first: {matches[0].name}")
        return matches[0]

    swissimage_csv = find_csv(urls_dir, "swissimage")
    dsm_csv = find_csv(urls_dir, "swisssurface3d-raster")

    if not swissimage_csv or not dsm_csv:
        print(f"Missing CSV files in {urls_dir}. Make sure both datasets are exported from Swisstopo.")
        sys.exit(1)

    raw_si = raw_dir / "swissimage"
    raw_dsm = raw_dir / "dsm"
    raw_hs = raw_dir / "hillshade"

    tiles_si = tiles_dir / "swissimage_50cm"
    tiles_hs = tiles_dir / "hillshade_patches"
    fused_dir = processed_dir / "images_hs_fusion"

    # ---- 1. Download SWISSIMAGE ----
    run(f"python scripts/preprocessing/download_tiles.py "
        f"--csv {swissimage_csv} --out {raw_si}", quiet)

    # ---- 2. Download DSM ----
    run(f"python scripts/preprocessing/download_tiles.py "
        f"--csv {dsm_csv} --out {raw_dsm}", quiet)

    # ---- 3. Generate hillshade ----
    run(f"python scripts/preprocessing/generate_hillshade.py "
        f"--src {raw_dsm} --out {raw_hs} --az 315 --alt 45", quiet)

    # ---- 4. Crop & resample SWISSIMAGE (0.1 -> 0.5 m) ----
    run(f"python scripts/preprocessing/crop_resample_tiles.py "
        f"--src {raw_si} --out {tiles_si} --src_res 0.1 --dst_res 0.5 "
        f"--tilesize 640 --overlap 210 --quiet", quiet)

    # ---- 5. Crop hillshade (already 0.5 m) ----
    run(f"python scripts/preprocessing/crop_resample_tiles.py "
        f"--src {raw_hs} --out {tiles_hs} --src_res 0.5 --dst_res 0.5 "
        f"--tilesize 640 --overlap 210 --quiet", quiet)

    # ---- 6. Fuse RGB + Hillshade ----
    run(f"python scripts/preprocessing/fuse_rgb_hs.py "
        f"--rgb_dir {tiles_si} --hs_dir {tiles_hs} --out_dir {fused_dir} --channel 1", quiet)

    print(f"\n✅ Canton {canton.capitalize()} preprocessing complete!")
    print(f"   Fused patches ready at: {fused_dir}")

if __name__ == "__main__":
    main()
