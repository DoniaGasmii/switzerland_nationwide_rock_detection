# Preprocessing Pipeline

This directory contains scripts to prepare nationwide Swiss geospatial data for rock detection:

1. **Download** 1×1 km SWISSIMAGE (RGB) and DSM tiles from Swisstopo (per canton)  
2. **Generate** hillshade from DSM  
3. **Crop & resample** into 640×640 patches at 0.5 m resolution (with 210 px overlap)  
4. **Fuse** RGB + hillshade by replacing the green channel  

Outputs are organized per canton under `data/switzerland_data/`. The final fused patches in `processed/` are ready for inference.

## Scripts

- `download_tiles.py` – downloads raw GeoTIFFs using Swisstopo CSV exports  
- `generate_hillshade.py` – creates hillshade from DSM tiles  
- `crop_resample_tiles.py` – generates overlapping patches  
- `fuse_rgb_hs.py` – fuses RGB and hillshade patches  

## Bash Usage

```bash
# 1. Download raw tiles (example: Valais canton)
python scripts/preprocessing/download_tiles.py \
  --csv data/switzerland_data/URLs/canton_valais/ch.swisstopo.swisssurface3d-raster.csv \
  --out data/switzerland_data/raw/canton_valais/dsm

# 2. Generate hillshade
python scripts/preprocessing/generate_hillshade.py \
  --src data/switzerland_data/raw/canton_valais/dsm \
  --out data/switzerland_data/raw/canton_valais/hillshade \
  --az 315 --alt 45

# 3. Crop & resample to 640×640 @ 0.5m
python scripts/preprocessing/crop_resample_tiles.py \
  --src data/switzerland_data/raw/canton_valais/swissimage \
  --out data/switzerland_data/tiles/canton_valais/swissimage_50cm \
  --tilesize 640 --resolution 0.5 --overlap 210

# 4. Fuse RGB + hillshade
python scripts/preprocessing/fuse_rgb_hs.py \
  --rgb_dir data/switzerland_data/tiles/canton_valais/swissimage_50cm \
  --hs_dir data/switzerland_data/tiles/canton_valais/hillshade_patches \
  --out_dir data/switzerland_data/processed/canton_valais \
  --channel 1
```

> Replace `canton_valais` with your target folder. Ensure CSV exports are saved under `data/switzerland_data/URLs/` per canton.