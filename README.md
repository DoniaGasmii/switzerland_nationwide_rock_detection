# 🇨🇭 Switzerland Nationwide Rock Detection

This repository extends and scales the original **Large Rocks Mapping** project by [**Alexis Rufer**](https://github.com/alexs-rufer/large-rocks-mapping.git), adapting his regional rock detection pipeline to cover the **entire country of Switzerland**.

Alexis initiated the project, developing and training the first two models (`active_teacher.pt`, `baseline_best.pt`) focused on local datasets.
This continuation aims to **expand the pipeline nationwide**, improve automation, and test a new variant model (`final_model.pt`) for enhanced performance.

---

## Models

| Model                 | Description                                                   | Source                                                                                        |
| --------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **baseline_best.pt**  | Original baseline model by Alexis                             | [Google Drive link](https://drive.google.com/drive/folders/12j4Fw6odNt1Sird00nCmjMQe3X7Qq17J) |
| **active_teacher.pt** | Second model trained using an active learning strategy        | [Google Drive link](https://drive.google.com/drive/folders/12j4Fw6odNt1Sird00nCmjMQe3X7Qq17J) |
| **final_model.pt**    | Additional model trained later to further improve performance | *(Link to be added)*                                                                          |

---

## 📁 Repository structure

```
switzerland_nationwide_rock_detection/
│
├── data/
│   ├── mnt25/                # Low-res altitude layer (MNT25, 200m)
│   ├── raw/                  # Raw unprocessed data (1×1 km tiles from Swisstopo)
│   │   ├── canton_valais/
│   │   ├── canton_bern/
│   │   └── canton_geneva/
│   ├── tiles/                # Cropped 640×640 patches @ 0.5m resolution
│   ├── processed/            # Final fused RGB+Hillshade patches for inference
│   └── URLs/                 # Swisstopo export CSVs (download links per canton)
│
├── models/                   # Trained models (.pt)
│   ├── active_teacher.pt
│   ├── baseline_best.pt
│   └── final_model.pt
│
├── scripts/
│   ├── inference/
│   │   └── run_inference.py  # Runs inference on fused patches
│   └── preprocessing/
│       ├── download_tiles.py     # Downloads 1×1 km Swisstopo tiles from CSV
│       ├── generate_hillshade.py # Creates hillshade DSM tiles using gdaldem
│       ├── crop_resample_tiles.py# Cuts 640×640 patches @ 0.5m (with overlap)
│       └── fuse_rgb_hs.py        # Fuses RGB patches with hillshade
│
└── README.md
```

Each `data/raw/` and `data/URLs/` subfolder contains **26 subfolders**, one per Swiss canton.

---

## How to use the scripts

### 1. Download Swisstopo tiles

Example (Geneva, DSM tiles):

```bash
python scripts/preprocessing/download_tiles.py --csv data/URLs/canton_valais/ch.swisstopo.swisssurface3d-raster.csv --out data/raw/canton_valais/dsm
```

> On the Swisstopo site, select:
>
> * **Mode de sélection:** Sélection par canton → *Geneva*
> * **Format:** Cloud Optimized GeoTIFF
> * **Résolution:** 0.1 m (for SWISSIMAGE) or 0.5 m (for DSM)
> * **Système de coordonnées:** MN95 (LV95 / EPSG:2056)
> * Click **Chercher** → **Exporter tous les liens** → Save CSV to `data/URLs/canton_<name>/`

---

### 2. Crop and resample to 640×640 @ 0.5 m

```bash
python scripts/preprocessing/crop_resample_tiles.py --src data/raw/canton_valais/swissimage --out data/tiles/canton_valais/swissimage_50cm --tilesize 640 --resolution 0.5 --overlap 210
```

---

### 3. Generate hillshade from DSM

```bash
python scripts/preprocessing/generate_hillshade.py --src data/raw/canton_valais/dsm --out data/raw/canton_valais/hillshade --az 315 --alt 45
```

---

### 4. Fuse RGB and hillshade patches

```bash
python scripts/preprocessing/fuse_rgb_hs.py --rgb_dir data/tiles/canton_valais/swissimage_50cm --hs_dir data/tiles/canton_valais/hillshade_patches --out_dir data/processed/images_hs_fusion --channel 1
```

---

### 5. Run inference (example)

```bash
python scripts/inference/run_inference.py --model models/final_model.pt --input data/processed/images_hs_fusion --output outputs/predictions
```

---

## Summary of pipeline

1. **Download** → 1×1 km Swisstopo tiles (SWISSIMAGE + DSM)
2. **Generate** → Hillshades from DSM
3. **Crop & resample** → 640×640 @ 0.5 m
4. **Fuse** → RGB + hillshade channels
5. **Infer** → Rock detection on fused images
6. **Visualize** → Convert YOLO bounding boxes to polygons in QGIS
