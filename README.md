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

## Repository structure

```
switzerland_nationwide_rock_detection/
│
├── data/
│   ├── swisstopo_data/       # Ground truth labeled data (Valais + Grisons)
│   │   ├── images/           # Train/val/test splits
│   │   └── labels/           # YOLO format annotations
│   └── switzerland_data/     # Nationwide inference data
│       ├── raw/              # Raw 1×1 km tiles from Swisstopo
│       ├── tiles/            # Cropped 640×640 patches @ 0.5m resolution
│       ├── processed/        # Fused RGB+Hillshade patches for inference
│       └── URLs/             # Swisstopo export CSVs (download links per canton)
│
├── models/                   # Trained models (.pt)
│   ├── active_teacher.pt
│   ├── baseline_best.pt
│   └── final_model.pt
│
├── scripts/
│   ├── inference/
│   │   └── run_inference.py      # Runs inference on fused patches
│   ├── postprocessing/
│   │   └── yolo_to_shapefile.py  # Converts YOLO predictions to GIS shapefile
│   └── preprocessing/
│       ├── download_tiles.py         # Downloads 1×1 km Swisstopo tiles from CSV
│       ├── generate_hillshade.py     # Creates hillshade DSM tiles using gdaldem
│       ├── crop_resample_tiles.py    # Cuts 640×640 patches @ 0.5m (with overlap)
│       └── fuse_rgb_hs.py            # Fuses RGB patches with hillshade
│
└── README.md
```

Each `data/switzerland_data/raw/` and `data/switzerland_data/URLs/` subfolder contains **26 subfolders**, one per Swiss canton.

---

##  Pipeline Workflow

### **Step 1: Download Swisstopo tiles**

Download 1×1 km tiles for a specific canton:

```bash
python scripts/preprocessing/download_tiles.py \
  --csv data/switzerland_data/URLs/canton_valais/ch.swisstopo.swisssurface3d-raster.csv \
  --out data/switzerland_data/raw/canton_valais/dsm
```

> **How to get the CSV:**
> 1. Visit [Swisstopo](https://www.swisstopo.admin.ch/)
> 2. Select: **Mode de sélection** → Sélection par canton → *Your canton*
> 3. **Format:** Cloud Optimized GeoTIFF
> 4. **Résolution:** 0.1 m (for SWISSIMAGE) or 0.5 m (for DSM)
> 5. **Système de coordonnées:** MN95 (LV95 / EPSG:2056)
> 6. Click **Chercher** → **Exporter tous les liens** → Save CSV to `data/switzerland_data/URLs/canton_<name>/`

---

### **Step 2: Generate hillshade from DSM**

```bash
python scripts/preprocessing/generate_hillshade.py \
  --src data/switzerland_data/raw/canton_valais/dsm \
  --out data/switzerland_data/raw/canton_valais/hillshade \
  --az 315 \
  --alt 45
```

---

### **Step 3: Crop and resample to 640×640 @ 0.5m**

Create overlapping 640×640 patches from 1km tiles:

```bash
python scripts/preprocessing/crop_resample_tiles.py \
  --src data/switzerland_data/raw/canton_valais/swissimage \
  --out data/switzerland_data/tiles/canton_valais/swissimage_50cm \
  --tilesize 640 \
  --resolution 0.5 \
  --overlap 210
```

**Note:** The 210px overlap ensures no rocks are missed at patch boundaries, but creates duplicate detections that must be handled in post-processing.

---

### **Step 4: Fuse RGB and hillshade patches**

Combine RGB imagery with hillshade elevation data (green channel replacement):

```bash
python scripts/preprocessing/fuse_rgb_hs.py \
  --rgb_dir data/switzerland_data/tiles/canton_valais/swissimage_50cm \
  --hs_dir data/switzerland_data/tiles/canton_valais/hillshade_patches \
  --out_dir data/switzerland_data/processed/canton_valais \
  --channel 1
```

---

### **Step 5: Run inference**

Run rock detection on fused patches:

```bash
python scripts/inference/run_inference.py \
  --model models/baseline_best.pt \
  --source data/switzerland_data/processed/canton_valais \
  --output outputs/predictions/canton_valais \
  --conf 0.10 \
  --iou 0.40
```

This produces YOLO `.txt` prediction files (one per patch).

---

### **Step 6: Convert to GIS shapefile**

Convert YOLO predictions to geospatial shapefile for QGIS visualization:

```bash
python scripts/postprocessing/yolo_to_shapefile.py \
  --tif_dir data/switzerland_data/tiles/canton_valais/swissimage_50cm \
  --labels_dir outputs/predictions/canton_valais/predict/labels \
  --out outputs/shapefiles/canton_valais_detections.shp \
  --nms_iou 0.5
```

**Parameters:**
- `--tif_dir`: Directory containing georeferenced patch images (.tif)
- `--labels_dir`: Directory containing YOLO prediction files (.txt)
- `--out`: Output shapefile path (.shp or .gpkg)
- `--nms_iou`: IoU threshold for cross-patch duplicate removal (0=disabled, 0.3-0.7 recommended)

**Output:** Shapefile with rock detections in EPSG:2056 (Swiss coordinate system), ready to load in QGIS.

**Handling Duplicates:** The `--nms_iou` parameter removes duplicate detections caused by overlapping patches. A value of 0.5 means boxes with >50% overlap are considered duplicates, and only the highest-confidence detection is kept.

---

## Summary of Pipeline

```
1. Download      → 1×1 km Swisstopo tiles (SWISSIMAGE + DSM)
2. Generate      → Hillshades from DSM
3. Crop          → 640×640 patches @ 0.5m (with 210px overlap)
4. Fuse          → RGB + hillshade channels (green channel replacement)
5. Infer         → Rock detection on fused patches (YOLO predictions)
6. Convert       → YOLO predictions → GIS shapefile (with duplicate removal)
7. Visualize     → Load shapefile in QGIS
```

---

## Dependencies

```bash
pip install ultralytics rasterio geopandas shapely numpy opencv-python
```

---

## Documentation

- **Model improvements:** See [models/improvements.md](models/IMPROVEMENTS.md) for detailed experiment tracking

---

## Credits

- **Original project:** [Alexis Rufer](https://github.com/alexs-rufer/large-rocks-mapping.git)
- **Data provider:** [Swisstopo](https://www.swisstopo.admin.ch/)
- **Supervision:** Prof. Devis Tuia, Valérie Zermatten (EPFL ECEO Lab)
```

