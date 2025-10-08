# 🇨🇭 Swiss Rock Inference Pipeline

This repository contains a lightweight pipeline to:
1.  Download Swiss aerial images (SWISSIMAGE) & DSM (SwissSurface3D)
2.  Generate hillshade layers and fused tiles
3.  Run object detection inference (YOLO-based) on tiled Swiss data
4.  Convert detections to polygons for QGIS visualization

##  Repo Structure
```
switzerland_nationwide_rock_detection/
├─ README.md
├─ .gitignore
├─ environment.yaml              
├─ notebooks/
│   └─ 01_prepare_nationwide_dataset.ipynb
├─ scripts/
│   ├─ make_grid_lv95.py
│   ├─ download_tiles.py
│   ├─ generate_hillshade.py
│   ├─ cut_tiles.py
│   ├─ fuse_green.py
│   └─ run_inference.py
├─ data/
│    ├─ raw/
│    │   ├─ region_01/
│    │   │   ├─ swissimage/
│    │   │   ├─ dsm/
│    │   │   └─ hillshade/
│    │   ├─ region_02/
│    │   ├─ canton_bern/
│    │   └─ ...
│    ├─ tiles/
│    │   ├─ region_01/
│    │   │   ├─ swissimage/
│    │   │   └─ hillshade/
│    │   ├─ region_02/
│    │   └─ canton_bern/
│    ├─ processed/
│    │   ├─ region_01/images_hs_fusion/
│    │   ├─ region_02/images_hs_fusion/
│    │   └─ canton_bern/images_hs_fusion/
│    │
│    └─ outputs/
│        ├─ predictions/
│        │   ├─ region_01/
│        │   ├─ region_02/
│        │   └─ canton_bern/
│        └─ shapefiles/
│            ├─ region_01/
│            ├─ region_02/
│            └─ canton_bern/├─ outputs/
└─ docs/
    └─ figures/  
````

##  Quick Start

```bash
conda env create -f environment.yml
conda activate swiss-rock
````

Then run:

```bash
python scripts/make_grid_lv95.py
python scripts/download_tiles.py --tile-list data/tile_list.csv
python scripts/generate_hillshade.py
python scripts/cut_tiles.py
python scripts/fuse_green.py
python scripts/run_inference.py
```

## 🗂 Data sources

* [SWISSIMAGE 10cm](https://www.swisstopo.admin.ch/fr/orthophotos-swissimage-10-cm)
* [SwissSurface3D DSM](https://www.swisstopo.admin.ch/fr/modele-altimetrique-swisssurface3d)
* [MNT25 (altitude)](https://www.swisstopo.admin.ch/fr/modele-altimetrique-mnt25-200m)

