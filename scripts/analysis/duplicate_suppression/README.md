# Duplicate Suppression

![alt text](image.png)

Scripts for detecting and removing duplicate rock detections caused by overlapping patches during tile processing.

---

## Problem

When extracting 640×640 patches from 1km tiles with 210px overlap:
- **Overlap region:** 210×210 pixels at each patch boundary
- **Result:** Rocks in overlap regions detected multiple times
- **Impact:** Inflates false positive count in final predictions

---

## Scripts

### `find_duplicates_from_labels.py`
Finds duplicate detections by converting patch-local to tile-global coordinates.

**Usage:**
```bash
python scripts/analysis/duplicate_suppression/find_duplicates_from_labels.py
```

**Output:**
- Console: Summary of duplicates per tile
- File: `outputs/duplicate_suppression/real_duplicates.txt`

**How it works:**
1. Groups patches by tile ID (e.g., `2781_1141`)
2. Converts normalized YOLO coords to global tile pixels:
   ```
   stride = 640 - 210 = 430 pixels
   global_x = (cx * 640) + (col * stride)
   global_y = (cy * 640) + (row * stride)
   ```
3. Compares all box pairs between overlapping patches
4. Marks as duplicate if distance < 15 pixels (~7.5m)

---

### `visualize_real_duplicates.py`
Creates side-by-side visualizations showing all duplicate rocks (red) vs other rocks (green).

**Usage:**
```bash
python scripts/analysis/duplicate_suppression/visualize_real_duplicates.py
```

**Output:**
- PNG files in `outputs/duplicate_visualizations_grouped/`
- One image per unique patch pair showing ALL duplicates

---

### `find_duplicates_with_tunable_threshold.py`
Test different distance thresholds to optimize duplicate detection.

**Usage:**
```bash
python scripts/analysis/duplicate_suppression/find_duplicates_with_tunable_threshold.py --threshold 15
```

**Threshold selection:**
- **5px** (2.5m): Very strict, may miss some duplicates
- **10px** (5m): Balanced, catches most duplicates
- **15px** (7.5m): Recommended - catches all real duplicates
- **20px** (10m): Permissive, risk of false positives

---

## Expected Results (Test Set)

- **Patches analyzed:** 96 from 6 unique tiles
- **Duplicates found:** ~8 duplicate rocks
- **Unique patch pairs:** ~3 pairs with overlaps
- **Average distance:** 2-3 pixels between duplicates

---

## Usage in Production Pipeline

### During Post-Processing
After running inference on nationwide tiles:

```bash
# 1. Run inference on all patches
python scripts/inference/run_inference.py \
  --model models/final_model.pt \
  --source data/switzerland_data/processed/canton_valais \
  --output outputs/predictions/valais

# 2. Detect duplicates
python scripts/analysis/duplicate_suppression/find_duplicates_from_labels.py

# 3. Apply NMS during shapefile creation
python scripts/postprocessing/yolo_to_shapefile.py \
  --tif_dir data/switzerland_data/tiles/canton_valais \
  --labels_dir outputs/predictions/valais/predict/labels \
  --out outputs/shapefiles/valais_clean.shp \
  --nms_iou 0.5  # Remove duplicates
```

---

## ⚙️ Parameters

### Patch Configuration
- **Patch size:** 640×640 pixels
- **Overlap:** 210 pixels (33%)
- **Stride:** 430 pixels
- **Effective overlap region:** 210×210 px at boundaries

### Duplicate Detection
- **Distance threshold:** 15 pixels (recommended)
- **At 0.5m resolution:** 15px = 7.5 meters
- **For 16m rocks:** Well within expected variation

---

## 📝 Notes

- Test set patches are already extracted with overlap, so duplicates exist in ground truth
- Duplicates are expected and normal - not a data quality issue
- Suppression should be applied in post-processing, not during training
- For non-georeferenced TIFs, use label-based method (not shapefile NMS)
```
