# Duplicate Suppression Analysis

This folder contains scripts for analyzing and removing duplicate rock detections caused by overlapping patches.

## Problem

When a 1km×1km tile is split into overlapping 640×640 patches (with 210px overlap), rocks in overlap regions get detected multiple times, creating duplicate bounding boxes.

## Scripts

### 1. `find_duplicates_from_labels.py` 
Finds real duplicates by converting patch-local YOLO coordinates to tile-global coordinates.

**Usage:**
```bash
python scripts/analysis/duplicate_suppression/find_duplicates_from_labels.py
```

**Output:**
- Console: List of real duplicate pairs
- File: `outputs/duplicate_suppression/real_duplicates.txt`

---

## Expected Results

### If TIFs have proper georeferencing:
- Both methods should find similar numbers of duplicates
- Shapefile method is faster and easier

### If TIFs lack georeferencing:
- Shapefile method finds FALSE POSITIVES
- Label-based method is correct
- Need to either:
  - Fix preprocessing to preserve georeferencing, OR
  - Use label-based method for duplicate removal

---

## Parameters

### Patch Extraction (from preprocessing)
- **Patch size:** 640×640 pixels
- **Overlap:** 210 pixels
- **Stride:** 430 pixels (640 - 210)

### Duplicate Detection
- **Distance threshold:** 5 pixels (label-based method)
- **IoU threshold:** 0.9 (shapefile method)

---

## Next Steps

1. Run `find_duplicates_from_labels.py` to find real duplicates
2. Run `compare_methods.py` to see if georeferencing is the issue
3. If discrepancy exists, fix preprocessing or use label-based NMS