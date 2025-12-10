# Rock Detection Model Improvements

This document tracks all improvements made to the baseline rock detection models to reduce false positives and improve performance for nationwide deployment.

---

## Baseline Performance

### Objective
Re-evaluate Alexis's trained models on the test set to establish a reproducible baseline before implementing improvements.

### 📁 Directory Structure
```
outputs/
└── baseline_reproduction/
    ├── baseline_best_test/
    │   ├── results.csv
    │   ├── predictions.json
    │   └── *.png (plots)
    └── active_teacher_test/
        ├── results.csv
        ├── predictions.json
        └── *.png (plots)
```

### Models Tested
- `baseline_best.pt` - Best supervised model from Alexis's work
- `active_teacher.pt` - Semi-supervised model with Active Teacher framework

### Dataset
- **Test set:** 96 images, 367 labeled rocks
- **Source:** Swisstopo ground truth (Valais + Grisons regions)
- **Resolution:** 0.5m/pixel, 640×640 patches
- **Input:** RGB + hillshade fusion (green channel replacement)

### Evaluation Settings
- Confidence threshold: 0.10
- IoU threshold: 0.4
- Device: NVIDIA RTX 2080 Ti

### Results

| Model | Metric | Alexis (Report) | Our Results | Difference |
|-------|--------|----------------|-------------|------------|
| **baseline_best** | Precision | 71.0% | 75.4% | +4.4% ✅ |
| | Recall | 72.8% | 76.0% | +3.2% ✅ |
| | F1 | 71.9% | 75.7% | +3.8% ✅ |
| | F2 | 72.5% | 75.9% | +3.4% ✅ |
| | mAP50 | 77.2% | 79.2% | +2.0% ✅ |
| | mAP50-95 | 42.5% | 44.9% | +2.4% ✅ |
| **active_teacher** | Precision | 57.6% | 72.8% | +15.2% ⚠️ |
| | Recall | 84.2% | 75.7% | -8.5% ⚠️ |
| | F1 | 68.5% | 74.2% | +5.7% |
| | F2 | 77.1% | 75.1% | -2.0% ✅ |
| | mAP50 | 77.4% | 77.4% | 0.0% ✅ |
| | mAP50-95 | 43.9% | 43.7% | -0.2% ✅ |

### Key Observations
1. **Baseline performance confirmed** - Models work as expected
2. **33 negative samples** in test set (images with no rocks)
3. **2 duplicate labels removed** automatically by YOLO during validation

### Conclusion
Baseline models successfully reproduced. Both models perform well (~75-79% mAP50) but suffer from **false positives** as noted by Swisstopo feedback.

---

## Improvement Plan

### Planned Experiments
1. **Duplicate Suppression** - Remove overlapping detections across patch boundaries
2. **Hard Negative Mining** - Add challenging negative samples (urban, forest, glacier)
3. **Data Augmentation** - Augment negative samples only
4. **Focal Loss** - Re-train with focal loss to reduce false positives
5. **Targeted Negative Samples** - Add 30-50% negative samples from diverse terrain

---
