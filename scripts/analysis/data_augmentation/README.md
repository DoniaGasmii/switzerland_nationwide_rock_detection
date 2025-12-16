# Data Augmentation Analysis

Scripts for analyzing training data distribution and creating augmented datasets to improve model performance.

---

##  Objective

Test two augmentation strategies to reduce false positives:
1. **Approach A (Uniform):** Standard YOLO augmentation on all images (tested already by Alexis)
2. **Approach B (Targeted):** More augmentation on sparse/difficult images

---

## Scripts

### `analyze_training_distribution.py`
Analyzes distribution of rocks per training image and recommends augmentation strategy.

**Usage:**
```bash
python scripts/analysis/data_augmentation/analyze_training_distribution.py
```

**Output:**
- Console: Statistics on rock distribution
- Plot: `outputs/training_analysis/training_distribution.png`
- Stats: `outputs/training_analysis/training_distribution.txt`

**Categories:**
- **Empty** (0 rocks): Negative samples
- **Sparse** (1-3 rocks): Difficult cases (need more augmentation)
- **Medium** (4-10 rocks): Balanced samples
- **Dense** (11+ rocks): Easy cases (many examples)

---

### `create_augmented_dataset.py`
Creates targeted augmentation dataset (Approach B) with more emphasis on sparse images.

**Usage:**
```bash
# Create augmented dataset
python scripts/analysis/data_augmentation/create_augmented_dataset.py
```

**Augmentation Strategy:**
- Empty (0 rocks): No augmentation (keep as-is)
- Sparse (1-3 rocks): 5x augmentation
- Medium (4-10 rocks): 2x augmentation
- Dense (11+ rocks): No augmentation (keep as-is)

**Augmentations Applied:**
- Horizontal/vertical flips
- Random rotation (±15°)
- Shift/scale/rotate
- Gaussian noise/blur
- Brightness/contrast adjustment
- HSV color jittering

**Output:**
- Dataset: `data/augmented_swisstopo_data/`
- Config: `data/augmented_swisstopo_data/data.yaml`

---

##  Training Workflow

### **Approach B: Targeted Augmentation**

1. Train with reduced online augmentation (dataset is pre-augmented):
```bash
yolo train \
  model=models/baseline_best.pt \
  data=data/augmented_swisstopo_data/data.yaml \
  epochs=50 \
  batch=8 \
  imgsz=640 \
  device=0 \
  project=outputs/training \
  name=approach_b_targeted \
  hsv_h=0.01 \
  hsv_s=0.4 \
  hsv_v=0.3 \
  degrees=5 \
  translate=0.05 \
  scale=0.3 \
  fliplr=0.5 \
  mosaic=0.5
```

---

##  Evaluation

After training both models, compare on test set:
```bash
# Evaluate Approach A
yolo val \
  model=models/baseline_best.pt \
  data=data/swisstopo_data/data.yaml \
  split=test

# Evaluate Approach B
yolo val \
  model=models/augmented_data_model.pt \
  data=data/swisstopo_data/data.yaml \
  split=test
```
---