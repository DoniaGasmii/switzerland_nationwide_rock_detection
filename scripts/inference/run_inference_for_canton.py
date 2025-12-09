#!/usr/bin/env python3
"""
Run YOLOv11 inference for a specific Swiss canton.

Author: Alexis Rufer (original)
Modified by: Donia Gasmi — added canton-based automation

Example usage:
---------------
python scripts/inference/run_inference.py \
  --model models/final_model.pt \
  --canton valais \
  --conf 0.10 --iou 0.40
"""

import sys
from pathlib import Path
import argparse
import torch
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser("YOLOv11 inference (canton-based)")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--canton", required=True, help="Canton name, e.g. valais, geneva, zurich")
    parser.add_argument("--conf", type=float, default=0.10, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.40, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="", help="'cpu', 'cuda:0', etc.")
    args = parser.parse_args()

    canton = args.canton.lower()

    # ---------------------- Path setup ----------------------
    ROOT = Path(__file__).resolve().parents[2]  # go up to project root
    source_path = ROOT / "data" / "processed" / f"canton_{canton}" / "images_hs_fusion"
    out_dir = ROOT / "data" / "predictions" / f"canton_{canton}"

    if not source_path.exists():
        sys.exit(f" Processed data not found for canton '{canton}' at: {source_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------- Device setup ---------------------
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------- Inference ------------------------
    print(f" Running inference on canton '{canton}'...")
    print(f"   Source: {source_path}")
    print(f"   Output: {out_dir}")

    model = YOLO(model_path)
    if device != "cpu":
        model.to(device)

    model.predict(
        source=source_path,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        show_labels=False,
        project=out_dir,
        name="predict",
        verbose=True,
        show_conf=True,
    )

    print(f"\nInference complete for canton '{canton}'")
    print(f"   Results saved under: {out_dir / 'predict'}")


if __name__ == "__main__":
    main()
