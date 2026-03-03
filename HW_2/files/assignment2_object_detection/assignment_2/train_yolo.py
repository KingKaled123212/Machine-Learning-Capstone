"""
Train YOLOv8n on Penn-Fudan or Oxford Pets.

Usage:
    python train_yolo.py --dataset pennfudan --epochs 15
    python train_yolo.py --dataset oxford_pets --epochs 20 --batch-size 8
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import time
import torch
from pathlib import Path

from yolo import YOLOWrapper
from utils.general import set_seed, get_device, setup_logger, Timer
from utils.visualize import plot_comparison_bar


# ---------------------------------------------------------------------------
# Dataset YAML preparation
# ---------------------------------------------------------------------------

def prepare_pennfudan_yolo(image_size: int = 512) -> str:
    """
    Penn-Fudan doesn't have YOLO-format annotations natively.
    We convert the mask-based annotations to YOLO format (normalized xywh).
    Returns path to dataset.yaml.
    """
    from pathlib import Path
    import numpy as np
    from PIL import Image

    data_root = Path("data/pennfudan")
    yaml_path = data_root / "dataset.yaml"

    if yaml_path.exists():
        return str(yaml_path)

    print("Converting Penn-Fudan masks → YOLO format labels …")

    for split in ("train", "val", "test"):
        img_dir  = data_root / split / "images"
        mask_dir = data_root / split / "masks"
        lbl_dir  = data_root / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.png")):
            mask_path = mask_dir / img_path.name.replace(".png", "_mask.png")
            if not mask_path.exists():
                continue

            img  = Image.open(img_path)
            W, H = img.size
            mask = np.array(Image.open(mask_path))

            obj_ids = np.unique(mask)
            obj_ids = obj_ids[obj_ids != 0]

            lines = []
            for oid in obj_ids:
                ys, xs = np.where(mask == oid)
                if len(xs) == 0:
                    continue
                xmin, xmax = xs.min(), xs.max()
                ymin, ymax = ys.min(), ys.max()
                # YOLO format: class cx cy w h (normalised)
                cx = ((xmin + xmax) / 2) / W
                cy = ((ymin + ymax) / 2) / H
                bw = (xmax - xmin) / W
                bh = (ymax - ymin) / H
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            lbl_file = lbl_dir / (img_path.stem + ".txt")
            lbl_file.write_text("\n".join(lines))

    # Write YAML
    yaml_content = f"""path: {data_root.resolve()}
train: train/images
val:   val/images
test:  test/images

nc: 1
names: ['person']
"""
    yaml_path.write_text(yaml_content)
    print(f"Wrote {yaml_path}")
    return str(yaml_path)


def prepare_oxford_pets_yolo(image_size: int = 512) -> str:
    """
    Convert Oxford Pets subset annotations to YOLO format.
    Returns path to dataset.yaml.
    """
    import xml.etree.ElementTree as ET
    import numpy as np
    from PIL import Image

    data_root = Path("data/oxford_pets")
    yaml_path = data_root / "dataset.yaml"

    if yaml_path.exists():
        # Check if labels dirs exist
        if (data_root / "train" / "labels").exists():
            return str(yaml_path)

    print("Converting Oxford Pets XML → YOLO format labels …")

    # Infer class list from train images
    train_img_dir = data_root / "train" / "images"
    breed_names   = sorted({p.stem.rsplit("_", 1)[0] for p in train_img_dir.glob("*.jpg")})
    class_to_idx  = {b: i for i, b in enumerate(breed_names)}

    for split in ("train", "val", "test"):
        img_dir = data_root / split / "images"
        ann_dir = data_root / split / "annotations"
        lbl_dir = data_root / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(img_dir.glob("*.jpg")):
            img  = Image.open(img_path)
            W, H = img.size
            breed = img_path.stem.rsplit("_", 1)[0]
            cls_id = class_to_idx.get(breed, 0)

            xml_path = ann_dir / (img_path.stem + ".xml")
            lines    = []

            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    bb = obj.find("bndbox")
                    if bb is None:
                        continue
                    try:
                        xmin = float(bb.find("xmin").text)
                        ymin = float(bb.find("ymin").text)
                        xmax = float(bb.find("xmax").text)
                        ymax = float(bb.find("ymax").text)
                    except (TypeError, ValueError):
                        continue
                    cx = ((xmin + xmax) / 2) / W
                    cy = ((ymin + ymax) / 2) / H
                    bw = (xmax - xmin) / W
                    bh = (ymax - ymin) / H
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                # Whole-image box fallback
                lines.append(f"{cls_id} 0.5 0.5 1.0 1.0")

            (lbl_dir / (img_path.stem + ".txt")).write_text("\n".join(lines))

    # Overwrite YAML with correct class info
    yaml_content = (
        f"path: {data_root.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n"
        f"\n"
        f"nc: {len(breed_names)}\n"
        f"names: {breed_names}\n"
    )
    yaml_path.write_text(yaml_content)
    print(f"Wrote {yaml_path}")
    return str(yaml_path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)
    device_str = "0" if torch.cuda.is_available() else "cpu"
    logger = setup_logger("yolo", log_dir=f"outputs/yolo/{args.dataset}/logs")

    logger.info(f"{'='*60}")
    logger.info(f"YOLOv8n Training  |  Dataset: {args.dataset}  |  Epochs: {args.epochs}")
    logger.info(f"{'='*60}")

    # --- Prepare YOLO dataset format ---
    logger.info("Preparing YOLO dataset format …")
    if args.dataset == "pennfudan":
        yaml_path = prepare_pennfudan_yolo(args.image_size)
        n_classes = 1
    elif args.dataset == "oxford_pets":
        yaml_path = prepare_oxford_pets_yolo(args.image_size)
        import yaml as _yaml
        with open(yaml_path) as f:
            cfg = _yaml.safe_load(f)
        n_classes = cfg["nc"]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Dataset YAML: {yaml_path}")

    # --- Build model ---
    wrapper = YOLOWrapper(
        num_classes=n_classes,
        pretrained=True,
        model_size="n",
        device=device_str,
    )

    # --- Train ---
    save_dir = f"outputs/yolo/{args.dataset}"
    logger.info(f"Starting training for {args.epochs} epochs …")

    with Timer() as t_total:
        results = wrapper.train(
            data_yaml=yaml_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            image_size=args.image_size,
            project=save_dir,
            name="run",
            patience=args.patience,
            amp=args.amp,
        )

    logger.info(f"Training complete in {t_total.elapsed_str()}")

    # --- Evaluate on test split ---
    logger.info("\n=== Final Test Evaluation ===")
    best_weights = Path(save_dir) / "run" / "weights" / "best.pt"
    if best_weights.exists():
        wrapper.load(str(best_weights))

    test_metrics = wrapper.evaluate(
        data_yaml=yaml_path,
        split="test",
        image_size=args.image_size,
        batch_size=args.batch_size,
    )
    logger.info(f"mAP@0.5   : {test_metrics['map50']:.4f}")
    logger.info(f"Precision : {test_metrics['precision']:.4f}")
    logger.info(f"Recall    : {test_metrics['recall']:.4f}")

    # --- Inference speed ---
    speed = 0.0
    logger.info(f"Speed     : {speed:.1f} img/s")
    test_metrics["inference_speed"] = speed
    test_metrics["training_time"]   = t_total.elapsed

    # --- Save results ---
    out_path = Path(save_dir) / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(test_metrics, indent=2))
    logger.info(f"Results saved to {out_path}")

    return test_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8n")
    p.add_argument("--dataset",    type=str, default="pennfudan",
                   choices=["pennfudan", "oxford_pets"])
    p.add_argument("--epochs",     type=int, default=15)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--patience",   type=int, default=5)
    p.add_argument("--amp",        action="store_true", default=True)
    p.add_argument("--no-amp",     dest="amp", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
