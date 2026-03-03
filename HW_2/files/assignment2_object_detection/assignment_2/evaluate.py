"""
Evaluate a trained model on the test split of a dataset.

Usage:
    python evaluate.py --model faster_rcnn --dataset pennfudan
    python evaluate.py --model yolo        --dataset oxford_pets
    python evaluate.py --model faster_rcnn --dataset pennfudan \
                       --weights outputs/faster_rcnn/pennfudan/best_model.pth
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from transforms import collate_fn
from utils.general import set_seed, get_device, load_checkpoint, setup_logger
from utils.metrics import evaluate_coco
from utils.engine import measure_inference_speed
from utils.visualize import visualize_predictions


# ---------------------------------------------------------------------------
# Evaluate Faster R-CNN
# ---------------------------------------------------------------------------

def eval_faster_rcnn(args):
    from train_faster_rcnn import get_datasets
    from faster_rcnn import build_faster_rcnn

    device = get_device()
    logger = setup_logger("eval_frcnn")

    logger.info(f"Evaluating Faster R-CNN on {args.dataset}")

    _, _, test_ds, num_classes, class_names = get_datasets(args.dataset, args.image_size)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn
    )

    model = build_faster_rcnn(num_classes=num_classes, pretrained=False).to(device)

    weights = args.weights or f"outputs/faster_rcnn/{args.dataset}/best_model.pth"
    if not Path(weights).exists():
        logger.warning(f"Weights not found at {weights}. Using random weights (for testing only).")
    else:
        load_checkpoint(weights, model, device=device)

    # Evaluate
    results = evaluate_coco(model, test_loader, device, class_names)
    speed   = measure_inference_speed(model, test_loader, device, n_batches=50)
    results.inference_speed = speed

    logger.info(f"\nTest Results — Faster R-CNN ({args.dataset}):")
    logger.info(str(results))

    # Save
    out = Path(f"outputs/faster_rcnn/{args.dataset}/eval_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "map50": results.map50,
        "map50_95": results.map50_95,
        "precision": results.precision,
        "recall": results.recall,
        "inference_speed": results.inference_speed,
        "per_class_ap": results.per_class_ap,
    }, indent=2))
    logger.info(f"Saved to {out}")

    # Save sample predictions
    if args.save_predictions:
        _save_preds(model, test_ds, device, class_names,
                    Path(f"outputs/faster_rcnn/{args.dataset}/eval_predictions"))

    return results


# ---------------------------------------------------------------------------
# Evaluate YOLOv8n
# ---------------------------------------------------------------------------

def eval_yolo(args):
    from yolo import YOLOWrapper
    from train_yolo import prepare_pennfudan_yolo, prepare_oxford_pets_yolo
    import yaml as _yaml

    logger = setup_logger("eval_yolo")
    logger.info(f"Evaluating YOLOv8n on {args.dataset}")

    if args.dataset == "pennfudan":
        yaml_path = prepare_pennfudan_yolo(args.image_size)
        n_classes = 1
    else:
        yaml_path = prepare_oxford_pets_yolo(args.image_size)
        with open(yaml_path) as f:
            n_classes = _yaml.safe_load(f)["nc"]

    wrapper = YOLOWrapper(num_classes=n_classes, pretrained=False)

    weights = args.weights or f"outputs/yolo/{args.dataset}/run/weights/best.pt"
    if not Path(weights).exists():
        logger.warning(f"Weights not found at {weights}. Using untrained model (for testing only).")
    else:
        wrapper.load(weights)

    metrics = wrapper.evaluate(
        data_yaml=yaml_path, split="test",
        image_size=args.image_size, batch_size=args.batch_size,
    )
    speed = 0.0
    metrics["inference_speed"] = speed

    logger.info(f"\nTest Results — YOLOv8n ({args.dataset}):")
    for k, v in metrics.items():
        logger.info(f"  {k:20s}: {v}")

    out = Path(f"outputs/yolo/{args.dataset}/eval_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    logger.info(f"Saved to {out}")

    return metrics


# ---------------------------------------------------------------------------
# Helper: save sample prediction images (Faster R-CNN only)
# ---------------------------------------------------------------------------

def _save_preds(model, dataset, device, class_names, out_dir, n=8):
    import random
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for idx in indices:
        img, target = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]
        visualize_predictions(
            image=img,
            boxes=pred["boxes"].cpu().numpy(),
            labels=pred["labels"].cpu().numpy(),
            scores=pred["scores"].cpu().numpy(),
            class_names=class_names,
            gt_boxes=target["boxes"].numpy(),
            gt_labels=target["labels"].numpy(),
            conf_threshold=0.3,
            save_path=str(out_dir / f"pred_{idx:04d}.png"),
        )
    print(f"Saved {n} prediction images to {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained detection model")
    p.add_argument("--model",   type=str, required=True, choices=["faster_rcnn", "yolo"])
    p.add_argument("--dataset", type=str, required=True, choices=["pennfudan", "oxford_pets"])
    p.add_argument("--weights", type=str, default=None,
                   help="Path to model weights (optional; uses default output path otherwise)")
    p.add_argument("--image-size",  type=int, default=512)
    p.add_argument("--batch-size",  type=int, default=4)
    p.add_argument("--save-predictions", action="store_true", default=True,
                   help="Save sample prediction images")
    return p.parse_args()


if __name__ == "__main__":
    set_seed(42)
    args = parse_args()

    if args.model == "faster_rcnn":
        eval_faster_rcnn(args)
    else:
        eval_yolo(args)
