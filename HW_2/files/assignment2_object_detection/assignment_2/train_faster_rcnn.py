"""
Train Faster R-CNN (MobileNetV3-Large FPN) on Penn-Fudan or Oxford Pets.

Usage:
    python train_faster_rcnn.py --dataset pennfudan --epochs 15
    python train_faster_rcnn.py --dataset oxford_pets --epochs 20 --batch-size 2
    python train_faster_rcnn.py --dataset pennfudan --resume outputs/faster_rcnn/pennfudan/last.pth
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from pathlib import Path

from transforms import collate_fn
from faster_rcnn import build_faster_rcnn, print_model_summary
from utils.engine import train_one_epoch, evaluate_loss, warmup_lr_scheduler, measure_inference_speed
from utils.metrics import evaluate_coco
from utils.visualize import plot_training_curves, visualize_predictions
from utils.general import (
    set_seed, get_device, print_gpu_memory,
    save_checkpoint, load_checkpoint,
    setup_logger, TrainingHistory, EarlyStopping, Timer, load_config
)


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def get_datasets(dataset_name: str, image_size: int = 512):
    if dataset_name == "pennfudan":
        from pennfudan_dataset import PennFudanDataset
        root = Path("data/pennfudan")
        train_ds = PennFudanDataset(root / "train", split="train", image_size=image_size)
        val_ds   = PennFudanDataset(root / "val",   split="val",   image_size=image_size)
        test_ds  = PennFudanDataset(root / "test",  split="test",  image_size=image_size)
        num_classes = PennFudanDataset.num_classes()
        class_names = PennFudanDataset.class_names()

    elif dataset_name == "oxford_pets":
        from oxford_pets_dataset import OxfordPetsDataset
        root = Path("data/oxford_pets")
        train_ds, class_to_idx = OxfordPetsDataset.from_directory(root, "train", image_size)
        val_ds,   _            = OxfordPetsDataset.from_directory(root, "val",   image_size)
        val_ds.class_to_idx    = class_to_idx
        test_ds,  _            = OxfordPetsDataset.from_directory(root, "test",  image_size)
        test_ds.class_to_idx   = class_to_idx
        num_classes = train_ds.num_classes
        class_names = train_ds.class_names

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose 'pennfudan' or 'oxford_pets'.")

    return train_ds, val_ds, test_ds, num_classes, class_names


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    set_seed(42)
    device = get_device()
    logger = setup_logger("faster_rcnn", log_dir=f"outputs/faster_rcnn/{args.dataset}/logs")

    logger.info(f"{'='*60}")
    logger.info(f"Faster R-CNN Training  |  Dataset: {args.dataset}  |  Epochs: {args.epochs}")
    logger.info(f"{'='*60}")

    # --- Data ---
    logger.info("Loading datasets …")
    train_ds, val_ds, test_ds, num_classes, class_names = get_datasets(args.dataset, args.image_size)
    logger.info(f"  Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    logger.info(f"  Classes ({num_classes}): {class_names}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # --- Model ---
    logger.info("Building model …")
    model = build_faster_rcnn(
        num_classes=num_classes,
        pretrained=True,
        trainable_backbone_layers=args.trainable_layers,
        image_size=args.image_size,
    ).to(device)
    print_model_summary(model)
    print_gpu_memory()

    # --- Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    scaler    = GradScaler() if args.amp else None

    # Warm-up (first epoch)
    warmup_iters = min(500, len(train_loader) - 1)
    warmup_sched = warmup_lr_scheduler(optimizer, warmup_iters)

    early_stopping = EarlyStopping(patience=args.patience, mode="max")
    history        = TrainingHistory(f"outputs/faster_rcnn/{args.dataset}/history.json")

    start_epoch = 0

    # --- Resume ---
    if args.resume:
        ckpt = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    # --- Training loop ---
    save_dir = Path(f"outputs/faster_rcnn/{args.dataset}")
    save_dir.mkdir(parents=True, exist_ok=True)

    best_map = 0.0

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        with Timer() as t_train:
            train_metrics = train_one_epoch(
                model, optimizer, train_loader, device, epoch + 1, scaler,
                print_freq=10,
            )

        # Step warm-up only in epoch 0
        if epoch == 0:
            warmup_sched.step()
        scheduler.step()

        logger.info(f"Train loss: {train_metrics['loss']:.4f}  [{t_train.elapsed_str()}]")

        # --- Validation mAP ---
        with Timer() as t_val:
            val_results = evaluate_coco(model, val_loader, device, class_names)

        logger.info(f"Val   mAP@0.5: {val_results.map50:.4f}  "
                    f"P: {val_results.precision:.4f}  "
                    f"R: {val_results.recall:.4f}  [{t_val.elapsed_str()}]")

        # --- History & checkpoint ---
        history.update(
            train_loss=train_metrics["loss"],
            val_map50=val_results.map50,
            val_precision=val_results.precision,
            val_recall=val_results.recall,
        )
        history.save()

        is_best = val_results.map50 > best_map
        if is_best:
            best_map = val_results.map50
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {"map50": val_results.map50},
            save_dir / "last.pth",
            is_best=is_best,
        )

        if early_stopping(val_results.map50):
            logger.info("Early stopping.")
            break

    # --- Final test evaluation ---
    logger.info("\n=== Final Test Evaluation ===")
    best_ckpt_path = save_dir / "best_model.pth"
    if best_ckpt_path.exists():
        load_checkpoint(best_ckpt_path, model, device=device)

    test_results = evaluate_coco(model, test_loader, device, class_names)
    speed        = measure_inference_speed(model, test_loader, device, n_batches=50)
    test_results.inference_speed = speed

    logger.info(f"\nTest Results:\n{test_results}")

    # Save test results
    import json
    (save_dir / "test_results.json").write_text(json.dumps({
        "map50":           test_results.map50,
        "map50_95":        test_results.map50_95,
        "precision":       test_results.precision,
        "recall":          test_results.recall,
        "inference_speed": test_results.inference_speed,
        "per_class_ap":    test_results.per_class_ap,
    }, indent=2))

    # --- Plots ---
    plot_training_curves(
        history.data,
        save_path=str(save_dir / "training_curves.png"),
        title=f"Faster R-CNN — {args.dataset}",
    )

    # --- Sample predictions ---
    _save_sample_predictions(model, test_ds, device, class_names, save_dir)

    logger.info(f"\nAll outputs saved to {save_dir.resolve()}")
    return test_results


# ---------------------------------------------------------------------------
# Save a few sample prediction images
# ---------------------------------------------------------------------------

def _save_sample_predictions(model, dataset, device, class_names, save_dir, n=6):
    model.eval()
    (save_dir / "predictions").mkdir(exist_ok=True)
    import random
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for idx in indices:
        img, target = dataset[idx]
        with torch.no_grad():
            pred = model([img.to(device)])[0]

        from utils.visualize import visualize_predictions
        visualize_predictions(
            image=img,
            boxes=pred["boxes"].cpu().numpy(),
            labels=pred["labels"].cpu().numpy(),
            scores=pred["scores"].cpu().numpy(),
            class_names=class_names,
            gt_boxes=target["boxes"].numpy(),
            gt_labels=target["labels"].numpy(),
            conf_threshold=0.3,
            title=f"Sample {idx}",
            save_path=str(save_dir / "predictions" / f"sample_{idx:04d}.png"),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train Faster R-CNN")
    p.add_argument("--dataset",          type=str, default="pennfudan",
                   choices=["pennfudan", "oxford_pets"])
    p.add_argument("--epochs",           type=int, default=15)
    p.add_argument("--batch-size",       type=int, default=4)
    p.add_argument("--lr",               type=float, default=0.005)
    p.add_argument("--image-size",       type=int, default=512)
    p.add_argument("--trainable-layers", type=int, default=3)
    p.add_argument("--num-workers",      type=int, default=4)
    p.add_argument("--patience",         type=int, default=5)
    p.add_argument("--amp",              action="store_true", default=True,
                   help="Use mixed precision (fp16). Default: on.")
    p.add_argument("--no-amp",           dest="amp", action="store_false")
    p.add_argument("--resume",           type=str, default=None,
                   help="Path to checkpoint to resume from.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
