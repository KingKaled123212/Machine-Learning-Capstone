"""
Evaluation metrics for object detection.

Computes:
  - mAP@0.5  (COCO-style via pycocotools)
  - Precision and Recall at IoU=0.5
  - Per-class AP

Main entry point:
  evaluate_coco(model, data_loader, device) → MetricResults
"""

import json
import time
import copy
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MetricResults:
    map50:      float = 0.0
    map50_95:   float = 0.0
    precision:  float = 0.0
    recall:     float = 0.0
    per_class_ap: dict = field(default_factory=dict)
    inference_speed: float = 0.0   # images / second

    def __str__(self):
        lines = [
            f"mAP@0.5    : {self.map50:.4f}",
            f"mAP@0.5:95 : {self.map50_95:.4f}",
            f"Precision  : {self.precision:.4f}",
            f"Recall     : {self.recall:.4f}",
        ]
        if self.inference_speed > 0:
            lines.append(f"Speed      : {self.inference_speed:.1f} img/s")
        if self.per_class_ap:
            lines.append("Per-class AP@0.5:")
            for cls, ap in self.per_class_ap.items():
                lines.append(f"  {cls:30s}: {ap:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_coco(
    model,
    data_loader,
    device: torch.device,
    class_names: list = None,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.05,
) -> MetricResults:
    """
    Run inference and compute COCO mAP metrics.

    Args:
        model          : Faster R-CNN in eval mode (or any model returning
                         list of dicts with 'boxes', 'labels', 'scores')
        data_loader    : DataLoader yielding (images, targets)
        device         : torch.device
        class_names    : optional list of class name strings (index = label)
        iou_threshold  : IoU threshold for a detection to count as TP
        conf_threshold : minimum score to keep a prediction

    Returns:
        MetricResults
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        USE_COCO = True
    except ImportError:
        USE_COCO = False

    model.eval()

    all_predictions = []   # [{image_id, boxes, labels, scores}]
    all_targets     = []   # [{image_id, boxes, labels}]
    inference_times = []

    for images, targets in data_loader:
        images = [img.to(device) for img in images]

        t0 = time.perf_counter()
        predictions = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        inference_times.append(elapsed / len(images))

        for pred, tgt in zip(predictions, targets):
            image_id = tgt["image_id"].item()

            # Filter low-confidence predictions
            keep = pred["scores"] >= conf_threshold
            all_predictions.append({
                "image_id": image_id,
                "boxes":    pred["boxes"][keep].cpu().numpy(),
                "labels":   pred["labels"][keep].cpu().numpy(),
                "scores":   pred["scores"][keep].cpu().numpy(),
            })
            all_targets.append({
                "image_id": image_id,
                "boxes":    tgt["boxes"].cpu().numpy(),
                "labels":   tgt["labels"].cpu().numpy(),
            })

    speed = 1.0 / (sum(inference_times) / len(inference_times)) if inference_times else 0.0

    if USE_COCO:
        results = _coco_eval(all_predictions, all_targets, class_names)
    else:
        results = _simple_eval(all_predictions, all_targets, iou_threshold, class_names)

    results.inference_speed = speed
    return results


# ---------------------------------------------------------------------------
# COCO-API evaluation
# ---------------------------------------------------------------------------

def _coco_eval(predictions, targets, class_names=None) -> MetricResults:
    """Build COCO-format dicts and run COCOeval."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import io, contextlib

    # Collect all class ids
    all_labels = set()
    for t in targets:
        all_labels.update(t["labels"].tolist())
    for p in predictions:
        all_labels.update(p["labels"].tolist())
    all_labels = sorted(all_labels)

    # Build GT COCO dataset
    coco_gt_data = {
        "images":      [],
        "annotations": [],
        "categories":  [],
    }
    for lbl in all_labels:
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        coco_gt_data["categories"].append({"id": lbl, "name": name})

    ann_id = 1
    for t in targets:
        img_id = int(t["image_id"])
        coco_gt_data["images"].append({"id": img_id})
        for box, lbl in zip(t["boxes"], t["labels"]):
            x1, y1, x2, y2 = box.tolist()
            w, h = x2 - x1, y2 - y1
            coco_gt_data["annotations"].append({
                "id":           ann_id,
                "image_id":     img_id,
                "category_id":  int(lbl),
                "bbox":         [x1, y1, w, h],
                "area":         w * h,
                "iscrowd":      0,
            })
            ann_id += 1

    # Load GT
    coco_gt = COCO()
    coco_gt.dataset = coco_gt_data
    coco_gt.createIndex()

    # Build predictions list
    coco_preds = []
    for p in predictions:
        img_id = int(p["image_id"])
        for box, lbl, score in zip(p["boxes"], p["labels"], p["scores"]):
            x1, y1, x2, y2 = box.tolist()
            coco_preds.append({
                "image_id":    img_id,
                "category_id": int(lbl),
                "bbox":        [x1, y1, x2 - x1, y2 - y1],
                "score":       float(score),
            })

    if not coco_preds:
        return MetricResults()

    coco_dt = coco_gt.loadRes(coco_preds)

    # Run eval — suppress stdout
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats   # [mAP@0.5:0.95, mAP@0.5, ...]

    # Per-class AP at IoU=0.5
    per_class_ap = {}
    for lbl in all_labels:
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.iouThrs = np.array([0.5])
        ev.params.catIds  = [lbl]
        with contextlib.redirect_stdout(io.StringIO()):
            ev.evaluate(); ev.accumulate(); ev.summarize()
        per_class_ap[name] = float(ev.stats[1])

    # Precision / recall at IoU=0.5 (index 1 of summarize)
    prec   = float(coco_eval.stats[1])   # AP@0.5 as proxy for precision
    recall = float(coco_eval.stats[8])   # AR@0.5 (max dets=10)

    return MetricResults(
        map50=float(coco_eval.stats[1]),
        map50_95=float(coco_eval.stats[0]),
        precision=prec,
        recall=recall,
        per_class_ap=per_class_ap,
    )


# ---------------------------------------------------------------------------
# Fallback simple evaluation (no pycocotools)
# ---------------------------------------------------------------------------

def _simple_eval(predictions, targets, iou_threshold=0.5, class_names=None) -> MetricResults:
    """Simple Precision/Recall/mAP without COCO API."""
    all_labels = set()
    for t in targets:
        all_labels.update(t["labels"].tolist())
    all_labels = sorted(all_labels)

    per_class_ap = {}
    all_precisions, all_recalls = [], []

    for lbl in all_labels:
        tp_list, fp_list, scores_list = [], [], []
        n_gt = 0

        for pred, tgt in zip(predictions, targets):
            gt_boxes   = tgt["boxes"][tgt["labels"] == lbl]
            pred_mask  = pred["labels"] == lbl
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores= pred["scores"][pred_mask]

            n_gt += len(gt_boxes)
            matched = np.zeros(len(gt_boxes), dtype=bool)

            # Sort by score descending
            order = np.argsort(-pred_scores)
            for idx in order:
                pb  = pred_boxes[idx]
                iou = _iou_batch(pb[None], gt_boxes)

                if len(iou) == 0:
                    fp_list.append(1); tp_list.append(0)
                    scores_list.append(pred_scores[idx])
                    continue

                best_iou_idx = iou.argmax()
                if iou[0, best_iou_idx] >= iou_threshold and not matched[best_iou_idx]:
                    tp_list.append(1); fp_list.append(0)
                    matched[best_iou_idx] = True
                else:
                    tp_list.append(0); fp_list.append(1)
                scores_list.append(pred_scores[idx])

        if not tp_list:
            per_class_ap[str(lbl)] = 0.0
            continue

        order = np.argsort(-np.array(scores_list))
        tp_cum = np.cumsum(np.array(tp_list)[order])
        fp_cum = np.cumsum(np.array(fp_list)[order])

        recalls    = tp_cum / (n_gt + 1e-9)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

        ap = _compute_ap(recalls, precisions)
        name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
        per_class_ap[name] = float(ap)
        all_precisions.append(precisions[-1] if len(precisions) else 0.0)
        all_recalls.append(recalls[-1] if len(recalls) else 0.0)

    map50 = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0
    prec  = float(np.mean(all_precisions)) if all_precisions else 0.0
    rec   = float(np.mean(all_recalls)) if all_recalls else 0.0

    return MetricResults(map50=map50, map50_95=0.0, precision=prec, recall=rec, per_class_ap=per_class_ap)


def _iou_batch(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if len(boxes2) == 0:
        return np.zeros((len(boxes1), 0))
    inter_x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0])
    inter_y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1])
    inter_x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2])
    inter_y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3])
    inter_w  = np.maximum(0, inter_x2 - inter_x1)
    inter_h  = np.maximum(0, inter_y2 - inter_y1)
    inter    = inter_w * inter_h
    area1    = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2    = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union    = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def _compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """11-point interpolated AP."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p = precision[recall >= thr]
        ap += (p.max() if len(p) else 0.0) / 11.0
    return ap
