"""
Visualisation helpers for object detection results.

Provides:
  visualize_predictions(image, boxes, labels, scores, class_names, ...)
  visualize_dataset_sample(dataset, idx, class_names)
  plot_training_curves(history, save_path)
  plot_comparison_bar(results_dict, save_path)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from pathlib import Path
from PIL import Image


# Colour palette (one per class, repeats if needed)
_PALETTE = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#264653", "#A8DADC", "#F1FAEE", "#06D6A0", "#118AB2",
]


def _get_color(label_idx: int) -> str:
    return _PALETTE[label_idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Draw boxes on image
# ---------------------------------------------------------------------------

def visualize_predictions(
    image,
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray = None,
    class_names: list = None,
    gt_boxes: np.ndarray = None,
    gt_labels: np.ndarray = None,
    conf_threshold: float = 0.3,
    title: str = "",
    save_path: str = None,
    show: bool = False,
) -> plt.Figure:
    """
    Draw predicted (and optionally ground-truth) bounding boxes on an image.

    Args:
        image         : PIL Image, numpy HxWxC uint8, or CxHxW float tensor
        boxes         : Nx4 numpy array (xmin, ymin, xmax, ymax)
        labels        : N   numpy int array
        scores        : N   numpy float array (optional)
        class_names   : list of class name strings
        gt_boxes      : ground-truth boxes (drawn in dashed green if provided)
        gt_labels     : ground-truth labels
        conf_threshold: hide predictions below this score
        title         : figure title
        save_path     : save figure to this path (optional)
        show          : call plt.show() (requires display)

    Returns:
        matplotlib Figure
    """
    # Convert image to numpy HxWxC uint8
    if isinstance(image, torch.Tensor):
        if image.dtype == torch.float32:
            image = (image * 255).byte()
        image = image.permute(1, 2, 0).cpu().numpy()
    elif isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)

    # Draw GT boxes (dashed green)
    if gt_boxes is not None and len(gt_boxes):
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            lbl = gt_labels[i] if gt_labels is not None else 0
            color = "#00C853"
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none", linestyle="--",
            )
            ax.add_patch(rect)
            gt_name = (class_names[lbl] if class_names and lbl < len(class_names) else str(lbl))
            ax.text(x1, y1 - 4, f"GT: {gt_name}", fontsize=8, color=color,
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    # Draw prediction boxes
    if boxes is not None and len(boxes):
        for i, box in enumerate(boxes):
            score = float(scores[i]) if scores is not None else 1.0
            if score < conf_threshold:
                continue
            lbl   = int(labels[i]) if labels is not None else 0
            color = _get_color(lbl)
            x1, y1, x2, y2 = box
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor="none",
            )
            ax.add_patch(rect)
            name = class_names[lbl] if class_names and lbl < len(class_names) else str(lbl)
            label_txt = f"{name} {score:.2f}" if scores is not None else name
            ax.text(x1, y1 - 4, label_txt, fontsize=9, color="white",
                    bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Dataset sample viewer
# ---------------------------------------------------------------------------

def visualize_dataset_sample(
    dataset,
    idx: int,
    class_names: list = None,
    save_path: str = None,
) -> plt.Figure:
    """Draw a single labelled sample from a PyTorch Dataset."""
    img, target = dataset[idx]
    boxes  = target["boxes"].numpy()
    labels = target["labels"].numpy()
    return visualize_predictions(
        image=img,
        boxes=boxes,
        labels=labels,
        class_names=class_names,
        title=f"Sample #{idx}",
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# Training curve plot
# ---------------------------------------------------------------------------

def plot_training_curves(
    history: dict,
    save_path: str = "outputs/training_curves.png",
    title: str = "Training Curves",
) -> plt.Figure:
    """
    Plot loss and mAP curves across epochs.

    Args:
        history : dict with keys like 'train_loss', 'val_loss', 'val_map50'
        save_path: where to save the figure
        title    : figure title
    """
    keys   = [k for k in history if len(history[k]) > 0]
    epochs = range(1, len(history[keys[0]]) + 1)

    n_plots = sum([
        any("loss" in k for k in keys),
        any("map" in k or "map50" in k for k in keys),
    ])
    fig, axes = plt.subplots(1, max(n_plots, 1), figsize=(6 * max(n_plots, 1), 4))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Loss curves
    loss_keys = [k for k in keys if "loss" in k.lower()]
    if loss_keys:
        ax = axes[plot_idx]
        for k in loss_keys:
            ax.plot(epochs, history[k], label=k.replace("_", " ").title())
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        plot_idx += 1

    # mAP / metric curves
    metric_keys = [k for k in keys if "map" in k.lower() or "precision" in k.lower() or "recall" in k.lower()]
    if metric_keys:
        ax = axes[plot_idx]
        for k in metric_keys:
            ax.plot(epochs, history[k], label=k.replace("_", " ").upper())
        ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_title("Metrics")
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved training curves to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    results: dict,
    metrics: list = ("map50", "precision", "recall"),
    save_path: str = "outputs/comparison.png",
    title: str = "Model Comparison",
) -> plt.Figure:
    """
    Bar chart comparing multiple models across metrics.

    Args:
        results : {model_name: MetricResults or dict}
        metrics : list of metric attribute names to plot
        save_path, title: self-explanatory
    """
    model_names = list(results.keys())
    n_metrics   = len(metrics)
    x           = np.arange(len(model_names))
    width       = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(8, len(model_names) * 2.5), 5))

    colors = ["#4472C4", "#ED7D31", "#A9D18E", "#FFC000", "#5B9BD5"]

    for i, metric in enumerate(metrics):
        vals = []
        for name in model_names:
            r = results[name]
            v = getattr(r, metric, None) if hasattr(r, metric) else r.get(metric, 0.0)
            vals.append(float(v) if v is not None else 0.0)

        bars = ax.bar(
            x + i * width - (n_metrics - 1) * width / 2,
            vals,
            width,
            label=metric.upper().replace("_", " "),
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.8,
        )

        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved comparison chart to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Speed vs accuracy scatter
# ---------------------------------------------------------------------------

def plot_speed_accuracy_scatter(
    results: dict,
    save_path: str = "outputs/speed_vs_accuracy.png",
) -> plt.Figure:
    """
    Scatter plot of inference speed (x) vs mAP@0.5 (y).

    Args:
        results : {label: MetricResults}  (label = e.g. "Faster R-CNN (Penn-Fudan)")
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.cm.tab10.colors

    for i, (label, r) in enumerate(results.items()):
        speed = getattr(r, "inference_speed", 0.0) or r.get("inference_speed", 0)
        map50 = getattr(r, "map50", 0.0) or r.get("map50", 0)
        ax.scatter(speed, map50, s=120, color=colors[i % 10], zorder=3, label=label)
        ax.annotate(
            label,
            xy=(speed, map50),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Inference Speed (images/second)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Speed vs Accuracy Trade-off", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Saved speed-accuracy plot to {save_path}")

    return fig
