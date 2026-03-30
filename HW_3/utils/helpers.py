"""
utils/helpers.py
Shared utility functions for the NLP assignment.
"""

import os
import time
import math
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Count the total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    """Calculate elapsed time in minutes and seconds."""
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - mins * 60)
    return mins, secs


def plot_training_curves(train_losses, val_losses, title, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
    plt.close()


def plot_comparison_bar(results: dict, metric: str, title: str, save_path=None):
    """
    Plot a grouped bar chart comparing multiple model/embedding combinations.

    results: {
        "Model A - GloVe": 45.2,
        "Model A - OneHot": 38.1,
        ...
    }
    """
    labels = list(results.keys())
    values = list(results.values())

    colors = sns.color_palette("Set2", len(labels))
    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=0.8)
    plt.ylabel(metric)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {save_path}")
    plt.close()


def save_results(results: dict, path: str):
    """Save a results dictionary to a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for k, v in results.items():
            f.write(f"{k}: {v}\n")
    print(f"  Results saved to {path}")
