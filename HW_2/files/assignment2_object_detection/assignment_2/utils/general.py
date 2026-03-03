"""
General utilities: seeding, device selection, checkpoint saving/loading,
logging, early stopping, and memory monitoring.
"""

import os
import json
import time
import random
import logging
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(preferred: str = "cuda") -> torch.device:
    """Return best available device."""
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda")
        props = torch.cuda.get_device_properties(dev)
        print(f"Using GPU: {props.name}  ({props.total_memory / 1e9:.1f} GB VRAM)")
        return dev
    print("CUDA not available, using CPU.")
    return torch.device("cpu")


def print_gpu_memory() -> None:
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    alloc   = torch.cuda.memory_allocated()  / 1e9
    reserved= torch.cuda.memory_reserved()   / 1e9
    total   = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory: {alloc:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    save_path: str | Path,
    is_best: bool = False,
) -> None:
    """Save model checkpoint to disk."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "scheduler":  scheduler.state_dict() if scheduler is not None else None,
        "metrics":    metrics,
        "timestamp":  datetime.now().isoformat(),
    }
    torch.save(checkpoint, save_path)

    if is_best:
        best_path = save_path.parent / "best_model.pth"
        torch.save(checkpoint, best_path)
        print(f"  → New best model saved to {best_path}")


def load_checkpoint(
    path: str | Path,
    model,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
) -> dict:
    """Load a checkpoint. Returns the checkpoint dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    map_loc = device if device is not None else "cpu"
    ckpt = torch.load(path, map_location=map_loc)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])

    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}  |  {path}")
    return ckpt


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str, log_dir: str | Path = None, level=logging.INFO) -> logging.Logger:
    """Set up a logger that writes to stdout and optionally a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(Path(log_dir) / f"{name}_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------

class TrainingHistory:
    """Accumulates per-epoch metrics and persists to JSON."""

    def __init__(self, save_path: str | Path = None):
        self.data: dict[str, list] = {}
        self.save_path = Path(save_path) if save_path else None

    def update(self, **kwargs) -> None:
        for k, v in kwargs.items():
            self.data.setdefault(k, []).append(float(v) if v is not None else 0.0)

    def save(self) -> None:
        if self.save_path:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            self.save_path.write_text(json.dumps(self.data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TrainingHistory":
        h = cls(save_path=path)
        p = Path(path)
        if p.exists():
            h.data = json.loads(p.read_text())
        return h


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Args:
        patience : epochs to wait after last improvement
        mode     : 'max' (higher is better) or 'min' (lower is better)
        delta    : minimum change to qualify as improvement
    """

    def __init__(self, patience: int = 5, mode: str = "max", delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.delta     = delta
        self.counter   = 0
        self.best      = None
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """Call once per epoch. Returns True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        improved = (
            (self.mode == "max" and value >= self.best + self.delta) or
            (self.mode == "min" and value <= self.best - self.delta)
        )

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best:.4f})")
            if self.counter >= self.patience:
                self.should_stop = True
                print("  Early stopping triggered.")

        return self.should_stop


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self):
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def elapsed_str(self) -> str:
        s = self.elapsed
        m, s2 = divmod(int(s), 60)
        h, m2 = divmod(m, 60)
        return f"{h:02d}:{m2:02d}:{s2:02d}" if h else f"{m2:02d}:{s2:02d}"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
