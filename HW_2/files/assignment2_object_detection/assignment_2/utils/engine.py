"""
Training and evaluation engine for Faster R-CNN.

Provides:
  train_one_epoch(model, optimizer, data_loader, device, scaler) → dict
  evaluate_one_epoch(model, data_loader, device) → dict  (loss-only, no mAP)
  warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
"""

import math
import time
import sys
import torch
from torch.cuda.amp import autocast, GradScaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device: torch.device,
    epoch: int,
    scaler: GradScaler = None,
    print_freq: int = 20,
) -> dict:
    """
    Train for one epoch.

    Args:
        model       : Faster R-CNN (already on device)
        optimizer   : torch.optim instance
        data_loader : DataLoader yielding (images, targets) batches
        device      : 'cuda' or 'cpu'
        epoch       : current epoch number (for logging)
        scaler      : GradScaler for mixed precision (None = fp32)
        print_freq  : log every print_freq batches

    Returns:
        dict with keys: loss, loss_classifier, loss_box_reg,
                        loss_objectness, loss_rpn_box_reg, lr
    """
    model.train()
    use_amp = scaler is not None

    metric_logger = _MetricLogger()
    metric_logger.add_meter("lr", _SmoothedValue(window_size=1, fmt="{value:.6f}"))

    header = f"Epoch [{epoch}]"
    total_batches = len(data_loader)

    for i, (images, targets) in enumerate(data_loader):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast(enabled=use_amp):
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        # Sanity check
        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if use_amp:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        # Log
        metric_logger.update(loss=loss_value, **{k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if (i + 1) % print_freq == 0 or (i + 1) == total_batches:
            print(
                f"  {header} [{i+1:4d}/{total_batches}]  "
                f"loss: {loss_value:.4f}  "
                f"lr: {optimizer.param_groups[0]['lr']:.6f}"
            )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Validation loss (no mAP — mAP is computed by metrics.py with COCO API)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_loss(model, data_loader, device: torch.device) -> dict:
    """
    Compute average losses on a validation set.
    (Faster R-CNN returns losses only when targets are provided.)
    """
    model.train()    # keep in train mode to get losses
    metric_logger = _MetricLogger()

    for images, targets in data_loader:
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss_dict.values())
        metric_logger.update(loss=losses.item(), **{k: v.item() for k, v in loss_dict.items()})

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# LR warm-up scheduler
# ---------------------------------------------------------------------------

def warmup_lr_scheduler(optimizer, warmup_iters: int, warmup_factor: float = 1.0 / 1000):
    """
    Linear warm-up scheduler.  Wraps a torch.optim.Optimizer.
    Returns a LambdaLR scheduler that should be stepped every training batch
    for the first warmup_iters steps.
    """
    def f(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ---------------------------------------------------------------------------
# Helper: measure inference throughput
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_inference_speed(
    model,
    data_loader,
    device: torch.device,
    n_batches: int = 50,
) -> float:
    """
    Returns mean images/second over n_batches inference calls (batch size 1).
    """
    model.eval()
    times = []

    for i, (images, _) in enumerate(data_loader):
        if i >= n_batches:
            break
        images = [img.to(device) for img in images]

        # Warm-up on first call
        if i == 0:
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

        t0 = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        times.append((t1 - t0) / len(images))   # seconds per image

    mean_s_per_img = sum(times) / len(times)
    return 1.0 / mean_s_per_img                  # images per second


# ---------------------------------------------------------------------------
# Minimal metric logger
# ---------------------------------------------------------------------------

class _SmoothedValue:
    def __init__(self, window_size=20, fmt=None):
        from collections import deque
        self.deque  = deque(maxlen=window_size)
        self.total  = 0.0
        self.count  = 0
        self.fmt    = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        import torch as _t
        d = _t.tensor(list(self.deque))
        return d.median().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            global_avg=self.global_avg,
            value=self.deque[-1] if self.deque else 0.0,
        )


class _MetricLogger:
    def __init__(self):
        from collections import defaultdict
        self.meters = defaultdict(lambda: _SmoothedValue())

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.meters[k].update(v)

    def add_meter(self, name, meter):
        self.meters[name] = meter
