"""
Data augmentation transforms for object detection.

Provides:
  - get_train_transforms(image_size)   → albumentations Compose
  - get_val_transforms(image_size)     → albumentations Compose
  - apply_transforms(image, boxes, labels, transform)
  - ToTensor / collate_fn for DataLoader
"""

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Augmentation pipeline for training.
    bbox_params ensure bounding boxes are transformed alongside the image.
    """
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=0, value=(114, 114, 114),
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=5,
                border_mode=0, p=0.3,
            ),
            A.RandomSizedBBoxSafeCrop(
                height=image_size, width=image_size, erosion_rate=0.0, p=0.3,
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",       # (xmin, ymin, xmax, ymax) in pixels
            label_fields=["labels"],
            min_area=64,               # drop tiny boxes that survive crop
            min_visibility=0.3,
        ),
    )


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """Deterministic resize-and-pad only (no augmentation)."""
    return A.Compose(
        [
            A.LongestMaxSize(max_size=image_size),
            A.PadIfNeeded(
                min_height=image_size, min_width=image_size,
                border_mode=0, value=(114, 114, 114),
            ),
            A.ToFloat(max_value=255.0),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["labels"],
            min_area=64,
            min_visibility=0.3,
        ),
    )


# ---------------------------------------------------------------------------
# Helper: apply transform to image + boxes
# ---------------------------------------------------------------------------

def apply_transforms(
    image: np.ndarray,
    boxes: list,
    labels: list,
    transform: A.Compose,
):
    """
    Args:
        image   : HxWxC uint8 numpy array
        boxes   : list of [xmin, ymin, xmax, ymax] in pixel coords
        labels  : list of int class indices (same length as boxes)
        transform: albumentations Compose with bbox_params

    Returns:
        image_tensor : CxHxW float32 tensor  (values in [0, 1])
        boxes_tensor : Nx4 float32 tensor    (Pascal VOC pixel coords)
        labels_tensor: N   int64  tensor
    """
    transformed = transform(image=image, bboxes=boxes, labels=labels)

    img_t    = transformed["image"]          # already ToTensorV2 → CxHxW float
    bboxes_t = transformed["bboxes"]
    labels_t = transformed["labels"]

    if len(bboxes_t) == 0:
        boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
        labels_tensor = torch.zeros((0,),   dtype=torch.int64)
    else:
        boxes_tensor  = torch.as_tensor(bboxes_t, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_t, dtype=torch.int64)

    return img_t, boxes_tensor, labels_tensor


# ---------------------------------------------------------------------------
# Collate function (handles variable number of boxes per image)
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """
    Custom collate for detection dataloaders.
    Returns:
        images : list of CxHxW tensors
        targets: list of dicts with keys 'boxes' and 'labels'
    """
    images  = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# ---------------------------------------------------------------------------
# PIL → numpy helper
# ---------------------------------------------------------------------------

def pil_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to RGB uint8 numpy array."""
    return np.array(image.convert("RGB"), dtype=np.uint8)
