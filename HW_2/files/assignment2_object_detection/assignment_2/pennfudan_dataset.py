"""
PyTorch Dataset for the Penn-Fudan Pedestrian Dataset.

Directory structure expected (created by download_pennfudan.py):
    data/pennfudan/
        train/
            images/   *.png
            masks/    *_mask.png
        val/
            images/
            masks/
        test/
            images/
            masks/

Each mask PNG encodes instance IDs as pixel values (1, 2, 3 … for each
pedestrian instance).  This class converts masks to bounding boxes.

Label index: 1 = person  (0 is background, consistent with Faster R-CNN convention)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from transforms import pil_to_numpy, apply_transforms, get_train_transforms, get_val_transforms


PERSON_LABEL = 1   # background = 0


class PennFudanDataset(Dataset):
    """
    Penn-Fudan Pedestrian detection dataset.

    Args:
        root      : path to split directory, e.g. data/pennfudan/train
        split     : 'train', 'val', or 'test' (used only for selecting transforms)
        image_size: resize target (default 512)
        transform : optional albumentations Compose (overrides default)
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 512,
        transform=None,
    ):
        self.root       = Path(root)
        self.split      = split
        self.image_size = image_size

        self.img_dir  = self.root / "images"
        self.mask_dir = self.root / "masks"

        self.imgs = sorted(self.img_dir.glob("*.png"))

        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.imgs)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path  = self.imgs[idx]
        mask_path = self.mask_dir / img_path.name.replace(".png", "_mask.png")

        # Load image
        image_pil = Image.open(img_path).convert("RGB")
        image_np  = pil_to_numpy(image_pil)

        # Load mask and derive bounding boxes
        boxes, labels = self._mask_to_boxes(mask_path)

        # Apply transforms
        img_tensor, boxes_tensor, labels_tensor = apply_transforms(
            image_np, boxes, labels, self.transform
        )

        target = {
            "boxes":    boxes_tensor,          # Nx4  float32  (xmin,ymin,xmax,ymax)
            "labels":   labels_tensor,          # N    int64
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target

    # ------------------------------------------------------------------
    def _mask_to_boxes(self, mask_path: Path):
        """Extract per-instance bounding boxes from a mask image."""
        if not mask_path.exists():
            return [], []

        mask = np.array(Image.open(mask_path))

        # Each unique non-zero value corresponds to one pedestrian instance
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        boxes  = []
        labels = []

        for obj_id in obj_ids:
            ys, xs = np.where(mask == obj_id)
            if len(xs) == 0:
                continue
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            if xmax > xmin and ymax > ymin:          # skip degenerate boxes
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(PERSON_LABEL)

        return boxes, labels

    # ------------------------------------------------------------------
    @staticmethod
    def num_classes() -> int:
        """Number of classes INCLUDING background (index 0)."""
        return 2   # background(0) + person(1)

    @staticmethod
    def class_names() -> list:
        return ["__background__", "person"]
