"""
PyTorch Dataset for the Oxford-IIIT Pet subset.

Directory structure expected (created by download_oxford_pets.py):
    data/oxford_pets/
        train/
            images/       *.jpg
            annotations/  *.xml   (Pascal VOC format)
        val/  …
        test/ …
        dataset.yaml      (class names + paths for YOLO)

Labels start at 1 (0 = background), consistent with Faster R-CNN convention.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

from transforms import pil_to_numpy, apply_transforms, get_train_transforms, get_val_transforms


class OxfordPetsDataset(Dataset):
    """
    Oxford-IIIT Pet detection dataset (subset).

    Args:
        root         : path to split directory, e.g. data/oxford_pets/train
        class_to_idx : dict mapping breed name → int index (1-based, 0 = bg)
        split        : 'train', 'val', or 'test'
        image_size   : resize target (default 512)
        transform    : optional albumentations Compose (overrides default)
    """

    def __init__(
        self,
        root: str | Path,
        class_to_idx: dict,
        split: str = "train",
        image_size: int = 512,
        transform=None,
    ):
        self.root         = Path(root)
        self.class_to_idx = class_to_idx          # breed_name → 1-based int
        self.split        = split
        self.image_size   = image_size

        self.img_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"

        self.imgs = sorted(self.img_dir.glob("*.jpg"))

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
        img_path = self.imgs[idx]

        image_pil = Image.open(img_path).convert("RGB")
        image_np  = pil_to_numpy(image_pil)

        boxes, labels = self._parse_annotation(img_path)

        img_tensor, boxes_tensor, labels_tensor = apply_transforms(
            image_np, boxes, labels, self.transform
        )

        target = {
            "boxes":    boxes_tensor,
            "labels":   labels_tensor,
            "image_id": torch.tensor([idx]),
        }

        return img_tensor, target

    # ------------------------------------------------------------------
    def _parse_annotation(self, img_path: Path):
        """
        Parse Pascal VOC XML for this image.
        Returns (boxes, labels) lists.

        If no XML exists, fall back to a whole-image box with the breed label
        inferred from the filename.
        """
        xml_path = self.ann_dir / (img_path.stem + ".xml")

        if xml_path.exists():
            return self._parse_voc_xml(xml_path, img_path)
        else:
            return self._fallback_box(img_path)

    def _parse_voc_xml(self, xml_path: Path, img_path: Path):
        tree  = ET.parse(xml_path)
        root  = tree.getroot()

        # Infer breed from filename (everything before the last _NNN)
        breed = img_path.stem.rsplit("_", 1)[0]
        label = self.class_to_idx.get(breed)
        if label is None:
            # Try case-insensitive match
            for k, v in self.class_to_idx.items():
                if k.lower() == breed.lower():
                    label = v
                    break
        if label is None:
            label = 1    # default to first class if not found

        boxes  = []
        labels = []

        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                xmin = int(float(bb.find("xmin").text))
                ymin = int(float(bb.find("ymin").text))
                xmax = int(float(bb.find("xmax").text))
                ymax = int(float(bb.find("ymax").text))
            except (TypeError, ValueError):
                continue

            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        if not boxes:
            return self._fallback_box(img_path)

        return boxes, labels

    def _fallback_box(self, img_path: Path):
        """Return a whole-image bounding box when no XML annotation exists."""
        img = Image.open(img_path)
        w, h = img.size
        breed = img_path.stem.rsplit("_", 1)[0]
        label = self.class_to_idx.get(breed, 1)
        return [[0, 0, w, h]], [label]

    # ------------------------------------------------------------------
    @classmethod
    def from_directory(
        cls,
        data_root: str | Path,
        split: str,
        image_size: int = 512,
        transform=None,
    ):
        """
        Convenience constructor: infer class_to_idx from the directory names
        present in the train split.

        Usage:
            train_ds = OxfordPetsDataset.from_directory("data/oxford_pets", "train")
            val_ds   = OxfordPetsDataset.from_directory(
                           "data/oxford_pets", "val",
                           class_to_idx=train_ds.class_to_idx
                       )
        """
        data_root = Path(data_root)
        train_img_dir = data_root / "train" / "images"

        breed_names = set()
        for img_path in train_img_dir.glob("*.jpg"):
            breed = img_path.stem.rsplit("_", 1)[0]
            breed_names.add(breed)

        # Sort for reproducibility; 1-based (0 = background)
        class_to_idx = {b: i + 1 for i, b in enumerate(sorted(breed_names))}

        split_dir = data_root / split
        return cls(
            root=split_dir,
            class_to_idx=class_to_idx,
            split=split,
            image_size=image_size,
            transform=transform,
        ), class_to_idx

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        """Including background class 0."""
        return len(self.class_to_idx) + 1

    @property
    def class_names(self) -> list:
        idx_to_breed = {v: k for k, v in self.class_to_idx.items()}
        return ["__background__"] + [idx_to_breed[i + 1] for i in range(len(self.class_to_idx))]
