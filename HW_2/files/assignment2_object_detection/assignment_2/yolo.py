"""
YOLOv8n wrapper that provides a consistent interface for training and evaluation.

Usage:
    from models.yolo import YOLOWrapper
    wrapper = YOLOWrapper(num_classes=2, pretrained=True)
    wrapper.train(data_yaml="data/pennfudan/dataset.yaml", epochs=15)
    results = wrapper.evaluate(data_yaml="...", split="test")
"""

import os
import time
import torch
from pathlib import Path


class YOLOWrapper:
    """
    Thin wrapper around Ultralytics YOLO for a consistent project interface.

    Args:
        num_classes : number of detection classes (excluding background)
        pretrained  : load COCO-pretrained weights
        model_size  : 'n' (nano), 's', 'm', etc.
        device      : 'cuda', 'cpu', or device index
    """

    MODEL_MAP = {
        "n": "yolov8n.pt",
        "s": "yolov8s.pt",
        "m": "yolov8m.pt",
    }

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        model_size: str = "n",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )

        self.num_classes = num_classes
        self.model_size  = model_size
        self.device      = device

        weights = self.MODEL_MAP.get(model_size, "yolov8n.pt")
        self.model = YOLO(weights if pretrained else f"yolov8{model_size}.yaml")

        print(f"YOLOv8{model_size} loaded  |  classes: {num_classes}  |  device: {device}")

    # ------------------------------------------------------------------
    def train(
        self,
        data_yaml: str,
        epochs: int = 20,
        batch_size: int = 16,
        image_size: int = 512,
        project: str = "outputs/yolo",
        name: str = "train",
        patience: int = 5,
        amp: bool = True,
        **kwargs,
    ):
        """
        Fine-tune YOLOv8n on a custom dataset.

        Args:
            data_yaml  : path to YOLO-format dataset.yaml
            epochs     : max training epochs
            batch_size : images per batch
            image_size : input resolution (square)
            project    : output directory
            name       : experiment name (sub-folder inside project)
            patience   : early stopping patience
            amp        : use mixed precision (fp16)
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=self.device,
            project=project,
            name=name,
            patience=patience,
            amp=amp,
            exist_ok=True,
            **kwargs,
        )
        return results

    # ------------------------------------------------------------------
    def evaluate(
        self,
        data_yaml: str,
        split: str = "test",
        image_size: int = 512,
        batch_size: int = 16,
        conf: float = 0.25,
        iou: float = 0.5,
    ) -> dict:
        """
        Evaluate on a given split.

        Returns a dict with mAP50, mAP50-95, precision, recall, and speed.
        """
        results = self.model.val(
            data=data_yaml,
            split=split,
            imgsz=image_size,
            batch=batch_size,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=True,
        )

        metrics = {
            "map50":      float(results.box.map50),
            "map50_95":   float(results.box.map),
            "precision":  float(results.box.mp),
            "recall":     float(results.box.mr),
        }
        return metrics

    # ------------------------------------------------------------------
    def inference_speed(
        self,
        data_yaml: str,
        n_images: int = 100,
        image_size: int = 512,
        conf: float = 0.25,
    ) -> float:
        """
        Measure inference throughput (images/second) over n_images.

        Returns float: mean images per second.
        """
        import glob, random
        import numpy as np
        from PIL import Image
        import yaml

        # Load a few test images
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)

        data_root = Path(cfg.get("path", "."))
        test_imgs = list((data_root / "test" / "images").glob("*.jpg"))
        if not test_imgs:
            test_imgs = list((data_root / "val" / "images").glob("*.jpg"))

        sample = random.choices(test_imgs, k=min(n_images, len(test_imgs)))
        sample_paths = [str(p) for p in sample]

        # Warm-up
        if not sample_paths: return 0.0
        _ = self.model.predict(sample_paths[:2], imgsz=image_size, verbose=False, device=self.device)

        # Time
        t0 = time.perf_counter()
        self.model.predict(sample_paths, imgsz=image_size, verbose=False, device=self.device)
        elapsed = time.perf_counter() - t0

        return len(sample_paths) / elapsed

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Export best weights to path."""
        self.model.export(format="pt")  # saved by Ultralytics automatically

    def load(self, weights_path: str) -> None:
        """Load weights from a .pt file."""
        from ultralytics import YOLO
        self.model = YOLO(weights_path)

    # ------------------------------------------------------------------
    def predict(self, source, conf: float = 0.25, iou: float = 0.45, **kwargs):
        """Run inference on image(s) / directory / video."""
        return self.model.predict(
            source=source, conf=conf, iou=iou, device=self.device, **kwargs
        )

    # ------------------------------------------------------------------
    @staticmethod
    def make_dataset_yaml(
        data_root: str | Path,
        class_names: list,
        output_path: str | Path = None,
    ) -> str:
        """
        Generate a YOLO-compatible dataset.yaml string (and optionally write it).

        Args:
            data_root   : root directory with train/val/test sub-dirs
            class_names : list of class name strings (NO background class)
            output_path : if given, write yaml to this path

        Returns:
            yaml string
        """
        data_root = Path(data_root).resolve()
        content = (
            f"path: {data_root}\n"
            f"train: train/images\n"
            f"val:   val/images\n"
            f"test:  test/images\n"
            f"\n"
            f"nc: {len(class_names)}\n"
            f"names: {class_names}\n"
        )
        if output_path:
            Path(output_path).write_text(content)
            print(f"Wrote dataset yaml to {output_path}")
        return content
