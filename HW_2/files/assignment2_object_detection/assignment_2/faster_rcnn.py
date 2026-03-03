"""
Faster R-CNN model builder using MobileNetV3-Large FPN backbone.

Usage:
    from models.faster_rcnn import build_faster_rcnn
    model = build_faster_rcnn(num_classes=2, pretrained=True)
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def build_faster_rcnn(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: int = 3,
    image_size: int = 512,
    **kwargs,
) -> FasterRCNN:
    """
    Build a Faster R-CNN model with MobileNetV3-Large FPN backbone.

    Args:
        num_classes               : total number of classes including background (0)
        pretrained                : load ImageNet-pretrained backbone weights
        trainable_backbone_layers : number of backbone stages to fine-tune (0–6)
        image_size                : max image dimension for the transform pipeline

    Returns:
        model ready for training/inference
    """
    # Load pretrained model
    model = fasterrcnn_mobilenet_v3_large_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers,
        min_size=image_size,
        max_size=image_size,
        **kwargs,
    )

    # Replace the box predictor head to match our number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def build_faster_rcnn_custom_anchors(
    num_classes: int,
    pretrained: bool = True,
    image_size: int = 512,
) -> FasterRCNN:
    """
    Alternative builder with custom anchor sizes tuned for 512×512 images.
    Useful when default anchors are a poor fit for your object size distribution.
    """
    # Backbone
    backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained).features
    backbone.out_channels = 960   # MobileNetV3-Large final feature channels

    # Custom anchor generator
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    # ROI align pooler
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=["0"],
        output_size=7,
        sampling_ratio=2,
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=image_size,
        max_size=image_size,
    )

    return model


def get_model_params(model: nn.Module) -> int:
    """Return total parameter count."""
    return sum(p.numel() for p in model.parameters())


def get_trainable_params(model: nn.Module) -> int:
    """Return trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module) -> None:
    total     = get_model_params(model)
    trainable = get_trainable_params(model)
    print(f"Faster R-CNN (MobileNetV3-Large FPN)")
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    print(f"  Frozen parameters   : {total - trainable:,}")


if __name__ == "__main__":
    # Quick smoke test
    model = build_faster_rcnn(num_classes=2, pretrained=False)
    model.eval()
    print_model_summary(model)

    dummy = [torch.rand(3, 512, 512)]
    with torch.no_grad():
        out = model(dummy)
    print(f"\nSample output keys: {list(out[0].keys())}")
    print(f"Boxes shape       : {out[0]['boxes'].shape}")
    print(f"Labels shape      : {out[0]['labels'].shape}")
    print(f"Scores shape      : {out[0]['scores'].shape}")
