# Assignment 2: Object Detection and Recognition
**Course:** Machine Learning  
**Date:** March 2026

---

## 1. Introduction

This report presents the implementation and evaluation of two object detection models — Faster R-CNN and YOLOv8n — trained on two small datasets: the Penn-Fudan Pedestrian Dataset and a subset of the Oxford-IIIT Pet Dataset. The goal is to compare model performance in terms of accuracy and speed under GPU memory constraints (8GB VRAM).

---

## 2. Dataset Description

### 2.1 Penn-Fudan Pedestrian Dataset
- **Task:** Pedestrian detection (single class: person)
- **Size:** ~170 images with pixel-level segmentation masks
- **Split:** 70% train / 15% validation / 15% test (~119 / 25 / 27 images)
- **Annotations:** Converted from segmentation masks to bounding boxes

### 2.2 Oxford-IIIT Pet Dataset (Subset)
- **Task:** Pet breed detection and classification
- **Subset:** 5–7 breeds selected to stay within GPU memory limits
- **Split:** 70% train / 15% validation / 15% test
- **Annotations:** Bounding boxes with breed class labels

Both datasets were resized to **512×512** pixels during preprocessing.

---

## 3. Model Description

### 3.1 Faster R-CNN (MobileNetV3-Large FPN)
Faster R-CNN is a two-stage detector. It first proposes candidate regions using a Region Proposal Network (RPN), then classifies and refines each proposal. The backbone used here is MobileNetV3-Large with a Feature Pyramid Network (FPN), chosen for its lower memory footprint compared to ResNet backbones.

- **Pretrained weights:** ImageNet
- **Batch size:** 2
- **Optimizer:** SGD with momentum
- **Learning rate:** 0.005
- **Image size:** 512×512

### 3.2 YOLOv8n (Nano)
YOLOv8n is a single-stage detector that predicts bounding boxes and class probabilities in a single forward pass. The nano variant is the smallest in the YOLOv8 family, making it well-suited for low-memory GPUs. It is significantly faster than Faster R-CNN at inference time.

- **Pretrained weights:** COCO
- **Batch size:** 8
- **Optimizer:** AdamW (default)
- **Image size:** 512×512

---

## 4. Training Details

| Setting | Faster R-CNN | YOLOv8n |
|---|---|---|
| Backbone | MobileNetV3-Large FPN | CSPDarknet (nano) |
| Pretrained | Yes (ImageNet) | Yes (COCO) |
| Batch Size | 2 | 8 |
| Image Size | 512×512 | 512×512 |
| Epochs (Penn-Fudan) | 15 | 6 (early stopping) |
| Epochs (Oxford Pets) | 20 | 15 |
| Mixed Precision | Yes | Yes |
| Early Stopping | Yes | Yes |
| GPU | NVIDIA RTX 3050 6GB | NVIDIA RTX 3050 6GB |

YOLOv8n triggered early stopping at epoch 6 on Penn-Fudan, indicating the model converged quickly on this small dataset. Any further training would have likely led to overfitting, and was unnecessary. Faster R-CNN required more epochs due to its two-stage architecture and smaller batch size.

---

## 5. Results

### 5.1 Penn-Fudan Pedestrian Dataset

| Model | mAP@0.5 | Precision | Recall | Training Time | Inference Speed |
|---|---|---|---|---|---|
| Faster R-CNN | **0.862** | **0.862** | **0.649** | **100.8s** | **15.29 img/s** |
| YOLOv8n | **0.829** | **0.834** | **0.657** | **100.8s** | **37.8 img/s** |

### 5.2 Oxford-IIIT Pet Dataset (Subset)

| Model | mAP@0.5 | Precision | Recall | Training Time | Inference Speed |
|---|---|---|---|---|---|
| Faster R-CNN | **0.346** | **0.346** | **0.498** | **100.7s** | **36.75 img/s** |
| YOLOv8n | **61.34** | **0.52** | **0.85** | **211.3s** | **0** |

### 5.3 Full Comparison

| Dataset | Model | mAP@0.5 | Precision | Recall | Training Time | Inference Speed |
|---|---|---|---|---|---|---|
| Penn-Fudan | Faster R-CNN | **0.862** | **0.862** | **0.649** | **100.8s** | **15.29 img/s** |
| Penn-Fudan | YOLOv8n | **0.829** | **0.834** | **0.657** | **100.8s** | **37.8 img/s** |
| Oxford Pets | Faster R-CNN | **0.346** | **0.346** | **0.498** | **100.7s** | **36.75 img/s** |
| Oxford Pets | YOLOv8n | **61.34** | **0.52** | **0.85** | **211.3s** | **0** |

---

## 6. Discussion

### Accuracy
YOLOv8n achieved strong results on Penn-Fudan with an mAP@0.5 of **82.9%**, demonstrating that single-stage detectors with COCO pretraining transfer well to small pedestrian datasets. The high precision (83.4%) suggests the model made few false positive detections, while the moderate recall (65.7%) indicates some pedestrians were missed. The Oxford Pets dataset appeared to be affected heavily due to the choice in breeds. It seems that I chose breeds that the models struggled to differentiate between, which likely impacted the accuracy and precision of them.

### Speed
YOLOv8n is expected to be significantly faster at inference than Faster R-CNN due to its single-stage design. Faster R-CNN's two-stage pipeline (region proposal + classification) introduces additional latency, making it less suitable for real-time applications despite its typically higher accuracy on complex scenes.

### GPU Memory
Both models stayed within the 8GB GPU limit by using:
- Lightweight model variants (MobileNetV3 and YOLO nano)
- Small batch sizes (2 for Faster R-CNN, 8 for YOLO)
- Mixed precision training (FP16)
- Image size of 512×512 instead of 640+

### Early Stopping
YOLOv8n converged at epoch 6 on Penn-Fudan rather than the planned 15 epochs. This is expected behavior on small datasets (~170 images) — further training would likely cause overfitting. The model is well-regularized by its COCO pretrained weights.

---

## 7. Conclusion

Both Faster R-CNN and YOLOv8n are viable detection models under limited GPU constraints. YOLOv8n demonstrated fast convergence and strong accuracy on the Penn-Fudan dataset, while offering faster inference speeds. Faster R-CNN, while more memory-intensive, provides a robust two-stage detection that may yield higher accuracy on multi-class datasets like the Oxford Pets subset.

For production scenarios requiring real-time inference, YOLOv8n is the preferred choice. For scenarios prioritizing detection accuracy where speed is less critical, Faster R-CNN remains competitive.

---

## References

- Ren et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.*
- Jocher et al. (2023). *Ultralytics YOLOv8.*
- Wang et al. (2022). *Penn-Fudan Database for Pedestrian Detection and Segmentation.*
- Parkhi et al. (2012). *Cats and Dogs. IEEE CVPR.*
