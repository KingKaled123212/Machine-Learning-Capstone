# Assignment 2 — Object Detection and Recognition

Comparison of **Faster R-CNN (MobileNetV3-Large FPN)** vs **YOLOv8n** on:
- Dataset 1: Penn-Fudan Pedestrian Dataset
- Dataset 2: Oxford-IIIT Pet Dataset (7-breed subset)

---

## Project Structure

```
object_detection/
├── configs/
│   ├── faster_rcnn_config.yaml       # Faster R-CNN hyperparameters
│   └── yolo_config.yaml              # YOLOv8n hyperparameters
├── data/
│   ├── download_pennfudan.py         # Download + prepare Penn-Fudan
│   ├── download_oxford_pets.py       # Download + prepare Oxford Pets subset
│   ├── pennfudan_dataset.py          # PyTorch Dataset class for Penn-Fudan
│   ├── oxford_pets_dataset.py        # PyTorch Dataset class for Oxford Pets
│   └── transforms.py                 # Data augmentation transforms
├── models/
│   ├── faster_rcnn.py                # Faster R-CNN model builder
│   └── yolo.py                       # YOLOv8n wrapper
├── utils/
│   ├── engine.py                     # Train/eval one epoch functions
│   ├── metrics.py                    # mAP, precision, recall computation
│   ├── visualize.py                  # Prediction visualisation helpers
│   └── general.py                    # Misc helpers (seed, logging, etc.)
├── train_faster_rcnn.py              # Train Faster R-CNN (both datasets)
├── train_yolo.py                     # Train YOLOv8n (both datasets)
├── evaluate.py                       # Evaluate any saved model on test split
├── compare_models.py                 # Generate comparison table + plots
├── notebooks/
│   └── exploration.ipynb             # EDA and result visualisation notebook
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
```bash
python data/download_pennfudan.py
python data/download_oxford_pets.py
```

### 3. Train models
```bash
# Faster R-CNN on Penn-Fudan
python train_faster_rcnn.py --dataset pennfudan --epochs 15

# Faster R-CNN on Oxford Pets
python train_faster_rcnn.py --dataset oxford_pets --epochs 20

# YOLOv8n on Penn-Fudan
python train_yolo.py --dataset pennfudan --epochs 15

# YOLOv8n on Oxford Pets
python train_yolo.py --dataset oxford_pets --epochs 20
```

### 4. Evaluate
```bash
python evaluate.py --model faster_rcnn --dataset pennfudan
python evaluate.py --model yolo --dataset oxford_pets
```

### 5. Compare & report
```bash
python compare_models.py
```

---

## GPU Requirements
- Minimum: 8 GB VRAM (RTX 3070 / RTX 3060 etc.)
- Mixed precision (fp16) enabled by default
- Batch size auto-reduced if OOM is detected

## Expected Results

| Dataset       | Model        | mAP@0.5 | Precision | Recall | Train Time | Inf. Speed |
|---------------|--------------|---------|-----------|--------|------------|------------|
| Penn-Fudan    | Faster R-CNN | ~0.85   | ~0.89     | ~0.82  | ~18 min    | ~12 img/s  |
| Penn-Fudan    | YOLOv8n      | ~0.81   | ~0.86     | ~0.80  | ~9 min     | ~47 img/s  |
| Oxford Pets   | Faster R-CNN | ~0.67   | ~0.71     | ~0.65  | ~34 min    | ~11 img/s  |
| Oxford Pets   | YOLOv8n      | ~0.64   | ~0.68     | ~0.62  | ~16 min    | ~45 img/s  |
