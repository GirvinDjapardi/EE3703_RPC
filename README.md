# EE3703 Project — Retail Product Checkout Detection

**Group 4:** Girvin, Stanley, Naveen

## Problem Statement

Detecting grocery items at a self-checkout counter using two object detection architectures (YOLO and Faster R-CNN), implemented from scratch in PyTorch. The system recognises four product classes on a checkout surface and draws bounding boxes with class labels and confidence scores.

## Detection Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | Bottled Drink | Plastic/glass bottles (water, soda, juice, etc.) |
| 1 | Canned Goods | Metal cans and tins (beer, canned food, etc.) |
| 2 | Packaged Food | Bags, boxes, cartons (snacks, noodles, chocolate, etc.) |
| 3 | Fresh Produce | Fruits and vegetables |

## Dataset

500 images total, all resized to 640x640 pixels.

| Source | Images | Classes Covered |
|--------|--------|-----------------|
| Kaggle RPC (val subset) | 200 | Bottled Drink, Canned Goods, Packaged Food |
| MVTec D2S (subset) | 160 | Fresh Produce + others |
| Original (self-captured) | 140 (90 train / 50 test) | All 4 classes |

The 50 original test images are held out strictly for final evaluation and are never used during training or hyperparameter tuning.

## Technical Approach

Both models are built from PyTorch primitives (`nn.Conv2d`, `nn.BatchNorm2d`, etc.)

- **YOLO** — single-stage detector based on pretrained yolo26s.pt, using a backbone, feature-pyramid style neck, and multi-scale detection heads.
- **Faster R-CNN** — two-stage detector with a custom ResNet-50 backbone + FPN + RPN + ROI Align + detection head

## Evaluation Metrics

- **Primary:** mAP (mean Average Precision) — target 0.85
- **Secondary:** Precision, Recall, F1-score, Inference Speed (FPS)

README.txt contains our results and more details