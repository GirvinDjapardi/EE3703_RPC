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

300–500 images total, all resized to 640x640 pixels.

| Source | Images | Classes Covered |
|--------|--------|-----------------|
| Kaggle RPC (val subset) | 200 | Bottled Drink, Canned Goods, Packaged Food |
| MVTec D2S (subset) | 200 | Fresh Produce + others |
| Original (self-captured) | 100 (50 train / 50 test) | All 4 classes |

The 50 original test images are held out strictly for final evaluation and are never used during training or hyperparameter tuning.

## Technical Approach

Both models are built from PyTorch primitives (`nn.Conv2d`, `nn.BatchNorm2d`, etc.) — no pretrained weights or high-level detection APIs.

- **YOLO** — single-stage detector with backbone + FPN/PAN neck + multi-scale detection head
- **Faster R-CNN** — two-stage detector with ResNet-50 backbone + FPN + RPN + ROI Align + detection head

## Repo Structure

```
EE3703_Project/
├── Training Images/
│   ├── Kaggle Images/          # 200 images from Kaggle RPC (3 classes)
│   │   ├── images/             # 640x640 resized JPEGs
│   │   └── annotations.json   # COCO-format annotations
│   ├── MV Images/              # 200 images from MVTec D2S
│   └── Self Images/            # Original captured images (all 3 members)
│       ├── annotated_resized_Girvin-images/
│       ├── annotated_resized_Naveen-images/
│       └── stanley_dataset/
├── Kaggle RPC stuff/           # Data processing scripts and working files
│   ├── 01_data_exploration.ipynb
│   ├── 02_subsample_kaggle_rpc.py
│   ├── 03_preprocess_kaggle_rpc.py
│   └── PROJECT_CHECKLIST.md
├── README.md
└── .gitignore
```

## Evaluation Metrics

- **Primary:** mAP (mean Average Precision) — target 0.85
- **Secondary:** Precision, Recall, F1-score, Inference Speed (FPS)
