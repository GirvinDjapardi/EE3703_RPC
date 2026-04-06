EE3703 Project
Retail Product Checkout Detection

Group Members
- Girvin
- Stanley
- Naveen

Project Summary
This project studies grocery item detection at a checkout surface using two object detection pipelines:
- Faster R-CNN implemented manually in PyTorch in `FasterRCNN_Project.ipynb`
- YOLO using `yolo26s.pt` with custom dataset preparation, split control, reporting, and artifact generation in `YOLO_Project.ipynb`

The detection task uses 4 foreground classes:
- Bottled Drink
- Canned Goods
- Fresh Produce
- Packaged Food
Background-only images are also included as negative examples during training.

Both notebooks were designed to be run in Google Colab with a T4 GPU. The saved notebooks are intended to be submitted with visible outputs, including training logs, plots, evaluation metrics, qualitative predictions, and exported artifacts.

Project Structure
The key files and folders in this project are:

Demo_Project_Group_4/
|- FasterRCNN_Project.ipynb
|  Final Faster R-CNN training and evaluation notebook
|- YOLO_Project.ipynb
|  Final YOLO training and evaluation notebook based on yolo26s
|- README.txt
|  This summary file
|- Demo_Video.mp4
|  3-minute video file showing our best model for each model architecture running inference on the 50 Original Test Images.
|- scripts/
|  Supporting preprocessing and exploration .py scripts
|  |- 01_data_exploration.ipynb
|  |- 02_subsample_kaggle_rpc.py
|  |- 03_preprocess_kaggle_rpc.py

What The Supporting Scripts Do
- `scripts/01_data_exploration.ipynb`
  Exploratory notebook used during dataset understanding and class/source inspection.
- `scripts/02_subsample_kaggle_rpc.py`
  Selects about 200 Kaggle RPC validation images for the project subset, with an emphasis on class diversity.
- `scripts/03_preprocess_kaggle_rpc.py`
  Resizes the selected Kaggle subset to `640 x 640` and writes a clean COCO-format annotation file for downstream use.

Dependencies
The two final notebooks and the scripts rely on the following main packages:
- Python 3.12 was used in the Colab runs
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Pillow
- tqdm
- Ultralytics

They also use standard library modules such as:
- json
- csv
- pathlib
- shutil
- zipfile
- os
- sys
- math
- random
- time
- datetime
- statistics
- warnings
- subprocess
- collections

Recommended Execution Environment
- Google Colab
- GPU runtime with NVIDIA T4
- Project zip uploaded to Google Drive and extracted inside `/content/EE3703 RPC` (rename to EE3703 RPC)

Dataset Notes
- Final dataset volume used by the submitted notebooks is 500 images.
- The notebooks combine public images from Kaggle RPC and MVTec D2S with original self-captured grocery images.
- The test set is a fixed 50-image original self-image split.
- 5 Background-only images from Naveen are also handled explicitly in the dataset construction.

Notebook Summary
1. Faster R-CNN Notebook
- File: `FasterRCNN_Project.ipynb`
- Model type: custom two-stage detector implemented in PyTorch
- Main components include a ResNet-style backbone, FPN, RPN, ROI Align, classification and box regression heads, manual training loop, threshold selection, custom evaluation, failure analysis, and benchmark reporting.

2. YOLO Notebook
- File: `YOLO_Project.ipynb`
- Model type: YOLO pipeline based on `yolo26s.pt`
- The notebook adds custom preprocessing, fixed split reconstruction, YOLO-format dataset export, threshold tuning, custom result summaries, failure analysis, showcase predictions, and benchmark reporting on top of the YOLO training workflow.



Final Findings
The two final notebooks produced the following main results on the saved runs:


Faster R-CNN
- Run folder: `fasterrcnn_grocery_run`
- Epochs run: 78
- Best epoch: 58
- Batch size: 2
- Train images: 384
- Validation images: 66
- Test images: 50
- Train original images: 85
- Validation original images: 0
- Test original images: 50
- Score threshold: 0.60

- mAP@0.50: 0.8467
- mAP@0.50:0.95: 0.4929
- Precision: 0.7569
- Recall: 0.8862
- F1: 0.8165
- AP50 by class:
  - Bottled Drink: 0.7978
  - Canned Goods: 0.8004
  - Fresh Produce: 0.9471
  - Packaged Food: 0.8414
- Inference benchmark:
  - Mean latency: 46.80 ms
  - FPS: 21.37



YOLO
- Run folder: `yolo_project_run`
- Model weights: `yolo26s.pt`
- Epochs run: 100
- Best epoch: 52
- Batch size: 16
- Train images: 384
- Validation images: 66
- Test images: 50
- Train original self images: 69
- Validation original self images: 16
- Test original self images: 50
- Report score threshold: 0.15

- mAP@0.50: 0.9788
- mAP@0.50:0.95: 0.7531
- Precision: 0.9524
- Recall: 0.9756
- F1: 0.9639
- AP50 by class:
  - Bottled Drink: 1.0000
  - Canned Goods: 0.9692
  - Fresh Produce: 0.9698
  - Packaged Food: 0.9763
- Inference benchmark:
  - Mean latency: 18.42 ms
  - FPS: 54.30

High-Level Conclusions
- The Faster R-CNN result is consistent with the behaviour of a two-stage detector. The model first proposes candidate regions and then refines class and box predictions, which helps it achieve strong recall, good class-wise AP across all 4 categories, and solid localization quality for a fully custom implementation.
- The YOLO result is consistent with the behaviour of a modern one-stage detector. Using `yolo26s.pt` gives much stronger feature extraction and confidence ranking from the start, which explains the higher mAP, higher precision, higher recall, and much faster inference speed on the same fixed test split.
- In direct comparison, YOLO is the better-performing final detector for this project because it is both more accurate and more efficient on the saved run. Faster R-CNN remains technically valuable because it demonstrates the full two-stage detection pipeline in a much more transparent and manually implemented form.
