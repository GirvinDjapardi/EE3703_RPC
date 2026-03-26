"""
03_preprocess_kaggle_rpc.py
----------------------------
Preprocess the 200 selected Kaggle RPC images for training.

What this script does:
  1. Reads the selection manifest from 02_subsample (data/kaggle_rpc_selected_200.json)
  2. Loads the original COCO annotations from instances_val2019.json
  3. For each selected image:
     - Loads the JPEG from val2019/
     - Resizes to 640x640 (the project-required image size)
     - Scales all bounding box coordinates to match the new dimensions
     - Remaps the original category_id (1-200) to project class IDs:
         0 = Bottled Drink
         1 = Canned Goods
         2 = Packaged Food
       (Fresh Produce = 3, but not present in Kaggle RPC)
  4. Saves resized images to data/kaggle_rpc_200/images/
  5. Saves a clean COCO-format JSON to data/kaggle_rpc_200/annotations.json

Inputs:
  - data/kaggle_rpc_selected_200.json
  - kaggle_RPC-Dataset/retail_product_checkout/instances_val2019.json
  - kaggle_RPC-Dataset/retail_product_checkout/val2019/*.jpg

Outputs:
  - data/kaggle_rpc_200/images/*.jpg        (200 resized images)
  - data/kaggle_rpc_200/annotations.json    (COCO-format annotations)
"""

import json
import csv
import shutil
from collections import Counter
from pathlib import Path
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "kaggle_RPC-Dataset" / "retail_product_checkout"
REMAP_CSV    = PROJECT_ROOT / "kaggle_RPC-Dataset" / "remappping_kaggle_RPC.csv"
MANIFEST     = PROJECT_ROOT / "data" / "kaggle_rpc_selected_200.json"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "kaggle_rpc_200"
OUTPUT_IMGS  = OUTPUT_DIR / "images"
OUTPUT_JSON  = OUTPUT_DIR / "annotations.json"

TARGET_SIZE = (640, 640)

# Project class mapping: mapped_category string -> integer class ID
PROJECT_CLASSES = {
    "Bottled Drink": 0,
    "Canned Goods": 1,
    "Packaged Food": 2,
    # "Fresh Produce": 3  — not present in Kaggle RPC
}

# ── Load remapping (SKU name -> mapped category string) ───────────────────────
sku_to_mapped = {}
with open(REMAP_CSV, "r") as f:
    for row in csv.DictReader(f):
        sku_to_mapped[row["sku_name"]] = row["mapped_category"]

# ── Load manifest (selected image IDs) ────────────────────────────────────────
with open(MANIFEST, "r") as f:
    manifest = json.load(f)

selected_ids = {img["image_id"] for img in manifest["images"]}
print(f"Selected images from manifest: {len(selected_ids)}")

# ── Load original val annotations ─────────────────────────────────────────────
with open(DATASET_ROOT / "instances_val2019.json", "r") as f:
    val_data = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in val_data["categories"]}
cat_id_to_mapped = {
    cid: sku_to_mapped.get(name, "UNMAPPED")
    for cid, name in cat_id_to_name.items()
}

# Build original category_id -> project class_id mapping
cat_id_to_project_id = {}
for cid, mapped_str in cat_id_to_mapped.items():
    if mapped_str in PROJECT_CLASSES:
        cat_id_to_project_id[cid] = PROJECT_CLASSES[mapped_str]
    # N/A and UNMAPPED categories are skipped (not added to dict)

# ── Prepare output directory ──────────────────────────────────────────────────
OUTPUT_IMGS.mkdir(parents=True, exist_ok=True)

# ── Process images and annotations ────────────────────────────────────────────
# Build lookup for original image metadata
orig_img_lookup = {img["id"]: img for img in val_data["images"]}

# Group original annotations by image_id
from collections import defaultdict
orig_anns_by_img = defaultdict(list)
for ann in val_data["annotations"]:
    orig_anns_by_img[ann["image_id"]].append(ann)

coco_images = []
coco_annotations = []
ann_id_counter = 0
class_counter = Counter()
skipped_anns = 0

for img_idx, img_id in enumerate(sorted(selected_ids)):
    orig_img = orig_img_lookup[img_id]
    orig_w = orig_img["width"]
    orig_h = orig_img["height"]
    file_name = orig_img["file_name"]

    # Scale factors for bounding box transformation
    scale_x = TARGET_SIZE[0] / orig_w
    scale_y = TARGET_SIZE[1] / orig_h

    # Resize and save image
    src_path = DATASET_ROOT / "val2019" / file_name
    img = Image.open(src_path).convert("RGB")
    img_resized = img.resize(TARGET_SIZE, Image.LANCZOS)
    img_resized.save(OUTPUT_IMGS / file_name, "JPEG", quality=95)

    # Add to COCO images list (with new dimensions)
    coco_images.append({
        "id": img_id,
        "file_name": file_name,
        "width": TARGET_SIZE[0],
        "height": TARGET_SIZE[1],
    })

    # Process annotations for this image
    for ann in orig_anns_by_img[img_id]:
        orig_cat_id = ann["category_id"]

        # Skip annotations that don't map to a project class
        if orig_cat_id not in cat_id_to_project_id:
            skipped_anns += 1
            continue

        project_class_id = cat_id_to_project_id[orig_cat_id]

        # Scale bounding box: [x, y, width, height] in COCO format
        ox, oy, ow, oh = ann["bbox"]
        scaled_bbox = [
            round(ox * scale_x, 2),
            round(oy * scale_y, 2),
            round(ow * scale_x, 2),
            round(oh * scale_y, 2),
        ]

        # Clamp to image bounds
        scaled_bbox[0] = max(0, scaled_bbox[0])
        scaled_bbox[1] = max(0, scaled_bbox[1])
        scaled_bbox[2] = min(scaled_bbox[2], TARGET_SIZE[0] - scaled_bbox[0])
        scaled_bbox[3] = min(scaled_bbox[3], TARGET_SIZE[1] - scaled_bbox[1])

        # Skip degenerate boxes
        if scaled_bbox[2] <= 1 or scaled_bbox[3] <= 1:
            skipped_anns += 1
            continue

        scaled_area = round(scaled_bbox[2] * scaled_bbox[3], 2)

        coco_annotations.append({
            "id": ann_id_counter,
            "image_id": img_id,
            "category_id": project_class_id,
            "bbox": scaled_bbox,
            "area": scaled_area,
            "iscrowd": 0,
        })
        ann_id_counter += 1

        class_name = [k for k, v in PROJECT_CLASSES.items() if v == project_class_id][0]
        class_counter[class_name] += 1

    if (img_idx + 1) % 50 == 0:
        print(f"  Processed {img_idx + 1}/{len(selected_ids)} images...")

# ── Build COCO JSON ───────────────────────────────────────────────────────────
coco_output = {
    "info": {
        "description": "EE3703 Project — Kaggle RPC subset (200 images, 3 classes, resized to 640x640)",
        "version": "1.0",
        "year": 2026,
    },
    "categories": [
        {"id": 0, "name": "Bottled Drink", "supercategory": "product"},
        {"id": 1, "name": "Canned Goods", "supercategory": "product"},
        {"id": 2, "name": "Packaged Food", "supercategory": "product"},
        # Fresh Produce (id=3) will be added when merging with MVTec D2S
    ],
    "images": coco_images,
    "annotations": coco_annotations,
}

with open(OUTPUT_JSON, "w") as f:
    json.dump(coco_output, f, indent=2)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Images processed:     {len(coco_images)}")
print(f"Annotations kept:     {len(coco_annotations)}")
print(f"Annotations skipped:  {skipped_anns} (N/A or degenerate)")
print(f"Image size:           {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
print(f"\nAnnotations per class:")
for cls in PROJECT_CLASSES:
    print(f"  {cls:20s}: {class_counter[cls]:>5}")
print(f"\nOutputs:")
print(f"  Images: {OUTPUT_IMGS}")
print(f"  COCO JSON: {OUTPUT_JSON}")
