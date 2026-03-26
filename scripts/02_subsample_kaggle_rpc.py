"""
02_subsample_kaggle_rpc.py
--------------------------
Select ~200 images from the Kaggle RPC val set for training.

Strategy:
  1. Use only "pure-valid" val images (images with zero N/A annotations).
  2. Prioritise images that contain all 3 RPC classes (Bottled Drink,
     Canned Goods, Packaged Food) to maximise class diversity per image.
  3. Among those, rank by balance score — favour images where the
     minority classes (Bottled Drink, Canned Goods) are well-represented
     relative to the majority class (Packaged Food).
  4. Select the top 200 images.
  5. Save the selection as a JSON manifest for downstream use.

Inputs:
  - kaggle_RPC-Dataset/retail_product_checkout/instances_val2019.json
  - kaggle_RPC-Dataset/remappping_kaggle_RPC.csv

Outputs:
  - data/kaggle_rpc_selected_200.json   (list of selected image IDs + metadata)
  - Printed summary of class distribution in the selected subset
"""

import json
import csv
import os
from collections import Counter, defaultdict
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "kaggle_RPC-Dataset" / "retail_product_checkout"
REMAP_CSV    = PROJECT_ROOT / "kaggle_RPC-Dataset" / "remappping_kaggle_RPC.csv"
OUTPUT_DIR   = PROJECT_ROOT / "data"
OUTPUT_FILE  = OUTPUT_DIR / "kaggle_rpc_selected_200.json"

TARGET_COUNT = 200

# ── Load remapping ─────────────────────────────────────────────────────────────
sku_to_mapped = {}
with open(REMAP_CSV, "r") as f:
    for row in csv.DictReader(f):
        sku_to_mapped[row["sku_name"]] = row["mapped_category"]

# ── Load val annotations ───────────────────────────────────────────────────────
with open(DATASET_ROOT / "instances_val2019.json", "r") as f:
    val_data = json.load(f)

cat_id_to_name = {c["id"]: c["name"] for c in val_data["categories"]}
cat_id_to_mapped = {
    cid: sku_to_mapped.get(name, "UNMAPPED")
    for cid, name in cat_id_to_name.items()
}

CLASS_NAMES = ["Bottled Drink", "Canned Goods", "Packaged Food"]

# ── Group annotations by image ────────────────────────────────────────────────
img_anns = defaultdict(list)
img_has_na = set()

for ann in val_data["annotations"]:
    mapped = cat_id_to_mapped[ann["category_id"]]
    img_anns[ann["image_id"]].append(ann)
    if mapped == "N/A":
        img_has_na.add(ann["image_id"])

# ── Filter to pure-valid images (no N/A annotations) ──────────────────────────
all_img_ids = {img["id"] for img in val_data["images"]}
pure_valid_ids = all_img_ids - img_has_na
# Also exclude images with no annotations at all
pure_valid_ids = {iid for iid in pure_valid_ids if iid in img_anns}

print(f"Total val images:        {len(all_img_ids):,}")
print(f"Images with any N/A:     {len(img_has_na):,}")
print(f"Pure-valid images:       {len(pure_valid_ids):,}")

# ── Compute per-image class profile ───────────────────────────────────────────
img_profiles = {}
for img_id in pure_valid_ids:
    class_counts = Counter()
    for ann in img_anns[img_id]:
        mapped = cat_id_to_mapped[ann["category_id"]]
        if mapped in CLASS_NAMES:
            class_counts[mapped] += 1
    img_profiles[img_id] = class_counts

# ── Categorise by class diversity ─────────────────────────────────────────────
has_all_three = [iid for iid, profile in img_profiles.items()
                 if all(profile[cls] > 0 for cls in CLASS_NAMES)]
has_two = [iid for iid, profile in img_profiles.items()
           if sum(1 for cls in CLASS_NAMES if profile[cls] > 0) == 2]
has_one = [iid for iid, profile in img_profiles.items()
           if sum(1 for cls in CLASS_NAMES if profile[cls] > 0) == 1]

print(f"\nClass diversity breakdown (pure-valid):")
print(f"  All 3 classes: {len(has_all_three)} images")
print(f"  2 classes:     {len(has_two)} images")
print(f"  1 class:       {len(has_one)} images")

# ── Scoring: favour images with better minority-class representation ──────────
# Balance score = (Bottled Drink + Canned Goods) / total_valid_annotations
# Higher score = more minority-class representation
def balance_score(img_id):
    profile = img_profiles[img_id]
    total = sum(profile.values())
    if total == 0:
        return 0
    minority = profile["Bottled Drink"] + profile["Canned Goods"]
    return minority / total

# ── Select images ─────────────────────────────────────────────────────────────
# Priority 1: images with all 3 classes, ranked by balance score
has_all_three_ranked = sorted(has_all_three, key=balance_score, reverse=True)

selected_ids = []

# Take from all-3-class images first
selected_ids.extend(has_all_three_ranked[:TARGET_COUNT])

# If we need more (unlikely since we have 291), fill from 2-class images
if len(selected_ids) < TARGET_COUNT:
    remaining = TARGET_COUNT - len(selected_ids)
    has_two_ranked = sorted(has_two, key=balance_score, reverse=True)
    selected_ids.extend(has_two_ranked[:remaining])

selected_ids = selected_ids[:TARGET_COUNT]
print(f"\nSelected {len(selected_ids)} images")

# ── Compute final class distribution ──────────────────────────────────────────
final_counts = Counter()
final_img_counts = Counter()

for img_id in selected_ids:
    classes_present = set()
    for ann in img_anns[img_id]:
        mapped = cat_id_to_mapped[ann["category_id"]]
        if mapped in CLASS_NAMES:
            final_counts[mapped] += 1
            classes_present.add(mapped)
    for cls in classes_present:
        final_img_counts[cls] += 1

total_anns = sum(final_counts.values())
print(f"\n{'Class':20s} | {'Annotations':>12s} | {'% of Total':>10s} | {'In N Images':>12s}")
print("-" * 65)
for cls in CLASS_NAMES:
    count = final_counts[cls]
    pct = count / total_anns * 100 if total_anns > 0 else 0
    img_count = final_img_counts[cls]
    print(f"{cls:20s} | {count:>12,} | {pct:>9.1f}% | {img_count:>12,}")
print(f"{'TOTAL':20s} | {total_anns:>12,} |           | {len(selected_ids):>12,}")

# ── Save selection manifest ───────────────────────────────────────────────────
img_lookup = {img["id"]: img for img in val_data["images"]}

manifest = {
    "description": "200 images subsampled from Kaggle RPC val2019 for EE3703 project training",
    "source_split": "val2019",
    "selection_criteria": (
        "Pure-valid images only (no N/A annotations). "
        "Prioritised images containing all 3 classes. "
        "Ranked by minority-class balance score (Bottled Drink + Canned Goods ratio)."
    ),
    "class_names": CLASS_NAMES,
    "summary": {
        "total_images": len(selected_ids),
        "total_annotations": total_anns,
        "annotations_per_class": {cls: final_counts[cls] for cls in CLASS_NAMES},
        "images_per_class": {cls: final_img_counts[cls] for cls in CLASS_NAMES},
    },
    "images": [
        {
            "image_id": img_id,
            "file_name": img_lookup[img_id]["file_name"],
            "width": img_lookup[img_id]["width"],
            "height": img_lookup[img_id]["height"],
            "annotations_per_class": {
                cls: img_profiles[img_id].get(cls, 0) for cls in CLASS_NAMES
            },
        }
        for img_id in selected_ids
    ],
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(manifest, f, indent=2)

print(f"\nManifest saved to: {OUTPUT_FILE}")
