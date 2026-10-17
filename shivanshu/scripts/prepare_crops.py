"""
PolyRoute: Automated Bounding-Box Cropper & Dataset Builder
Converts multi-object / mosaic bounding box annotations into
clean single-object ImageFolder format for CNN training.
"""

import os
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.join(ROOT_DIR, "plastic image")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset_crops")
SPLITS = ["train", "valid", "test"]

def process_single_split(split: str):
    split_dir = os.path.join(BASE_DIR, split)
    csv_file = os.path.join(split_dir, "_annotations.csv")

    if not os.path.exists(csv_file):
        print(f"Skipping {split}: {csv_file} not found.")
        return split, 0, 0

    out_split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(out_split_dir, exist_ok=True)

    with open(csv_file, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    print(f"[{split.upper()}] Read {len(reader)} annotation boxes from CSV.")

    saved_count = 0
    skipped_count = 0

    # Group boxes by image to avoid re-opening the image multiple times
    img_boxes = {}
    for idx, row in enumerate(reader):
        fn = row["filename"]
        if fn not in img_boxes:
            img_boxes[fn] = []
        img_boxes[fn].append((idx, row))

    total_images = len(img_boxes)
    print(f"[{split.upper()}] Processing {total_images} unique image files...")

    # Cache opened images and extract crops
    for i, (fn, boxes) in enumerate(img_boxes.items()):
        img_path = os.path.join(split_dir, fn)
        if not os.path.exists(img_path):
            skipped_count += len(boxes)
            continue

        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size

                for box_idx, box_row in boxes:
                    cls_name = box_row["class"].strip().replace("/", "_")
                    xmin = max(0, float(box_row["xmin"]))
                    ymin = max(0, float(box_row["ymin"]))
                    xmax = min(img_w, float(box_row["xmax"]))
                    ymax = min(img_h, float(box_row["ymax"]))

                    # Skip invalid or microscopic bounding boxes (< 15 px)
                    if (xmax - xmin) < 15 or (ymax - ymin) < 15:
                        skipped_count += 1
                        continue

                    crop_dir = os.path.join(out_split_dir, cls_name)
                    os.makedirs(crop_dir, exist_ok=True)

                    base_stem = os.path.splitext(fn)[0]
                    crop_filename = f"{base_stem}_box{box_idx}.jpg"
                    crop_dest = os.path.join(crop_dir, crop_filename)

                    cropped = img.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
                    cropped.convert("RGB").save(crop_dest, "JPEG", quality=90)
                    saved_count += 1

        except Exception as e:
            skipped_count += len(boxes)
            continue

        if (i + 1) % 1500 == 0 or (i + 1) == total_images:
            print(f"[{split.upper()}] Processed {i + 1}/{total_images} images (Saved {saved_count} crops)...")

    return split, saved_count, skipped_count

def main():
    print(f"Starting bounding-box extraction from '{BASE_DIR}' into '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = {}
    for split in SPLITS:
        s, saved, skipped = process_single_split(split)
        summary[s] = {"saved": saved, "skipped": skipped}

    print("\n================ DATASET EXTRACTION SUMMARY ================")
    for split, counts in summary.items():
        print(f"Split {split.upper():<6} -> Crops Saved: {counts['saved']:<6} | Skipped: {counts['skipped']}")
    print("============================================================\n")

if __name__ == "__main__":
    main()
