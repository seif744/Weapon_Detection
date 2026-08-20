#!/usr/bin/env python3
"""
yolo2coco.py -- convert a YOLO-format dataset to the COCO layout RF-DETR wants.

Input (what you have):
    datasets_merged/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt      # class cx cy w h  (normalized)

Output (what RF-DETR wants):
    datasets_coco/
        train/_annotations.coco.json + images
        valid/_annotations.coco.json + images     <- note "valid", not "val"
        test/_annotations.coco.json  + images

Usage:
    python yolo2coco.py --src datasets_merged --dst datasets_coco
    python yolo2coco.py --src datasets_merged --dst datasets_coco --copy

By default images are SYMLINKED, not copied, to save disk.

Why symlinks are safe here but were NOT safe in Ultralytics (your section 3.2):
Ultralytics derives label paths by resolving the image path and swapping
"images" -> "labels", so a symlink sent it back to the unmerged directory.
COCO JSON names every file explicitly -- there is no path derivation, so
there is nothing to resolve wrong. If you'd rather not risk it, use --copy.

Class ids are written 1-indexed with a placeholder at 0, matching Roboflow's
COCO exports. If RF-DETR complains about category ids, that's the first
thing to check against a real Roboflow export.
"""

import argparse
import json
import os
import shutil
import sys

from PIL import Image

SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
            ".JPG", ".JPEG", ".PNG")


def load_names(src):
    """Read class names from dataset.yaml without needing pyyaml."""
    path = os.path.join(src, "dataset.yaml")
    names = {}
    if not os.path.exists(path):
        return names
    in_names = False
    for line in open(path):
        s = line.strip()
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            if not s or ":" not in s or not s.split(":")[0].strip().isdigit():
                break
            k, v = s.split(":", 1)
            names[int(k.strip())] = v.strip().strip("'\"")
    return names


def convert_split(src, dst, split, out_split, names, copy):
    img_dir = os.path.join(src, "images", split)
    lbl_dir = os.path.join(src, "labels", split)
    if not os.path.isdir(img_dir):
        print(f"  {split}: no images dir, skipping")
        return

    out_dir = os.path.join(dst, out_split)
    os.makedirs(out_dir, exist_ok=True)

    # category 0 is a placeholder, real classes start at 1
    max_id = max(names) if names else 0
    categories = [{"id": 0, "name": "none", "supercategory": "none"}]
    for i in range(max_id + 1):
        categories.append({
            "id": i + 1,
            "name": names.get(i, f"class{i}"),
            "supercategory": "none",
        })

    images, annotations = [], []
    img_id = 0
    ann_id = 0
    n_bg = 0
    n_bad = 0

    files = sorted(f for f in os.listdir(img_dir) if f.endswith(IMG_EXTS))
    total = len(files)

    for n, fname in enumerate(files):
        ipath = os.path.join(img_dir, fname)
        try:
            W, H = Image.open(ipath).size
        except Exception:
            n_bad += 1
            continue

        img_id += 1
        images.append({"id": img_id, "file_name": fname,
                       "width": W, "height": H})

        # place the image next to the json
        target = os.path.join(out_dir, fname)
        if not os.path.exists(target):
            if copy:
                shutil.copy2(ipath, target)
            else:
                os.symlink(os.path.abspath(ipath), target)

        lpath = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(lpath):
            n_bg += 1
            continue

        had_box = False
        for line in open(lpath):
            f = line.split()
            if len(f) != 5:
                continue
            try:
                c = int(float(f[0]))
                cx, cy, w, h = (float(x) for x in f[1:])
            except ValueError:
                continue

            x = (cx - w / 2) * W
            y = (cy - h / 2) * H
            bw, bh = w * W, h * H

            # clamp to image bounds
            x, y = max(0.0, x), max(0.0, y)
            bw = min(bw, W - x)
            bh = min(bh, H - y)
            if bw <= 1 or bh <= 1:
                continue

            ann_id += 1
            had_box = True
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": c + 1,          # shift for the placeholder
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
        if not had_box:
            n_bg += 1

        if n % 2000 == 0 and n:
            print(f"    {n}/{total}", end="\r", flush=True)

    out = {"info": {"description": f"converted from {src}"},
           "licenses": [],
           "images": images,
           "annotations": annotations,
           "categories": categories}

    with open(os.path.join(out_dir, "_annotations.coco.json"), "w") as f:
        json.dump(out, f)

    print(f"  {split} -> {out_split}: {len(images)} images, "
          f"{len(annotations)} boxes, {n_bg} background, {n_bad} unreadable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="e.g. datasets_merged")
    ap.add_argument("--dst", required=True, help="e.g. datasets_coco")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (uses disk)")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        sys.exit(f"no such directory: {args.src}")

    names = load_names(args.src)
    if names:
        print(f"classes from dataset.yaml: {names}")
    else:
        print("WARNING: no names read from dataset.yaml, using class0/1/2...")

    os.makedirs(args.dst, exist_ok=True)
    for split, out_split in SPLIT_MAP.items():
        convert_split(args.src, args.dst, split, out_split, names, args.copy)

    print(f"\ndone -> {args.dst}/")
    print("category ids are 1-indexed (0 is a placeholder), matching Roboflow exports.")


if __name__ == "__main__":
    main()
