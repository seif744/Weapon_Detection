#!/usr/bin/env python3
"""
audit_annotations.py -- find problematic bounding boxes in an RF-DETR
(Roboflow COCO) dataset.

NEVER modifies the source dataset. Everything is written to an inspection
directory; corrected copies are separate files.

    source rfdetr_env/bin/activate
    cd ~/weapons_project

    # inspect the schema first, change nothing
    python audit_annotations.py --dataset datasets_coco/train --inspect

    # full audit
    python audit_annotations.py --dataset datasets_coco/train

    # also audit the upstream YOLO labels, which the COCO conversion
    # already sanitized (it clamps to bounds and drops boxes under 1px)
    python audit_annotations.py --dataset datasets_coco/train \
        --source-labels datasets_merged/labels/train

Output:
    problematic_annotations/<split>/
        images/          copies of flagged images
        annotations/     per-image original annotation JSON
        corrected/       per-image corrected annotation JSON (safe fixes only)
        visualizations/  boxes drawn on the image, problems in red
        report.csv
        corrected_annotations.coco.json    full corrected dataset
"""

import argparse
import csv
import json
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- thresholds
DEFAULTS = {
    "MIN_BOX_WIDTH": 4.0,        # px
    "MIN_BOX_HEIGHT": 4.0,       # px
    "MIN_BOX_AREA": 32.0,        # px^2
    "MIN_BOX_AREA_RATIO": 1e-5,  # fraction of image area
    "DUP_IOU": 0.95,             # IoU above which two boxes are near-duplicate
}

SEV = {"INVALID": 3, "SUSPICIOUS": 2, "MINOR": 1}


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


# ---------------------------------------------------------------- inspection
def inspect(dataset):
    path = os.path.join(dataset, "_annotations.coco.json")
    if not os.path.exists(path):
        sys.exit(f"no _annotations.coco.json in {dataset}")

    with open(path) as f:
        d = json.load(f)

    print(f"\nSCHEMA — {path}")
    print("-" * 70)
    print(f"top-level keys      {list(d.keys())}")
    print(f"images              {len(d.get('images', []))}")
    print(f"annotations         {len(d.get('annotations', []))}")
    print(f"categories          {d.get('categories')}")

    if d.get("images"):
        print(f"\nfirst image record  {d['images'][0]}")
    if d.get("annotations"):
        a = d["annotations"][0]
        print(f"first annotation    {a}")
        print(f"\nbbox field          {a.get('bbox')}")
        print("bbox convention     COCO [x, y, width, height], absolute pixels")
        print("                    (x,y = TOP-LEFT corner, not centre)")

    # sanity: are bbox values plausibly absolute rather than normalized?
    if d.get("annotations"):
        vals = [v for a in d["annotations"][:200] for v in a["bbox"]]
        if vals and max(vals) <= 1.0:
            print("\n!! WARNING: all sampled bbox values <= 1.0")
            print("   These look NORMALIZED, not absolute pixels.")

    # image files present?
    n_missing = 0
    for im in d.get("images", [])[:500]:
        if not os.path.exists(os.path.join(dataset, im["file_name"])):
            n_missing += 1
    print(f"\nof first 500 images, {n_missing} missing on disk")

    ex = os.path.join(dataset, d["images"][0]["file_name"]) if d.get("images") else None
    if ex and os.path.islink(ex):
        print(f"images are SYMLINKS -> {os.path.realpath(ex)}")
    print()


# ---------------------------------------------------------------- the audit
def audit(args):
    dataset = args.dataset.rstrip("/")
    split = os.path.basename(dataset)
    ann_path = os.path.join(dataset, "_annotations.coco.json")

    with open(ann_path) as f:
        coco = json.load(f)

    out = os.path.join(args.outdir, split)
    for sub in ("images", "annotations", "corrected", "visualizations"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    th = dict(DEFAULTS)
    for k in th:
        v = getattr(args, k.lower(), None)
        if v is not None:
            th[k] = v

    imgs_by_id = {im["id"]: im for im in coco["images"]}
    valid_cat_ids = {c["id"] for c in coco["categories"]}
    anns_by_img = defaultdict(list)
    orphan_anns = []

    for a in coco["annotations"]:
        if a["image_id"] not in imgs_by_id:
            orphan_anns.append(a)
        else:
            anns_by_img[a["image_id"]].append(a)

    rows = []
    flagged_imgs = set()
    corrected_anns = {}      # ann id -> corrected dict
    dropped_anns = set()     # ann ids removed as definitively malformed

    counts = defaultdict(int)
    counts["images_scanned"] = len(coco["images"])
    counts["annotations_scanned"] = len(coco["annotations"])
    counts["orphan_annotations"] = len(orphan_anns)

    real_dims = {}   # image id -> (W,H) read from the actual file

    for n, im in enumerate(coco["images"]):
        iid = im["id"]
        fname = im["file_name"]
        ipath = os.path.join(dataset, fname)
        W_meta, H_meta = im.get("width"), im.get("height")

        # --- image-level checks
        img_issues = []
        if not os.path.exists(ipath):
            img_issues.append(("MISSING_IMAGE_FILE", "INVALID"))
            counts["missing_image_files"] += 1
            W, H = W_meta, H_meta
        else:
            try:
                W, H = Image.open(ipath).size
                real_dims[iid] = (W, H)
                if (W, H) != (W_meta, H_meta):
                    img_issues.append(
                        (f"DIM_MISMATCH json={W_meta}x{H_meta} file={W}x{H}",
                         "INVALID"))
                    counts["dim_mismatch"] += 1
            except Exception as e:
                img_issues.append((f"UNREADABLE_IMAGE {e}", "INVALID"))
                counts["unreadable_images"] += 1
                W, H = W_meta, H_meta

        if not W or not H or W <= 0 or H <= 0:
            img_issues.append(("INVALID_DIMENSIONS", "INVALID"))
            counts["invalid_dimensions"] += 1
            W = W or 1
            H = H or 1

        anns = anns_by_img.get(iid, [])
        if not anns:
            # NOT an error -- background images are legitimate training data.
            counts["images_without_annotations"] += 1

        # --- per-annotation checks
        for a in anns:
            issues = []
            fixed = dict(a)
            bbox = a.get("bbox")

            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                issues.append(("MALFORMED_BBOX", "INVALID"))
                counts["malformed"] += 1
                rows.append(_row(fname, iid, W, H, a, None, issues,
                                 "DROP", "INVALID",
                                 "bbox not a 4-element list"))
                dropped_anns.add(a["id"])
                flagged_imgs.add(iid)
                continue

            try:
                x, y, w, h = (float(v) for v in bbox)
            except (TypeError, ValueError):
                issues.append(("NON_NUMERIC_BBOX", "INVALID"))
                counts["malformed"] += 1
                rows.append(_row(fname, iid, W, H, a, None, issues,
                                 "DROP", "INVALID", "non-numeric bbox"))
                dropped_anns.add(a["id"])
                flagged_imgs.add(iid)
                continue

            area = w * h
            ratio = area / float(W * H) if W and H else 0.0

            # category
            if a.get("category_id") not in valid_cat_ids:
                issues.append(
                    (f"INVALID_CATEGORY {a.get('category_id')}", "INVALID"))
                counts["invalid_category"] += 1

            # negative dims
            if w < 0 or h < 0:
                issues.append(("NEGATIVE_DIMENSIONS", "INVALID"))
                counts["negative_dims"] += 1
                # safe fix: this is x2y2 stored as wh, or a sign error
                nx, ny = min(x, x + w), min(y, y + h)
                fixed["bbox"] = [nx, ny, abs(w), abs(h)]

            # zero area
            elif w == 0 or h == 0:
                issues.append(("ZERO_AREA", "INVALID"))
                counts["zero_area"] += 1
                dropped_anns.add(a["id"])

            # negative coords
            if x < 0 or y < 0:
                issues.append((f"NEGATIVE_COORDS x={x:.1f} y={y:.1f}",
                               "SUSPICIOUS"))
                counts["negative_coords"] += 1

            # out of bounds
            if x + w > W + 1 or y + h > H + 1:
                issues.append(
                    (f"OUT_OF_BOUNDS x2={x+w:.1f} y2={y+h:.1f} img={W}x{H}",
                     "SUSPICIOUS"))
                counts["out_of_bounds"] += 1

            # completely outside
            if x >= W or y >= H or x + w <= 0 or y + h <= 0:
                issues.append(("ENTIRELY_OUTSIDE_IMAGE", "INVALID"))
                counts["entirely_outside"] += 1
                dropped_anns.add(a["id"])

            # tiny
            tiny = []
            if 0 < w < th["MIN_BOX_WIDTH"]:
                tiny.append(f"w={w:.2f}")
            if 0 < h < th["MIN_BOX_HEIGHT"]:
                tiny.append(f"h={h:.2f}")
            if 0 < area < th["MIN_BOX_AREA"]:
                tiny.append(f"area={area:.2f}")
            if 0 < ratio < th["MIN_BOX_AREA_RATIO"]:
                tiny.append(f"ratio={ratio:.2e}")
            if tiny:
                issues.append(("TINY_BOX " + " ".join(tiny), "SUSPICIOUS"))
                counts["tiny_boxes"] += 1

            # area field disagreement
            if "area" in a and a["area"] and abs(a["area"] - area) > 1.0:
                issues.append(
                    (f"AREA_FIELD_MISMATCH json={a['area']:.1f} calc={area:.1f}",
                     "MINOR"))
                counts["area_mismatch"] += 1
                fixed["area"] = round(area, 2)

            if not issues:
                continue

            flagged_imgs.add(iid)

            # ---- decide the action
            sev = max(SEV.get(s, 1) for _, s in issues)
            names = [i for i, _ in issues]
            joined = "; ".join(names)

            if a["id"] in dropped_anns:
                action, status = "DROP", "INVALID"
                note = "zero-area or entirely outside image; unrecoverable"

            elif any(n.startswith("TINY_BOX") for n in names):
                # NEVER auto-change a box just for being small
                action, status = "NONE", "MANUAL_REVIEW"
                note = ("small box may be a legitimate small object; "
                        "check the visualization")

            elif any(n.startswith("OUT_OF_BOUNDS") or
                     n.startswith("NEGATIVE_COORDS") for n in names):
                # safe deterministic fix: clip to image
                fx, fy, fw, fh = fixed["bbox"] if "bbox" in fixed else (x, y, w, h)
                nx, ny = max(0.0, fx), max(0.0, fy)
                nw = min(fw + min(0.0, fx), W - nx)
                nh = min(fh + min(0.0, fy), H - ny)
                if nw > 1 and nh > 1:
                    fixed["bbox"] = [round(nx, 2), round(ny, 2),
                                     round(nw, 2), round(nh, 2)]
                    fixed["area"] = round(nw * nh, 2)
                    action, status = "CLIP_TO_BOUNDS", "FIXED"
                    note = f"clipped from [{x:.1f},{y:.1f},{w:.1f},{h:.1f}]"
                else:
                    action, status = "NONE", "MANUAL_REVIEW"
                    note = "clipping would leave a degenerate box"

            elif any(n == "NEGATIVE_DIMENSIONS" for n in names):
                action, status = "NORMALIZE_COORD_ORDER", "FIXED"
                note = "negative w/h reinterpreted as corner ordering"

            elif any(n.startswith("INVALID_CATEGORY") for n in names):
                action, status = "NONE", "MANUAL_REVIEW"
                note = "category id not in categories list; cannot infer intent"

            elif any(n.startswith("AREA_FIELD_MISMATCH") for n in names):
                action, status = "RECOMPUTE_AREA", "FIXED"
                note = "area recomputed from bbox"

            else:
                action, status = "NONE", "MANUAL_REVIEW"
                note = ""

            if status == "FIXED":
                corrected_anns[a["id"]] = fixed

            rows.append(_row(fname, iid, W, H, a, (x, y, w, h), issues,
                             action, status, note))

        for txt, sv in img_issues:
            flagged_imgs.add(iid)
            rows.append({
                "image": fname, "annotation": "", "image_width": W,
                "image_height": H, "class_id": "", "bbox": "",
                "bbox_width": "", "bbox_height": "", "bbox_area": "",
                "bbox_area_ratio": "", "issue": txt, "severity": sv,
                "action": "NONE", "status": "MANUAL_REVIEW", "notes": "",
            })

        if n % 2000 == 0 and n:
            print(f"  {n}/{len(coco['images'])}", end="\r", flush=True)

    for a in orphan_anns:
        rows.append({
            "image": "", "annotation": a.get("id"), "image_width": "",
            "image_height": "", "class_id": a.get("category_id"),
            "bbox": str(a.get("bbox")), "bbox_width": "", "bbox_height": "",
            "bbox_area": "", "bbox_area_ratio": "",
            "issue": f"ORPHAN_ANNOTATION image_id={a.get('image_id')}",
            "severity": "INVALID", "action": "DROP", "status": "INVALID",
            "notes": "references an image id not present in images[]",
        })

    # ---- copy flagged samples + visualize
    print(f"\ncopying {len(flagged_imgs)} flagged samples ...")
    for iid in sorted(flagged_imgs):
        im = imgs_by_id.get(iid)
        if not im:
            continue
        fname = im["file_name"]
        src = os.path.join(dataset, fname)
        if os.path.exists(src):
            # follow symlinks so the copy is a real file
            shutil.copy2(os.path.realpath(src),
                         os.path.join(out, "images", fname))

        stem = os.path.splitext(fname)[0]
        anns = anns_by_img.get(iid, [])
        with open(os.path.join(out, "annotations", stem + ".json"), "w") as f:
            json.dump({"image": im, "annotations": anns}, f, indent=2)

        corr = [corrected_anns.get(a["id"], a) for a in anns
                if a["id"] not in dropped_anns]
        if corr != anns:
            with open(os.path.join(out, "corrected", stem + ".json"), "w") as f:
                json.dump({"image": im, "annotations": corr}, f, indent=2)

        if args.visualize and os.path.exists(src):
            _visualize(src, im, anns, dropped_anns, th,
                       os.path.join(out, "visualizations", stem + ".jpg"),
                       real_dims.get(iid))

    # ---- report
    fields = ["image", "annotation", "image_width", "image_height", "class_id",
              "bbox", "bbox_width", "bbox_height", "bbox_area",
              "bbox_area_ratio", "issue", "severity", "action", "status",
              "notes"]
    rpath = os.path.join(out, "report.csv")
    with open(rpath, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---- full corrected COCO
    new_anns = []
    for a in coco["annotations"]:
        if a["id"] in dropped_anns:
            continue
        new_anns.append(corrected_anns.get(a["id"], a))
    corrected_coco = dict(coco)
    corrected_coco["annotations"] = new_anns
    cpath = os.path.join(out, "corrected_annotations.coco.json")
    with open(cpath, "w") as f:
        json.dump(corrected_coco, f)

    _summary(counts, rows, flagged_imgs, corrected_anns, dropped_anns,
             out, rpath, cpath, th)

    # ---- second validation pass on the corrected file
    print("\nRe-validating corrected annotations ...")
    _revalidate(cpath, dataset, real_dims, valid_cat_ids)

    # ---- confirm originals untouched
    with open(ann_path) as f:
        after = json.load(f)
    same = (len(after["annotations"]) == counts["annotations_scanned"]
            and len(after["images"]) == counts["images_scanned"])
    print(f"original dataset unmodified: {'YES' if same else 'NO — INVESTIGATE'}")


def _row(fname, iid, W, H, a, box, issues, action, status, note):
    x, y, w, h = box if box else ("", "", "", "")
    area = w * h if box else ""
    return {
        "image": fname,
        "annotation": a.get("id"),
        "image_width": W, "image_height": H,
        "class_id": a.get("category_id"),
        "bbox": str(a.get("bbox")),
        "bbox_width": round(w, 2) if box else "",
        "bbox_height": round(h, 2) if box else "",
        "bbox_area": round(area, 2) if box else "",
        "bbox_area_ratio": f"{area/(W*H):.3e}" if box and W and H else "",
        "issue": "; ".join(i for i, _ in issues),
        "severity": max((s for _, s in issues), key=lambda s: SEV.get(s, 0)),
        "action": action, "status": status, "notes": note,
    }


def _visualize(src, im, anns, dropped, th, dst, real_wh):
    img = cv2.imread(os.path.realpath(src))
    if img is None:
        return
    H, W = img.shape[:2]
    scale = 1.0
    if max(H, W) < 600:
        scale = 600.0 / max(H, W)
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

    for a in anns:
        b = a.get("bbox")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            continue
        try:
            x, y, w, h = (float(v) * scale for v in b)
        except (TypeError, ValueError):
            continue

        bad = (a["id"] in dropped or w < th["MIN_BOX_WIDTH"] * scale
               or h < th["MIN_BOX_HEIGHT"] * scale
               or x < 0 or y < 0
               or x + w > W * scale + 1 or y + h > H * scale + 1)
        col = (0, 0, 255) if bad else (0, 200, 0)
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
        cv2.putText(img, f"c{a.get('category_id')} {w/scale:.0f}x{h/scale:.0f}",
                    (int(x), max(12, int(y) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    if real_wh:
        cv2.putText(img, f"{real_wh[0]}x{real_wh[1]}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.imwrite(dst, img)


def _revalidate(cpath, dataset, real_dims, valid_cats):
    with open(cpath) as f:
        d = json.load(f)
    ids = {im["id"] for im in d["images"]}
    dims = {im["id"]: (im["width"], im["height"]) for im in d["images"]}
    bad = 0
    for a in d["annotations"]:
        if a["image_id"] not in ids:
            bad += 1; continue
        if a["category_id"] not in valid_cats:
            bad += 1; continue
        b = a.get("bbox")
        if not isinstance(b, (list, tuple)) or len(b) != 4:
            bad += 1; continue
        x, y, w, h = b
        W, H = dims[a["image_id"]]
        if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > W + 1 or y + h > H + 1:
            bad += 1
    print(f"  corrected set: {len(d['annotations'])} annotations, "
          f"{bad} still invalid")
    if bad == 0:
        print("  all corrected annotations valid for RF-DETR")


def _summary(c, rows, flagged, corrected, dropped, out, rpath, cpath, th):
    st = defaultdict(int)
    for r in rows:
        st[r["status"]] += 1
    print("\n" + "=" * 46)
    print("RF-DETR Dataset Annotation Audit")
    print("-" * 46)
    print(f"Images scanned:              {c['images_scanned']}")
    print(f"Annotations scanned:         {c['annotations_scanned']}")
    print(f"Images with suspicious boxes:{len(flagged):>4}")
    print(f"Tiny boxes detected:         {c['tiny_boxes']}")
    print(f"Out-of-bounds boxes:         {c['out_of_bounds']}")
    print(f"Negative coords:             {c['negative_coords']}")
    print(f"Negative dimensions:         {c['negative_dims']}")
    print(f"Zero/invalid-area boxes:     {c['zero_area']}")
    print(f"Entirely outside image:      {c['entirely_outside']}")
    print(f"Invalid category ids:        {c['invalid_category']}")
    print(f"Malformed annotations:       {c['malformed']}")
    print(f"Orphan annotations:          {c['orphan_annotations']}")
    print(f"Missing image files:         {c['missing_image_files']}")
    print(f"Dim mismatch (json vs file): {c['dim_mismatch']}")
    print(f"Images w/o annotations:      {c['images_without_annotations']}"
          "   (background - not an error)")
    print(f"Automatically corrected:     {len(corrected)}")
    print(f"Dropped as unrecoverable:    {len(dropped)}")
    print(f"Requires manual review:      {st['MANUAL_REVIEW']}")
    print("-" * 46)
    print(f"thresholds: {th}")
    print(f"\nProblematic samples:\n  {out}")
    print(f"\nVisualizations:\n  {os.path.join(out, 'visualizations')}")
    print(f"\nReport:\n  {rpath}")
    print(f"\nCorrected COCO:\n  {cpath}")
    print("=" * 46)


# ------------------------------------------------- upstream YOLO label audit
def audit_source_labels(labels_dir, images_dir, th):
    """The COCO conversion already clamps and drops sub-pixel boxes, so the
    JSON hides issues present in the YOLO source. This checks the source."""
    print(f"\nUPSTREAM YOLO LABEL CHECK — {labels_dir}")
    print("-" * 60)
    n = neg = oob = tiny = malformed = zero = 0
    for f in os.listdir(labels_dir):
        if not f.endswith(".txt"):
            continue
        for line in open(os.path.join(labels_dir, f)):
            p = line.split()
            if not p:
                continue
            n += 1
            if len(p) != 5:
                malformed += 1
                continue
            try:
                _, cx, cy, w, h = (float(v) for v in p)
            except ValueError:
                malformed += 1
                continue
            if w <= 0 or h <= 0:
                zero += 1
            if cx < 0 or cy < 0 or w < 0 or h < 0:
                neg += 1
            if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 \
               or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
                oob += 1
            if 0 < w * h < th["MIN_BOX_AREA_RATIO"]:
                tiny += 1
    print(f"labels parsed        {n}")
    print(f"malformed lines      {malformed}")
    print(f"zero/negative w,h    {zero}")
    print(f"negative values      {neg}")
    print(f"outside [0,1] bounds {oob}   <- clamped away by yolo2coco.py")
    print(f"tiny by area ratio   {tiny}")
    print("-" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="e.g. datasets_coco/train")
    ap.add_argument("--outdir", default="problematic_annotations")
    ap.add_argument("--inspect", action="store_true",
                    help="print the schema and exit, change nothing")
    ap.add_argument("--source-labels", default=None,
                    help="also audit upstream YOLO .txt labels")
    ap.add_argument("--no-visualize", dest="visualize", action="store_false")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.lower()}", type=float, default=None,
                        help=f"default {v}")
    args = ap.parse_args()

    inspect(args.dataset)
    if args.inspect:
        return

    if args.source_labels:
        th = dict(DEFAULTS)
        audit_source_labels(args.source_labels, None, th)

    audit(args)


if __name__ == "__main__":
    main()
