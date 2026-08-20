#!/usr/bin/env python3
"""
confusion_rfdetr.py -- build a detection confusion matrix for RF-DETR.

RF-DETR ships no confusion matrix. This reproduces the Ultralytics one so the
numbers are comparable to your old yolo_weapons matrix (writeup section 6.1).

CONVENTION MATCHES ULTRALYTICS: Predicted on Y, True on X, column-normalized.
That is the transpose of the sklearn convention -- read columns, not rows.
Column "firearm" tells you what fraction of true firearms were predicted as
each class. The "background" column is false positives; the "background" row
is misses.

Usage:
    source rfdetr_env/bin/activate
    pip install matplotlib
    python confusion_rfdetr.py \
        --checkpoint runs_rfdetr/rfdetr_m_v1/checkpoint_best_total.pth \
        --dataset datasets_coco/valid \
        --conf 0.25 --iou 0.45
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def box_iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True,
                    help="folder with _annotations.coco.json and images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="only process N images (quick smoke test)")
    args = ap.parse_args()

    ann_path = os.path.join(args.dataset, "_annotations.coco.json")
    with open(ann_path) as f:
        coco = json.load(f)

    # real classes only -- drop the id-0 placeholder
    cats = sorted((c for c in coco["categories"] if c["id"] != 0),
                  key=lambda c: c["id"])
    cat_ids = [c["id"] for c in cats]
    names = [c["name"] for c in cats] + ["background"]
    idx_of = {cid: i for i, cid in enumerate(cat_ids)}
    nc = len(cat_ids)
    BG = nc

    gt = {}
    for a in coco["annotations"]:
        if a["category_id"] not in idx_of:
            continue
        x, y, w, h = a["bbox"]
        gt.setdefault(a["image_id"], []).append(
            (idx_of[a["category_id"]], [x, y, x + w, y + h]))

    images = coco["images"]
    if args.limit:
        images = images[:args.limit]

    from rfdetr import (RFDETRNano, RFDETRSmall, RFDETRMedium,
                        RFDETRBase, RFDETRLarge)
    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium,
           "base": RFDETRBase, "large": RFDETRLarge}[args.model]

    print(f"loading {args.checkpoint} ...")
    model = cls(pretrain_weights=args.checkpoint)
    try:
        model.optimize_for_inference()
    except Exception:
        pass

    m = np.zeros((nc + 1, nc + 1), dtype=np.int64)
    unseen = set()

    for n, img in enumerate(images):
        p = os.path.join(args.dataset, img["file_name"])
        if not os.path.exists(p):
            continue
        arr = np.array(Image.open(p).convert("RGB"))

        det = model.predict(arr, threshold=args.conf)
        if det is not None and len(det) > 0:
            pb = np.asarray(det.xyxy, dtype=float)
            pc_raw = np.asarray(det.class_id).astype(int)
            keep = np.array([c in idx_of for c in pc_raw])
            for c in pc_raw[~keep] if len(pc_raw) else []:
                unseen.add(int(c))
            pb, pc = pb[keep], np.array([idx_of[c] for c in pc_raw[keep]])
        else:
            pb, pc = np.zeros((0, 4)), np.zeros(0, dtype=int)

        g = gt.get(img["id"], [])
        gc = np.array([c for c, _ in g], dtype=int)
        gb = np.array([b for _, b in g], dtype=float).reshape(-1, 4)

        ious = box_iou(gb, pb)
        used_g, used_p = set(), set()
        if ious.size:
            order = np.dstack(np.unravel_index(
                np.argsort(-ious, axis=None), ious.shape))[0]
            for gi, pi in order:
                if ious[gi, pi] < args.iou:
                    break
                if gi in used_g or pi in used_p:
                    continue
                used_g.add(gi)
                used_p.add(pi)
                m[pc[pi], gc[gi]] += 1          # [predicted, true]

        for gi in range(len(gc)):
            if gi not in used_g:
                m[BG, gc[gi]] += 1              # missed
        for pi in range(len(pc)):
            if pi not in used_p:
                m[pc[pi], BG] += 1              # false positive

        if n % 200 == 0:
            print(f"  {n}/{len(images)}", end="\r", flush=True)

    if unseen:
        print(f"\nWARNING: unmapped predicted class ids seen: {sorted(unseen)}")

    # column-normalize, Ultralytics style
    col = m.sum(0, keepdims=True).astype(float)
    norm = np.divide(m, np.where(col == 0, 1, col))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for a, data, title, fmt in [
        (axes[0], m, f"counts (conf={args.conf}, iou={args.iou})", "d"),
        (axes[1], norm, "column-normalized", ".2f"),
    ]:
        im = a.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        a.set_xticks(range(nc + 1), names, rotation=45, ha="right")
        a.set_yticks(range(nc + 1), names)
        a.set_xlabel("True")
        a.set_ylabel("Predicted")
        a.set_title(title)
        for i in range(nc + 1):
            for j in range(nc + 1):
                v = data[i, j]
                a.text(j, i, f"{v:{fmt}}", ha="center", va="center",
                       fontsize=10,
                       color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=axes, fraction=0.02)

    out = args.out or os.path.join(
        os.path.dirname(args.checkpoint) or ".", "confusion_matrix.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nwritten to {out}\n")

    print(f"{'class':<12} {'correct':>8} {'missed':>8} {'FP':>8}")
    for i, nm in enumerate(names[:-1]):
        total = m[:, i].sum()
        print(f"{nm:<12} {m[i, i] / total if total else 0:>8.2f} "
              f"{m[BG, i] / total if total else 0:>8.2f} {m[i, BG]:>8d}")


if __name__ == "__main__":
    main()
