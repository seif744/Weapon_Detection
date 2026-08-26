#!/usr/bin/env python3
"""
predict_images.py -- run RF-DETR over a folder of images.

Built for auditing false positives: point it at a folder of frames the model
got wrong, and it tells you what it fired on and how confident it was.

Usage:
    source rfdetr_env/bin/activate

    python predict_images.py \
        --checkpoint runs_rfdetr/rfdetr_m_v3/checkpoint_best_total.pth \
        --images false_preds \
        --conf 0.25

Prints a per-image table and writes annotated copies to <images>_annotated/.
Exit summary counts how many images fired at all, which is the number you
actually care about on a false-positive set (it should be zero).
"""

import argparse
import os

import cv2
import numpy as np

NAMES = {1: "knife", 2: "firearm", 3: "hammer"}
COLORS = {1: (80, 200, 255), 2: (60, 60, 255), 3: (120, 255, 120)}  # BGR
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
        ".JPG", ".JPEG", ".PNG")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="folder of images")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="*", type=int, default=[],
                    help="restrict to class ids (1=knife 2=firearm 3=hammer); "
                         "empty = all")
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--outdir", default=None,
                    help="default: <images>_annotated")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="upscale output images so boxes are readable")
    args = ap.parse_args()

    outdir = args.outdir or (args.images.rstrip("/") + "_annotated")
    os.makedirs(outdir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.images) if f.endswith(EXTS))
    if not files:
        raise SystemExit(f"no images found in {args.images}")

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

    keep = set(args.classes)
    n_fired = 0
    n_dets = 0
    max_conf_overall = 0.0
    worst = None

    print(f"\n{'image':<40} {'dets':>5} {'max conf':>9}  classes")
    print("-" * 78)

    for f in files:
        path = os.path.join(args.images, f)
        img = cv2.imread(path)
        if img is None:
            print(f"{f[:38]:<40} {'--':>5} {'unreadable':>9}")
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        det = model.predict(rgb, threshold=args.conf)

        rows = []
        if det is not None and len(det) > 0:
            b = np.asarray(det.xyxy)
            cf = np.asarray(det.confidence)
            kd = np.asarray(det.class_id).astype(int)
            for box, c, k in zip(b, cf, kd):
                if keep and int(k) not in keep:
                    continue
                rows.append((box, float(c), int(k)))

        if rows:
            n_fired += 1
            n_dets += len(rows)
            mc = max(r[1] for r in rows)
            if mc > max_conf_overall:
                max_conf_overall = mc
                worst = f
            names = ", ".join(sorted({f"{NAMES.get(k,k)}" for _, _, k in rows}))
            print(f"{f[:38]:<40} {len(rows):>5} {mc:>9.3f}  {names}")
        else:
            print(f"{f[:38]:<40} {0:>5} {'-':>9}")

        # annotate and save
        for box, c, k in rows:
            x1, y1, x2, y2 = [int(v) for v in box]
            col = COLORS.get(k, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            label = f"{NAMES.get(k, k)} {c:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
                        cv2.LINE_AA)

        if args.scale != 1.0:
            img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(outdir, f), img)

    print("-" * 78)
    print(f"{len(files)} images, {n_fired} fired, {n_dets} detections "
          f"at conf >= {args.conf}")
    if worst:
        print(f"highest confidence: {max_conf_overall:.3f} on {worst}")
    print(f"\nannotated copies in {outdir}/")

    if n_fired:
        print("\nOn a false-positive set, every one of these is an error.")
        print("Raising --conf only helps if max conf is below your threshold.")


if __name__ == "__main__":
    main()
