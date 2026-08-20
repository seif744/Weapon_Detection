#!/usr/bin/env python3
"""
video_eval_rfdetr.py -- same measurement as video_eval.py, for RF-DETR.

Output format is identical so the numbers are directly comparable to your
YOLO baseline (0.20 recall @ conf 0.25).

CLASS IDS DIFFER FROM YOLO. Your COCO json is 1-indexed with a placeholder:
    0 = placeholder    1 = knife    2 = firearm    3 = hammer
So firearm is --classes 2 here, not 1. Verify with --debug on the first run.

Usage:
    source rfdetr_env/bin/activate
    python video_eval_rfdetr.py \
        --checkpoint runs_rfdetr/rfdetr_m_v1/checkpoint_best_total.pth \
        --videos-dir datasets/test_videos \
        --ranges ranges.csv \
        --classes 2

    # first run: confirm which class id firearms actually come back as
    python video_eval_rfdetr.py --checkpoint ... --videos-dir ... \
        --ranges ranges.csv --classes 2 --debug
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
import numpy as np

THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.35, 0.50]


def load_ranges(path):
    ranges = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            v = row["video"].strip()
            s, e = float(row["start_sec"]), float(row["end_sec"])
            if e > s:
                ranges[v].append((s, e))
            else:
                ranges.setdefault(v, [])
    return ranges


def in_any_range(t, ranges):
    return any(s <= t <= e for s, e in ranges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--classes", nargs="+", type=int, default=[2],
                    help="class ids counting as a weapon (COCO: 1=knife 2=firearm 3=hammer)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--out", default="eval_results_rfdetr.csv")
    ap.add_argument("--debug", action="store_true",
                    help="print every class id seen, then exit after 40 frames")
    args = ap.parse_args()

    from rfdetr import (RFDETRNano, RFDETRSmall, RFDETRMedium,
                        RFDETRBase, RFDETRLarge)
    cls = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium,
           "base": RFDETRBase, "large": RFDETRLarge}[args.model]

    print(f"loading {args.checkpoint} ...")
    model = cls(pretrain_weights=args.checkpoint)
    try:
        model.optimize_for_inference()
    except Exception as e:
        print(f"  (optimize_for_inference skipped: {e})")

    ranges = load_ranges(args.ranges)
    keep = set(args.classes)

    hit_sec = defaultdict(float)
    tot_sec = defaultdict(float)
    fa_count = defaultdict(int)
    neg_sec = defaultdict(float)
    seen_ids = defaultdict(int)
    rows = []
    dbg_frames = 0

    for video, vranges in sorted(ranges.items()):
        path = os.path.join(args.videos_dir, video)
        if not os.path.exists(path):
            print(f"!! missing: {path}")
            continue

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_sec = args.stride / fps

        v_hit = defaultdict(float)
        v_fa = defaultdict(int)
        v_tot = v_neg = 0.0
        idx = 0

        print(f"\n{video} ...", end="", flush=True)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride:
                idx += 1
                continue

            t = idx / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # low threshold once, filter in python -> one forward pass per frame
            det = model.predict(rgb, threshold=0.01)

            best = 0.0
            if det is not None and len(det) > 0:
                confs = np.asarray(det.confidence)
                ids = np.asarray(det.class_id).astype(int)
                for c, k in zip(confs, ids):
                    seen_ids[int(k)] += 1
                    if int(k) in keep:
                        best = max(best, float(c))

            if args.debug:
                dbg_frames += 1
                if dbg_frames >= 40:
                    cap.release()
                    print("\n\n--- DEBUG: class ids seen (id: count) ---")
                    for k in sorted(seen_ids):
                        print(f"  class_id {k}: {seen_ids[k]} detections")
                    print("\nWhichever id dominates on gun footage is firearm.")
                    print("Re-run without --debug using --classes <that id>.")
                    return

            positive = in_any_range(t, vranges)
            if positive:
                v_tot += step_sec
            else:
                v_neg += step_sec

            for th in THRESHOLDS:
                fired = best >= th
                if positive and fired:
                    v_hit[th] += step_sec
                elif not positive and fired:
                    v_fa[th] += 1
            idx += 1
        cap.release()
        print(" done")

        for th in THRESHOLDS:
            hit_sec[th] += v_hit[th]
            fa_count[th] += v_fa[th]
            tot_sec[th] += v_tot
            neg_sec[th] += v_neg

        print(f"  {'conf':>6} {'recall':>8} {'FA/min':>8}")
        for th in THRESHOLDS:
            r = v_hit[th] / v_tot if v_tot else float("nan")
            fpm = v_fa[th] / (v_neg / 60) if v_neg else 0.0
            print(f"  {th:>6.2f} {r:>8.2f} {fpm:>8.1f}")
            rows.append({"video": video, "conf": th,
                         "recall": round(r, 4) if v_tot else "",
                         "fa_per_min": round(fpm, 2)})

    print("\n" + "=" * 40)
    print("OVERALL")
    print(f"{'conf':>6} {'recall':>8} {'FA/min':>8}")
    for th in THRESHOLDS:
        r = hit_sec[th] / tot_sec[th] if tot_sec[th] else float("nan")
        fpm = fa_count[th] / (neg_sec[th] / 60) if neg_sec[th] else 0.0
        print(f"{th:>6.2f} {r:>8.2f} {fpm:>8.1f}")
        rows.append({"video": "ALL", "conf": th,
                     "recall": round(r, 4) if tot_sec[th] else "",
                     "fa_per_min": round(fpm, 2)})

    print("\nclass ids seen across all frames:",
          {k: seen_ids[k] for k in sorted(seen_ids)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "conf", "recall", "fa_per_min"])
        w.writeheader()
        w.writerows(rows)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
