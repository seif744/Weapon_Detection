#!/usr/bin/env python3
"""
video_eval.py -- turns "it looked better" into a number.

You annotate, per video, the time ranges where a weapon is visible.
This measures, at several confidence thresholds:

    recall            = fraction of weapon-visible SECONDS with >=1 detection
    false alarms/min  = detections in time ranges with NO weapon

That's the pair of numbers you report. mAP on a val split does not
measure the deployment domain; this does.

--- Step 1: annotate ------------------------------------------------------
Make a CSV called ranges.csv. One row per contiguous weapon-visible span.
Videos with NO weapon at all: still list them, with a single row 0,0.

    video,start_sec,end_sec
    weapon_sample.webm,3.2,9.8
    weapon_sample.webm,14.0,21.5
    store_clip.mp4,0,0

Scrub each clip in VLC and write down start/end. ~10 min per video.

--- Step 2: run -----------------------------------------------------------
    python video_eval.py \
        --model runs/detect/merged_v1/weights/best.pt \
        --videos-dir datasets/test_videos \
        --ranges ranges.csv \
        --classes 1

    (--classes 1 = firearm only. Use "1 0" to include knife.)

Results print per-video and overall, and dump to eval_results.csv.
Re-run this exact command after ANY change. If the numbers don't move,
the change didn't work.
"""

import argparse
import csv
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO

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
    ap.add_argument("--model", required=True)
    ap.add_argument("--videos-dir", required=True)
    ap.add_argument("--ranges", required=True)
    ap.add_argument("--classes", nargs="+", type=int, default=[1],
                    help="class ids to count as a weapon (merged_v1: 0=knife 1=firearm 2=hammer)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--stride", type=int, default=5,
                    help="evaluate every Nth frame; 5 is plenty and 5x faster")
    ap.add_argument("--out", default="eval_results.csv")
    args = ap.parse_args()

    ranges = load_ranges(args.ranges)
    model = YOLO(args.model)
    keep = set(args.classes)

    # per-threshold accumulators
    hit_sec = defaultdict(float)     # weapon-visible seconds with a detection
    tot_sec = defaultdict(float)     # total weapon-visible seconds
    fa_count = defaultdict(int)      # detections outside any range
    neg_sec = defaultdict(float)     # total weapon-free seconds

    rows = []

    for video, vranges in sorted(ranges.items()):
        path = os.path.join(args.videos_dir, video)
        if not os.path.exists(path):
            print(f"!! missing: {path}")
            continue

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step_sec = args.stride / fps

        v_hit = defaultdict(float)
        v_tot = 0.0
        v_fa = defaultdict(int)
        v_neg = 0.0

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
            res = model.predict(source=frame, imgsz=args.imgsz,
                                conf=0.01, verbose=False)[0]

            best = 0.0
            if res.boxes is not None and len(res.boxes) > 0:
                confs = res.boxes.conf.cpu().numpy()
                clss = res.boxes.cls.cpu().numpy().astype(int)
                for c, k in zip(confs, clss):
                    if int(k) in keep:
                        best = max(best, float(c))

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

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video", "conf", "recall", "fa_per_min"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
