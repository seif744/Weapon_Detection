#!/usr/bin/env python3
"""
render_rfdetr.py -- burn RF-DETR detections into video so you can watch them.

Usage:
    source rfdetr_env/bin/activate

    # one clip
    python render_rfdetr.py \
        --checkpoint runs_rfdetr/rfdetr_m_v1/checkpoint_best_total.pth \
        --video datasets/test_videos/weapon_sample.webm \
        --conf 0.25

    # every clip in a folder
    python render_rfdetr.py \
        --checkpoint runs_rfdetr/rfdetr_m_v1/checkpoint_best_total.pth \
        --videos-dir datasets/test_videos \
        --conf 0.25

Output goes to ./rendered/ as .mp4 (h264 via mp4v -- plays anywhere).
Burns in the frame number and per-frame detection count so you can pause on
a miss and find the exact timestamp.
"""

import argparse
import os

import cv2
import numpy as np

NAMES = {1: "knife", 2: "firearm", 3: "hammer"}
COLORS = {1: (80, 200, 255), 2: (60, 60, 255), 3: (120, 255, 120)}  # BGR


def render(model, src, dst, conf, classes, scale):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"!! could not open {src}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    oW, oH = int(W * scale), int(H * scale)

    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (oW, oH))

    n = hits = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        det = model.predict(rgb, threshold=conf)

        found = 0
        if det is not None and len(det) > 0:
            for box, c, k in zip(np.asarray(det.xyxy),
                                 np.asarray(det.confidence),
                                 np.asarray(det.class_id).astype(int)):
                if classes and int(k) not in classes:
                    continue
                found += 1
                x1, y1, x2, y2 = [int(v) for v in box]
                col = COLORS.get(int(k), (255, 255, 255))
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
                label = f"{NAMES.get(int(k), k)} {c:.2f}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 6),
                              (x1 + tw + 4, y1), col, -1)
                cv2.putText(frame, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                            cv2.LINE_AA)

        if found:
            hits += 1

        # HUD: frame number, timestamp, detection count
        hud = f"f{n}  t={n/fps:5.2f}s  det={found}"
        cv2.rectangle(frame, (0, 0), (230, 22), (0, 0, 0), -1)
        cv2.putText(frame, hud, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0) if found else (0, 0, 255), 1, cv2.LINE_AA)

        if scale != 1.0:
            frame = cv2.resize(frame, (oW, oH))
        writer.write(frame)
        n += 1

        if n % 100 == 0:
            print(f"  {n} frames", end="\r", flush=True)

    cap.release()
    writer.release()
    pct = 100 * hits / n if n else 0
    print(f"  {os.path.basename(dst)}: {n} frames, "
          f"{hits} with detections ({pct:.0f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video")
    ap.add_argument("--videos-dir")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="*", type=int, default=[],
                    help="restrict drawing (1=knife 2=firearm 3=hammer); "
                         "empty draws all")
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "base", "large"])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="upscale output, e.g. 2.0 for tiny CCTV clips")
    ap.add_argument("--outdir", default="rendered")
    args = ap.parse_args()

    if not args.video and not args.videos_dir:
        raise SystemExit("pass --video or --videos-dir")

    os.makedirs(args.outdir, exist_ok=True)

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

    if args.video:
        srcs = [args.video]
    else:
        srcs = [os.path.join(args.videos_dir, f)
                for f in sorted(os.listdir(args.videos_dir))
                if f.lower().endswith((".mp4", ".webm", ".avi", ".mov", ".mkv"))]

    keep = set(args.classes)
    for s in srcs:
        stem = os.path.splitext(os.path.basename(s))[0]
        dst = os.path.join(args.outdir, f"{stem}_conf{args.conf}.mp4")
        print(f"\n{os.path.basename(s)} ...")
        render(model, s, dst, args.conf, keep, args.scale)

    print(f"\nwritten to {args.outdir}/")


if __name__ == "__main__":
    main()

