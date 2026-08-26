#!/usr/bin/env python3
"""
frame_probe.py -- why does a frame detect standalone but not in the video?

Reads the video through cv2.VideoCapture exactly the way render_rfdetr.py
does, runs the model on the frame you name, and dumps that frame to disk so
you can compare against however you extracted it before.

    python frame_probe.py \
        --checkpoint runs_rfdetr/rfdetr_m_v3/checkpoint_best_total.pth \
        --video datasets/test_videos/evaluation.mp4 \
        --frame 88

    # scan a window instead of one frame
    python frame_probe.py --checkpoint ... --video ... --frame 88 --window 10
"""

import argparse

import cv2
import numpy as np

NAMES = {0: "none", 1: "knife", 2: "firearm", 3: "hammer"}


def report(model, frame, tag):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    out = []
    for th in (0.01, 0.25, 0.50, 0.76):
        d = model.predict(rgb, threshold=th)
        n = 0 if d is None else len(d)
        if n:
            cf = np.asarray(d.confidence)
            kd = np.asarray(d.class_id).astype(int)
            order = np.argsort(-cf)[:5]
            top = [f"{NAMES.get(int(kd[i]), kd[i])}:{cf[i]:.3f}"
                   for i in order]
        else:
            top = []
        out.append(f"    th={th:<5} {n:>3} dets   {' '.join(top)}")
    print(f"  {tag}")
    print("\n".join(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--frame", type=int, required=True)
    ap.add_argument("--window", type=int, default=0,
                    help="also probe N frames either side")
    ap.add_argument("--model", default="medium",
                    choices=["nano", "small", "medium", "base", "large"])
    args = ap.parse_args()

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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\nvideo: {W}x{H}, {fps:.3f} fps reported, {total} frames reported")

    lo = max(0, args.frame - args.window)
    hi = args.frame + args.window

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if lo <= i <= hi:
            out = f"frame_{i:05d}_from_cv2.png"
            cv2.imwrite(out, frame)
            print(f"\nframe {i}  (t={i/fps:.3f}s)  saved -> {out}")
            report(model, frame, "as decoded by cv2.VideoCapture")
        if i > hi:
            break
        i += 1
    cap.release()

    print("\n" + "-" * 62)
    print("Now run the SAME model on the PNG this just wrote:")
    print(f"  python predict_images.py --checkpoint {args.checkpoint} \\")
    print("      --images . --conf 0.01")
    print()
    print("If the PNG scores higher than the in-video number above, the")
    print("difference is decoding (cv2 vs ffmpeg colour conversion, or a")
    print("JPEG round-trip on however you saved the frame originally).")
    print("If they match, the frame you tested before was a different frame.")


if __name__ == "__main__":
    main()
