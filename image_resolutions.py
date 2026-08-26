#!/usr/bin/env python3
"""
image_resolutions.py -- resolution stats for a YOLO or COCO dataset.

Reads image headers only (PIL doesn't decode pixels for .size), so it's fast
even on 30k files.

    python image_resolutions.py --root datasets_neg
    python image_resolutions.py --root datasets_neg_coco --coco
    python image_resolutions.py --root datasets_neg --by-prefix

--by-prefix groups by filename prefix, which maps to source dataset. That's
usually the more interesting cut: it tells you whether one source is dragging
the distribution.
"""

import argparse
import os
import re
import statistics as st
from collections import Counter, defaultdict

from PIL import Image

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp",
        ".JPG", ".JPEG", ".PNG")


def gather(root, coco):
    """-> {split: [(w, h, filename), ...]}"""
    out = defaultdict(list)
    if coco:
        splits = [d for d in ("train", "valid", "test")
                  if os.path.isdir(os.path.join(root, d))]
        dirs = {s: os.path.join(root, s) for s in splits}
    else:
        base = os.path.join(root, "images")
        if not os.path.isdir(base):
            base = root
        splits = [d for d in os.listdir(base)
                  if os.path.isdir(os.path.join(base, d))] or ["."]
        dirs = {s: os.path.join(base, s) for s in splits}

    for split, d in dirs.items():
        files = [f for f in os.listdir(d) if f.endswith(EXTS)]
        for n, f in enumerate(files):
            try:
                w, h = Image.open(os.path.join(d, f)).size
                out[split].append((w, h, f))
            except Exception:
                pass
            if n % 3000 == 0 and n:
                print(f"  {split}: {n}/{len(files)}", end="\r", flush=True)
    return out


def summarize(name, rows):
    if not rows:
        return
    ws = [r[0] for r in rows]
    hs = [r[1] for r in rows]
    px = [r[0] * r[1] for r in rows]
    ar = [r[0] / r[1] for r in rows]

    imin = rows[px.index(min(px))]
    imax = rows[px.index(max(px))]

    print(f"\n{name}  ({len(rows)} images)")
    print("-" * 68)
    print(f"{'':10}{'min':>10}{'mean':>10}{'median':>10}{'max':>10}")
    print(f"{'width':10}{min(ws):>10}{st.mean(ws):>10.0f}"
          f"{st.median(ws):>10.0f}{max(ws):>10}")
    print(f"{'height':10}{min(hs):>10}{st.mean(hs):>10.0f}"
          f"{st.median(hs):>10.0f}{max(hs):>10}")
    print(f"{'megapix':10}{min(px)/1e6:>10.2f}{st.mean(px)/1e6:>10.2f}"
          f"{st.median(px)/1e6:>10.2f}{max(px)/1e6:>10.2f}")
    print(f"{'aspect':10}{min(ar):>10.2f}{st.mean(ar):>10.2f}"
          f"{st.median(ar):>10.2f}{max(ar):>10.2f}")
    print(f"\nsmallest  {imin[0]}x{imin[1]}  {imin[2][:44]}")
    print(f"largest   {imax[0]}x{imax[1]}  {imax[2][:44]}")

    c = Counter((r[0], r[1]) for r in rows)
    print("\nmost common dimensions:")
    for (w, h), n in c.most_common(8):
        print(f"  {w}x{h:<6} {n:>7}  {100*n/len(rows):5.1f}%")

    # how many are below the model input, i.e. get UPSCALED
    for res in (576, 704):
        below = sum(1 for r in rows if max(r[0], r[1]) < res)
        print(f"  smaller than {res}px on the long edge: "
              f"{below} ({100*below/len(rows):.1f}%)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--coco", action="store_true",
                    help="COCO layout (images sit beside the json)")
    ap.add_argument("--by-prefix", action="store_true",
                    help="also break down by source filename prefix")
    args = ap.parse_args()

    data = gather(args.root, args.coco)
    allrows = [r for rows in data.values() for r in rows]

    for split in sorted(data):
        summarize(f"SPLIT: {split}", data[split])

    if len(data) > 1:
        summarize("ALL SPLITS COMBINED", allrows)

    if args.by_prefix:
        groups = defaultdict(list)
        for r in allrows:
            p = re.split(r"[0-9(]", r[2], maxsplit=1)[0] or "?"
            groups[p].append(r)
        print("\n" + "=" * 68)
        print("BY SOURCE PREFIX")
        print("=" * 68)
        print(f"{'prefix':<24}{'n':>7}{'med w':>8}{'med h':>8}"
              f"{'min':>12}{'max':>12}")
        for p, rows in sorted(groups.items(), key=lambda x: -len(x[1])):
            if len(rows) < 20:
                continue
            ws = [r[0] for r in rows]
            hs = [r[1] for r in rows]
            px = [r[0] * r[1] for r in rows]
            lo = rows[px.index(min(px))]
            hi = rows[px.index(max(px))]
            print(f"{p[:23]:<24}{len(rows):>7}{st.median(ws):>8.0f}"
                  f"{st.median(hs):>8.0f}{lo[0]}x{lo[1]:<10}{hi[0]}x{hi[1]}")


if __name__ == "__main__":
    main()
