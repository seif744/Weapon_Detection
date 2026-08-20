#!/usr/bin/env python3
"""
plot_rfdetr.py -- rebuild the training curves from RF-DETR's metrics.csv.

RF-DETR logs via Lightning and does not write a results.png like Ultralytics.
This makes one.

Usage:
    pip install matplotlib          # if not already in rfdetr_env
    python plot_rfdetr.py runs_rfdetr/rfdetr_m_v1/metrics.csv

Writes results.png next to the csv.

The csv is sparse -- most rows only carry the learning rate. Val rows carry
val metrics, epoch-end rows carry train losses. This pulls each column
independently and takes the last value per epoch.
"""

import csv
import os
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def series(rows, col):
    """Last non-empty value of `col` per epoch -> (epochs, values)."""
    d = OrderedDict()
    for r in rows:
        v = r.get(col, "")
        if v not in ("", None):
            try:
                d[int(float(r["epoch"]))] = float(v)
            except (ValueError, KeyError):
                continue
    return list(d.keys()), list(d.values())


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "metrics.csv"
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # the final row holds test metrics, not a training epoch -- drop it
    rows = [r for r in rows if not r.get("test/mAP_50", "")]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(os.path.dirname(path) or "rfdetr", fontsize=13)

    # --- losses
    a = ax[0][0]
    for col, label in [("train/loss", "train"), ("val/loss", "val")]:
        e, v = series(rows, col)
        if e:
            a.plot(e, v, marker="o", ms=3, label=label)
    a.set_title("loss")
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=.3)

    # --- overall mAP
    a = ax[0][1]
    for col, label in [("val/mAP_50", "mAP50"),
                       ("val/mAP_50_95", "mAP50-95"),
                       ("val/ema_mAP_50_95", "mAP50-95 (EMA)")]:
        e, v = series(rows, col)
        if e:
            a.plot(e, v, marker="o", ms=3, label=label)
    a.set_title("val mAP")
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=.3)

    # --- per class
    a = ax[1][0]
    for col in [c for c in rows[0] if c.startswith("val/AP/")]:
        e, v = series(rows, col)
        if e:
            a.plot(e, v, marker="o", ms=3, label=col.split("/")[-1])
    a.set_title("val AP 50:95 per class")
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=.3)

    # --- precision / recall
    a = ax[1][1]
    for col, label in [("val/precision", "precision"),
                       ("val/recall", "recall"),
                       ("val/F1", "F1")]:
        e, v = series(rows, col)
        if e:
            a.plot(e, v, marker="o", ms=3, label=label)
    a.set_title("val P / R / F1")
    a.set_xlabel("epoch")
    a.legend()
    a.grid(alpha=.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(path) or ".", "results.png")
    plt.savefig(out, dpi=130)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
