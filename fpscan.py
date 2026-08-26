# fpscan.py
import sys, glob, numpy as np
from PIL import Image
from rfdetr import RFDETRLarge
m = RFDETRLarge(pretrain_weights="runs_rfdetr/rfdetr_l_v4/checkpoint_best_total.pth",
                resolution=704, num_classes=4)
for p in sorted(glob.glob(sys.argv[1])):
    d = m.predict(Image.open(p).convert("RGB"), threshold=0.25)
    for c, s, b in zip(d.class_id, d.confidence, d.xyxy):
        if s > 0.5:
            print(f"{p}  cls={c}  {s:.3f}  {[int(v) for v in b]}")
