import glob, os
from collections import Counter
from PIL import Image

bins = Counter()
for lp in glob.glob('datasets_merged/labels/train/*.txt'):
    ip = None
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG'):
        p = lp.replace('/labels/', '/images/')[:-4] + ext
        if os.path.exists(p):
            ip = p
            break
    if not ip:
        continue
    try:
        W, H = Image.open(ip).size
    except Exception:
        continue
    for line in open(lp):
        f = line.split()
        if len(f) == 5 and f[0] == '1':
            r = (float(f[3]) * W) / (float(f[4]) * H)
            if r < 0.8:    bins['tall (<0.8)'] += 1
            elif r < 1.3:  bins['SQUARE 0.8-1.3  <- muzzle-on'] += 1
            elif r < 2.0:  bins['moderate 1.3-2.0'] += 1
            elif r < 3.0:  bins['side profile 2.0-3.0'] += 1
            else:          bins['long (>3.0)'] += 1

total = sum(bins.values())
for k in sorted(bins, key=lambda x: -bins[x]):
    print(f"{k:32s} {bins[k]:7d}  {100*bins[k]/total:5.1f}%")
print(f"{'TOTAL':32s} {total:7d}")

