import cv2, glob, os, random, shutil

random.seed(0)
os.makedirs('degraded/images', exist_ok=True)
os.makedirs('degraded/labels', exist_ok=True)

files = glob.glob('datasets_merged/images/train/*')
sample = random.sample(files, min(8000, len(files)))

made = 0
for p in sample:
    base = os.path.splitext(os.path.basename(p))[0]
    lbl = f'datasets_merged/labels/train/{base}.txt'
    if not os.path.exists(lbl):
        continue
    img = cv2.imread(p)
    if img is None:
        continue
    h, w = img.shape[:2]
    s = random.uniform(0.3, 0.6)
    img = cv2.resize(img, (max(1, int(w*s)), max(1, int(h*s))), interpolation=cv2.INTER_AREA)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.5, 1.5))
    cv2.imwrite(f'degraded/images/deg_{base}.jpg', img,
                [cv2.IMWRITE_JPEG_QUALITY, random.randint(25, 50)])
    shutil.copy(lbl, f'degraded/labels/deg_{base}.txt')
    made += 1

print(f'made {made}')
