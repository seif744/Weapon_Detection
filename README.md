# Weapon Detection — RF-DETR

CCTV weapon detection. Three classes: knife, firearm, hammer.

Server: `easemyai-a6000` · RTX 6000 Ada (48 GB) · CUDA 13.0 · project root `~/weapons_project`

---

## Documentation

| Document | Covers |
|---|---|
| `RFDETR_Weapon_Detection_Validation_Report.pdf` | **Results, metrics and analysis, v1–v5.** Authoritative for all performance figures. |
| This README | Setup, layout, scripts, operational pitfalls |


---

## Quick start

```bash
cd ~/weapons_project
source rfdetr_env/bin/activate

python render_rfdetr.py \
    --checkpoint runs_rfdetr/rfdetr_m_v4/checkpoint_best_total.pth \
    --video test_videos/evaluation.mp4 \
    --conf 0.5 --model large --outdir rendered
```

**`--model large` is not optional** for v4 and v5 checkpoints. See Pitfalls.

---

## Environments

Two virtualenvs, deliberately separate. Do not merge them — Ultralytics and RF-DETR want
different torch builds and installing one into the other breaks the other.

| Env | Purpose | Key packages |
|---|---|---|
| `rfdetr_env` | Everything current | rfdetr 1.9.3, torch 2.13.0+cu130 |
| `yoloenv` | Legacy YOLOv8 baseline only | ultralytics 8.4.121 |


```bash
python3 -m venv yoloenv && source yoloenv/bin/activate && pip install ultralytics
```

---

## Directory layout

```
~/weapons_project/
├── datasets_merged/     SOURCE OF TRUTH. YOLO format, 27,767 train images
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/        *.txt, normalized cxcywh
├── datasets_neg/        = datasets_merged + 784 SOMPT background frames
├── datasets_coco/       RF-DETR format. SYMLINKS into datasets_merged
├── datasets_neg_coco/   RF-DETR format. SYMLINKS into datasets_neg
├── sompt_neg/           the 784 background frames + empty labels
├── sompt_screen/        v4 screening output — 255 documented false positives
├── SOMPT22/             extracted source dataset (CC BY 4.0)
├── test_videos/         evaluation clips
├── false_preds/         known false-positive images — negative test set
├── runs/detect/         YOLO run outputs (legacy)
├── runs_rfdetr/         RF-DETR run outputs
└── *.py                 scripts, see below
```

**`datasets_coco` and `datasets_neg_coco` contain symlinks, not images.** Delete or move
`datasets_merged` / `datasets_neg` and both become ~31,000 dead pointers. Back up the
YOLO-format directories; the COCO ones regenerate in two minutes with `yolo2coco.py`.

---

## Dataset

| Split | Images | Boxes | Background |
|---|---|---|---|
| train | 27,767 (28,551 with negatives) | 32,665 | 784 |
| valid | 2,813 | 3,213 | 2 |
| test | 509 | 565 | 0 |

Class distribution: firearm 30,297 (92.7 %) · knife 1,739 (5.3 %) · hammer 631 (1.9 %).

Resolution: median 512×384, mean 712×613, range 80×89 to 5879×4279. **68.6 % of training
images are below 704 px on the long edge** and are upscaled to reach model input.

### Known dataset problems
- **The test split is not representative.** 39 % knife against a 5.3 % dataset share, zero
  hammers despite 631 present, zero background images, median 1280×960 against a training
  median of 512×384. **Do not quote test-split figures.**
- **Contamination.** Clipart, toy guns in retail packaging and storefront signage are
  labelled as weapons. Not quantified.


---

## Models

Full results in the validation report. Checkpoints in

| Version | Config | Status |
|---|---|---|
| `merged_v1` | YOLOv8m, 960px, 80ep | **AGPL-3.0 — cannot ship.** Baseline only |
| `rfdetr_m_v1` | Medium/576, 50ep | seed=null, not reproducible |
| `rfdetr_m_v2` | Medium/576, →60ep | overfitted, no regularisation |
| `rfdetr_m_v3` | Medium/576, 60ep | regularisation added |
| `rfdetr_l_v4` | Large/704, 60ep | **current recommendation** |
| `rfdetr_l_v5` | Large/704, →38ep | +784 negatives, regressed on validation |

**`checkpoint_best_total.pth` is usually the EMA copy**, not the raw weights — v4 is the
exception. Three files exist per run: `_best_regular`, `_best_ema`, `_best_total`.


### Large vs Medium

Same architecture — same DINOv2-S backbone, 4 decoder layers, 300 queries. 33.6 M vs
33.4 M params, and that difference is the positional encoding table. **The only real
difference is input resolution: 576 vs 704.** Describe v4/v5 as resolution experiments,
not capacity experiments.

Because of this a Large checkpoint loads into `RFDETRMedium` without error — it just runs
at the wrong resolution. Silent failure mode.

---

## Scripts

| Script | Purpose |
|---|---|
| `yolo2coco.py` | YOLO txt → COCO JSON. Required before any RF-DETR training |
| `train_rfdetr_v*.py` | Training configs, one per version |
| `video_eval_rfdetr.py` | Video recall + false alarms against `ranges.csv` |
| `render_rfdetr.py` | Burn detections into video for viewing |
| `predict_images.py` | Run over a folder of images, per-image table |
| `confusion_rfdetr.py` | Detection confusion matrix (Ultralytics convention) |
| `plot_rfdetr.py` | Training curves from `metrics.csv` |
| `audit_annotations.py` | Structural audit of bounding boxes |
| `image_resolutions.py` | Resolution statistics per split and source |

RF-DETR generates **no plots of its own** — no `results.png`, no confusion matrix, no
training mosaics. Everything Ultralytics provided by default had to be written.

---

## Common workflows

### Convert a dataset for training

```bash
python yolo2coco.py --src datasets_neg --dst datasets_neg_coco
# add --copy for real files instead of symlinks
```

Confirm the output line reports the expected image, box and background counts.

### Train

```bash
source rfdetr_env/bin/activate
nvidia-smi && free -g && df -h .
nohup python train_rfdetr_v5.py > train_v5.log 2>&1 &
disown
```

`disown` is required. See Pitfalls.

### Evaluate

```bash
# video recall — clips must be listed in ranges.csv
python video_eval_rfdetr.py --checkpoint <ckpt> --videos-dir test_videos \
    --ranges ranges.csv --classes 2 --model large --out eval.csv

# confusion matrix
python confusion_rfdetr.py --checkpoint <ckpt> --dataset datasets_coco/valid --model large

# false positives on the negative test set
python predict_images.py --checkpoint <ckpt> --images false_preds \
    --conf 0.5 --classes 2 --model large --outdir screen | tee screen.txt

# training curves
python plot_rfdetr.py runs_rfdetr/<version>/metrics.csv
```

### Reading the confusion matrix

`confusion_rfdetr.py` follows the Ultralytics convention: **predicted class on the vertical
axis, true class on the horizontal, column-normalised.** Read down columns, not across rows.

The validation report presents the transpose (true class per row) for readability. Both
describe the same matrix; check which you are looking at before quoting a figure.

The `background` **column** is false positives. The `background` **row** is missed
detections.

---

## Pitfalls

Everything here has cost real time at least once.

**`--model` defaults to `medium`.** Running a Large checkpoint without `--model large`
loads fine and runs at 576 instead of 704. No error.

**Class IDs differ between formats.** YOLO is 0-indexed (`0=knife 1=firearm 2=hammer`).
COCO is 1-indexed with a placeholder at 0 (`1=knife 2=firearm 3=hammer`). Firearm is
`--classes 1` for YOLO scripts and `--classes 2` for RF-DETR. Getting it wrong produces a
plausible-looking wrong number.

**Ctrl-C on `tail -f` kills the training job.** `nohup` protects against SIGHUP, not SIGINT,
and both are in the same process group. Use `disown`, or tail from a separate terminal.

**Check `ps` before relaunching anything.** A run that looks dead often isn't. Three
simultaneous training jobs once filled 62 GB of RAM because each launch appeared to fail.

**Lightning wipes `output_dir` on start.** Never point two runs at the same directory.

**RAM is `num_workers × prefetch_factor × batch_size`.** At 704, 16 × 4 × 32 does not fit in
62 GB. Use `num_workers=8, prefetch_factor=2`.

**Ultralytics settings are global and may point outside the project.** Check with
`yolo settings`; `runs_dir`, `weights_dir` and `datasets_dir` have all been wrong here.

**Symlinks broke under Ultralytics but are safe under COCO.** Ultralytics derives label
paths by swapping `images`→`labels` in the resolved path, so a symlink silently pointed it
at the wrong labels. COCO JSON names every file explicitly, so there is nothing to resolve
wrong.

**Pasting multi-line scripts into a terminal truncates.** Use `nano`, then `tail -3` to
confirm the file ends where it should.

**Validation mAP does not predict video performance.** They have disagreed four times. v2
and v3 differ by 0.001 on validation and by 0.31 on video recall. Measure on video.

**The eval harness has ±0.02 run-to-run variance.** Two identical runs on the same
checkpoint gave 0.6532 and 0.67. Differences under ~0.05 are not distinguishable.

**Validation contains no background images**, so its false-positive count measures spurious
boxes on images that contain a weapon elsewhere — not detections on weapon-free scenes. Use
`false_preds` and `sompt_neg` for that.


Results on Video Inference: https://drive.google.com/drive/folders/1WYCrUeq95WJ6cNs8-RWvVclLzFOcwToM?usp=drive_link
---
Results on Image Inference:



<img width="300" height="300" alt="Scene1_2" src="https://github.com/user-attachments/assets/ecd1068d-f130-4c61-85f1-2933c4450dd7" />



<img width="300" height="300" alt="Scene2_10" src="https://github.com/user-attachments/assets/3eab1620-6e3e-433b-94bd-82db6638b685" />



<img width="300" height="300" alt="Scene4_4" src="https://github.com/user-attachments/assets/f45557ce-64e9-4595-9ee3-044acdb69f18" />



<img width="300" height="300" alt="Scene6_10" src="https://github.com/user-attachments/assets/cd8cd735-2d6f-4b07-9dd0-cf82e567db00" />



<img width="300" height="300" alt="Scene2_17" src="https://github.com/user-attachments/assets/2358ce3f-3a6d-4df3-8d5c-bfe65cbebd18" />

---


## Licensing

| Component | Licence | Status |
|---|---|---|
| RF-DETR (Nano–Large) | Apache 2.0 | Clear |
| RF-DETR XL / 2XL | PML 1.0 | **Avoid** — DINOv3 backbone |
| DINOv2 (code + weights) | Apache 2.0 | Clear |
| SOMPT22 | CC BY 4.0 | Clear, attribution required |
| Ultralytics YOLOv8 | AGPL-3.0 | **Retired** — extends to trained weights |
| Training dataset | Unknown | **Open risk** |

SOMPT22 attribution: Simsek, Cigla & Kayabol, arXiv 2208.02580. The underlying footage was
recorded from public webcam streams the authors did not film.

**Open licensing risks:** dataset provenance across six-plus merged sources, and deployment
terms pending legal review.

---



## Current state

Use **`runs_rfdetr/rfdetr_m_v4/checkpoint_best_total.pth`** with `--model large`.

Firearm detection is strong. The outstanding problem is false alarms: 32.5 % of ordinary
CCTV frames produce a detection at confidence 0.5, and two high-confidence failures are
documented on weapon-free footage at 0.83 and 0.95. Threshold tuning cannot address these;
more and better negatives are the required intervention.

Knife is a data-volume constraint, not a modelling one — it has not moved across two
architectures, two resolutions, two epoch budgets and two regularisation regimes. Inter-class
confusion is effectively absent; the model either finds an object or it does not.

See the validation report for the full analysis.
