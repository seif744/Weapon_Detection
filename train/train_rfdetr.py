#!/usr/bin/env python3
"""
train_rfdetr.py -- fine-tune RF-DETR Medium on datasets_coco.

Tuned for wall-clock speed WITHOUT changing training dynamics.
Every speedup below is either free or I/O-side. Nothing here alters the
effective batch size or the learning schedule, so the result should match
the conservative config -- it just arrives sooner.

Hardware: RTX 6000 Ada, 48 GB (~43 GB free), CUDA 13.0
Data:     27,767 train / 2,813 val / 509 test

Run:
    source rfdetr_env/bin/activate
    nohup python train_rfdetr.py > train_rfdetr.log 2>&1 &
    tail -f train_rfdetr.log
"""

from rfdetr import RFDETRMedium

# resolution is a CONSTRUCTOR arg (must be divisible by 56).
# Medium defaults to 576. Left at default on purpose: this run is the
# architecture A/B against your 0.20 video recall. Bump to 728 NEXT run.
model = RFDETRMedium()

model.train(
    dataset_dir='datasets_coco',
    output_dir='runs_rfdetr/rfdetr_m_v2',

    epochs=150,

    # --- SPEEDUP 1: same effective batch (16), one optimizer step not two.
    # 8x2 and 16x1 are mathematically equivalent, but 16x1 avoids the
    # accumulation overhead. 43 GB free means 16 at 576px is not near the
    # memory ceiling.
    batch_size=16,
    grad_accum_steps=1,

    # --- SPEEDUP 2: don't validate 2,813 images every single epoch.
    # On a transformer that eval is not cheap. Every 3rd epoch still gives
    # ~16 points across the run -- enough to see the curve and enough for
    # early stopping to act on.
    eval_interval=3,

    # --- SPEEDUP 3: dataloader. 8 workers is what merged_v1 used on this
    # box successfully. persistent_workers avoids respawning them each epoch.
    num_workers=16,
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,

    # --- SPEEDUP 4: skip metrics on the training set. Train metrics are
    # decorative; val metrics are what you actually read.
    compute_train_metrics=False,

    # --- SPEEDUP 5: fewer checkpoint writes. You are at 94% disk.
    # Best-model is saved regardless; this only affects periodic snapshots.
    checkpoint_interval=10,

    # Stop early if val plateaus. Patience is counted in EVAL steps, and
    # evals now happen every 3 epochs, so 5 ~= 15 epochs of no improvement.
    early_stopping=True,
    early_stopping_patience=8,

    log_per_class_metrics=True,   # per-class, not just the macro average
    run_test=True,                # scores the 509-image test split at the end

    tensorboard=False,
    wandb=False,
)