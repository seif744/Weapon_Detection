#!/usr/bin/env python3
"""
train_rfdetr_v3.py -- RF-DETR Medium, regularized.

WHY THIS RUN EXISTS
-------------------
v1 (50 ep): best EMA mAP50:95 0.5627 @ ep41 | video recall 0.89 @ conf 0.25
v2 (150 ep, stopped at 60): best 0.5595 @ ep35 | video recall 0.65 @ conf 0.25

More epochs did not help. v2's loss curves showed classic overfitting: train
loss fell 6.6 -> 3.4 and was still falling at ep60, while val loss flatlined
at ~5.5 from ep32. The v2 training_config.json then showed why:

    drop_path       0.0        <- NO stochastic depth on a ViT
    weight_decay    1e-4       <- low
    warmup_epochs   0.0        <- none
    lr_drop         100        <- run stopped at 60, so LR NEVER decayed
    seed            null       <- runs not reproducible

This run turns on regularization and actually exercises the LR schedule.

WHAT TO EXPECT -- read this before you interpret the result
-----------------------------------------------------------
The video-eval harness has roughly +/-0.02 run-to-run variance (two identical
runs on the v2 checkpoint gave 0.6532 and 0.67). Anything under ~0.05 of
movement is NOT distinguishable from noise.

This run targets generalization. It will NOT fix:
  - knife, pinned at ~0.30 across two architectures and two epoch budgets.
    1,739 instances vs 30,297 firearms. That is a data volume problem.
  - the high-confidence false positive on a puffer jacket (0.83). The dataset
    contains ZERO images labeled "person, no weapon". Regularization cannot
    teach a rule from examples that do not exist.

If this lands at ~0.55 val like the last two, the ceiling is the dataset, and
the next work is contamination removal + hard negatives, not hyperparameters.

Hardware: RTX 6000 Ada, 48 GB. Data: 27,767 train / 2,813 val / 509 test.
Expect ~9 min/epoch, so ~9 hours for 60 epochs, less if early stopping fires.

Run:
    source rfdetr_env/bin/activate
    nohup python train_rfdetr_v3.py > train_v3.log 2>&1 &
    tail -f train_v3.log
"""

from rfdetr import RFDETRMedium

# resolution must be divisible by patch_size * num_windows.
# For Medium that is 16 * 2 = 32.  576 / 32 = 18, so 576 is valid.
# (An earlier note in this project said "divisible by 56" -- that was wrong.)
# Held at 576 deliberately: the test clips are 596x336, so a higher input
# resolution would only interpolate harder, not recover detail.
model = RFDETRMedium(resolution=576)

model.train(
    # ---------------------------------------------------------------
    # PATHS
    # ---------------------------------------------------------------
    dataset_dir='datasets_coco',
    output_dir='runs_rfdetr/rfdetr_m_v3',

    # ---------------------------------------------------------------
    # CHANGED FROM v2 -- all five target the overfitting in the curves
    # ---------------------------------------------------------------

    # (1) STOCHASTIC DEPTH.  v2 had this at 0.0, i.e. no regularization at
    # all on a vision transformer.  drop_path randomly skips residual
    # blocks during training, forcing the network not to rely on any single
    # path.  It is the standard ViT regularizer and it is the most direct
    # answer to "train loss falls, val loss doesn't".
    # 0.15 is a moderate value; 0.1-0.3 is the usual range.
    drop_path=0.15,

    # (2) WEIGHT DECAY 1e-4 -> 1e-3.  Penalizes large weights. Same goal as
    # above by a different mechanism.  10x sounds aggressive but 1e-4 is on
    # the low side for a fine-tune on 27k images.
    weight_decay=1e-3,

    # (3) LR DROP 100 -> 25.  This is the untested lever.  v1 ran 50 epochs
    # and v2 stopped at 60, so with lr_drop=100 the learning rate sat at a
    # flat 1e-4 for the entire life of BOTH models.  Neither ever entered
    # the low-LR phase where a model settles into a minimum instead of
    # orbiting it.  At 25 it fires with 35 epochs of refinement left.
    lr_drop=25,

    # (4) WARMUP.  v2 had none -- full LR from step one.  A short warmup
    # stabilizes the first epochs, which matters more now that the head is
    # randomly initialized (90 COCO classes -> 4).
    warmup_epochs=1.0,

    # (5) SEED.  v2 had seed=null, so neither previous run is reproducible.
    # Given the harness noise, being able to repeat a run exactly is worth
    # more than it sounds.
    seed=42,

    # ---------------------------------------------------------------
    # EPOCHS -- reduced, not increased
    # ---------------------------------------------------------------
    # v2 was given 150 and stopped itself at 60. Asking for more than that
    # again would just repeat a settled experiment. 60 with lr_drop=25 means
    # 25 epochs at full LR + 35 at decayed LR.
    epochs=60,
    early_stopping=True,
    early_stopping_patience=8,     # 8 evals x eval_interval 3 = 24 epochs

    # ---------------------------------------------------------------
    # UNCHANGED FROM v2 -- so the comparison stays interpretable
    # ---------------------------------------------------------------
    batch_size=32,
    grad_accum_steps=1,            # effective batch 16
    eval_interval=3,
    num_workers=16,                # v2 ran at 95% GPU util, so this is enough
    persistent_workers=True,
    prefetch_factor=4,
    pin_memory=True,
    compute_train_metrics=False,
    checkpoint_interval=10,
    log_per_class_metrics=True,
    run_test=True,
    tensorboard=False,
    wandb=False,
)

# ---------------------------------------------------------------------
# DELIBERATELY NOT CHANGED, AND WHY
# ---------------------------------------------------------------------
# do_random_resize_via_padding=True
#   Would preserve aspect ratio instead of squashing 596x336 into a 576
#   square (a 1.7x vertical stretch). Genuinely untested and plausibly
#   significant for CCTV. Left out so this run has ONE theme -- change it
#   alone in v4 and you will know what it did.
#
# lr_scheduler='cosine'
#   Smooth decay instead of a single step. Often slightly better. Left out
#   for the same reason: lr_drop=25 already tests "does decay help at all",
#   which is the question worth answering first.
#
# RFDETRLarge
#   The curves show a model that plateaus, not one starved for capacity.
#   More parameters would fit the training distribution harder -- and that
#   distribution contains clipart, catalog photos, and toy guns.
#
# resolution=728 or 896
#   Only pays off when the source frame is LARGER than the model input.
#   The test clips are 336 pixels tall. This would be upscaling noise.
#   Revisit if the client's cameras turn out to be 1080p.
#
# ---------------------------------------------------------------------
# WHEN IT FINISHES
# ---------------------------------------------------------------------
#   python plot_rfdetr.py runs_rfdetr/rfdetr_m_v3/metrics.csv
#
#   python video_eval_rfdetr.py \
#       --checkpoint runs_rfdetr/rfdetr_m_v3/checkpoint_best_total.pth \
#       --videos-dir datasets/test_videos --ranges ranges.csv \
#       --classes 2 --out eval_v3.csv
#
# Compare against v1 = 0.89 and v2 = 0.65 at conf 0.25.
# In the loss panel, look for the train and val curves staying closer
# together than they did in v2. That is what these changes are for.
#
# Then prune the run directory -- checkpoint_*.ckpt and last.ckpt are
# ~535 MB each and only useful for resuming:
#   rm runs_rfdetr/rfdetr_m_v3/checkpoint_*.ckpt runs_rfdetr/rfdetr_m_v3/last.ckpt
