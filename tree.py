from rfdetr import RFDETRLarge
m = RFDETRLarge(pretrain_weights="runs_rfdetr/rfdetr_l_v4/checkpoint_best_total.pth",
                resolution=704, num_classes=4)
inner = m.model.model
print(type(inner))
for n, mod in inner.named_modules():
    if any(k in n for k in ("projector", "neck", "input_proj", "backbone")) \
       and mod.__class__.__name__ in ("Conv2d", "GroupNorm", "BatchNorm2d", "Sequential"):
        print(f"{n:70s} {mod.__class__.__name__}")
