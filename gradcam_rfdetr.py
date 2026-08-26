import sys, numpy as np, torch
from PIL import Image
from pytorch_grad_cam import GradCAM, HiResCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from rfdetr import RFDETRLarge
from rfdetr.utilities.tensors import NestedTensor

CKPT  = "/home/easemyai/weapons_project/runs_rfdetr/rfdetr_l_v4/checkpoint_best_total.pth"
RES   = 704
CLS   = 2                                  # class index to explain
IMG   = sys.argv[1]
LAYER = sys.argv[2]
METH  = sys.argv[3] if len(sys.argv) > 3 else "gradcam"
OUT   = sys.argv[4] if len(sys.argv) > 4 else "cam.png"

wrap = RFDETRLarge(pretrain_weights=CKPT, resolution=RES, num_classes=4)
net  = wrap.model.model.eval().float().cuda()

named = dict(net.named_modules())
if LAYER not in named:
    sys.exit("no such layer. candidates:\n" + "\n".join(k for k in named if "proj" in k))
target_layers = [named[LAYER]]


class LogitsOnly(torch.nn.Module):
    """RF-DETR returns a dict; pytorch-grad-cam zips over the output, which
    iterates dict keys and hands the target a string. Return a bare tensor."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        mask = torch.zeros(x.shape[0], x.shape[2], x.shape[3], dtype=torch.bool, device=x.device)
        o = self.net(NestedTensor(x, mask))
        return o["pred_logits"] if isinstance(o, dict) else o[0]      # [B, Q, C]


class QueryLogit:
    """Explain the single query with the highest score for class CLS."""
    def __init__(self):
        self.qi = None

    def __call__(self, logits):                                        # [Q, C]
        if self.qi is None:
            self.qi = int(logits[:, CLS].argmax())
            print(f"explaining query {self.qi}  "
                  f"sigmoid={torch.sigmoid(logits[self.qi, CLS]):.3f}")
        return logits[self.qi, CLS]


# preprocess: square resize + imagenet norm
pil  = Image.open(IMG).convert("RGB")
rgb  = np.array(pil.resize((RES, RES), Image.BILINEAR), np.float32) / 255.0
mean = np.array([0.485, 0.456, 0.406], np.float32)
std  = np.array([0.229, 0.224, 0.225], np.float32)
x    = torch.from_numpy(((rgb - mean) / std).transpose(2, 0, 1))[None].cuda()
x.requires_grad_(True)

model = LogitsOnly(net)

with torch.no_grad():
    lg = model(x)[0]
    print(f"raw max sigmoid for class {CLS} = "
          f"{float(torch.sigmoid(lg[:, CLS]).max()):.4f}")

Method = {"gradcam": GradCAM, "hirescam": HiResCAM, "eigencam": EigenCAM}[METH]
targets = None if METH == "eigencam" else [QueryLogit()]

with Method(model=model, target_layers=target_layers) as cam:
    g = cam(input_tensor=x, targets=targets)[0]

print(f"cam raw range {g.min():.4f} .. {g.max():.4f}")
g = (g - g.min()) / (g.max() - g.min() + 1e-9)
vis = show_cam_on_image(rgb, g, use_rgb=True)
Image.fromarray(vis).resize(pil.size, Image.BILINEAR).save(OUT)
print("wrote", OUT)
