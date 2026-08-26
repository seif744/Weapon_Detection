"""
visualize.py
Production-grade Feature Map PCA Visualizer for RF-DETR.
Ensures strict RGB pipeline and centered, sign-pinned PCA projections.
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRBase, RFDETRLarge
except ImportError:
    raise ImportError("Could not import 'rfdetr'. Run this from your RF-DETR environment.")

SIZES = {
    "nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium,
    "base": RFDETRBase, "large": RFDETRLarge
}

# Fallback names if wrapper.names is unavailable
DEFAULT_NAMES = {0: "placeholder", 1: "knife", 2: "firearm", 3: "hammer"}
COLORS = {1: (80, 200, 255), 2: (60, 60, 255), 3: (120, 255, 120)}


def load_model(checkpoint: str, size: str):
    """Native instantiation to preserve EMA state and head dimensions."""
    print(f"Loading {checkpoint} via native wrapper...", file=sys.stderr)
    return SIZES[size](pretrain_weights=checkpoint)


def get_core_module(wrapper) -> torch.nn.Module:
    """Safely extracts the core PyTorch module without fragile dir() scraping."""
    for path in (("model", "model"), ("model",)):
        obj = wrapper
        try:
            for attr in path:
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.Module):
            return obj
    raise RuntimeError("Could not resolve core nn.Module from wrapper.")


class FeatureExtractor:
    """Safely walks DETR nested outputs and tuples to extract spatial feature maps."""
    def __init__(self, module: torch.nn.Module):
        self.candidates: list[torch.Tensor] = []
        self._handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.candidates = [] # Clear state to prevent OOM loops
        self._walk(outputs)

    def _walk(self, obj, depth=0):
        if depth > 6 or obj is None:
            return
        if isinstance(obj, torch.Tensor):
            if obj.dim() in (3, 4):
                self.candidates.append(obj.detach().float())
            return
        if hasattr(obj, "tensors"):
            return self._walk(obj.tensors, depth + 1)
        if hasattr(obj, "decompose"):
            return self._walk(obj.decompose()[0], depth + 1)
        if isinstance(obj, dict):
            for v in obj.values():
                self._walk(v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for v in obj:
                self._walk(v, depth + 1)
            return

    def remove(self):
        self._handle.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.remove()


def resolve_to_spatial(feat: torch.Tensor) -> torch.Tensor:
    """Resolves arbitrary ViT sequences or CNN maps to [1, C, H, W]."""
    if feat.dim() == 4:
        return feat

    B, A, Bd = feat.shape
    for seq_dim, chan_dim in ((1, 2), (2, 1)):
        L = feat.shape[seq_dim]
        for extra in (0, 1, 4, 5):
            if L <= extra: continue
            H = int(round(math.sqrt(L - extra)))
            if H * H != (L - extra): continue
            
            t = feat if seq_dim == 2 else feat.permute(0, 2, 1)
            t = t[..., extra:].contiguous()
            return t.view(t.size(0), t.size(1), H, H)

    raise ValueError(f"Could not resolve shape {tuple(feat.shape)} to [1, C, H, W].")


def compute_pca_rgb(feat: torch.Tensor) -> np.ndarray:
    """Computes mathematically rigorous, mean-centered PCA."""
    feat = resolve_to_spatial(feat)
    if feat.size(0) != 1:
        feat = feat[:1]

    _, C, H, W = feat.shape
    if C < 3:
        raise ValueError(f"Layer has {C} channels; requires >= 3 for RGB PCA.")

    flat = feat.squeeze(0).reshape(C, -1).t().contiguous() 

    # Strict mean centering
    mean = flat.mean(dim=0, keepdim=True)
    centred = flat - mean
    
    # SVD on centered data
    _, _, V = torch.pca_lowrank(centred, q=3, center=False)
    proj = centred @ V 

    # Deterministic sign pinning
    idx = proj.abs().argmax(dim=0)
    signs = torch.sign(proj[idx, torch.arange(3, device=proj.device)])
    signs[signs == 0] = 1.0
    proj = proj * signs

    # Normalize to uint8
    lo = proj.min(dim=0, keepdim=True).values
    hi = proj.max(dim=0, keepdim=True).values
    norm = (proj - lo) / torch.clamp(hi - lo, min=1e-5)
    return (norm * 255.0).to(torch.uint8).view(H, W, 3).cpu().numpy()


def annotate_frame(bgr_image: np.ndarray, det, names_dict: dict) -> np.ndarray:
    """Renders bounding boxes without relying on fragile third-party parsers."""
    out = bgr_image.copy()
    if det is None or len(det) == 0:
        return out
        
    for box, conf, k in zip(np.asarray(det.xyxy), np.asarray(det.confidence), np.asarray(det.class_id).astype(int)):
        x1, y1, x2, y2 = [int(v) for v in box]
        col = COLORS.get(k, (255, 255, 255))
        
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        label = f"{names_dict.get(k, str(k))} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="pca.jpg")
    parser.add_argument("--size", default="medium", choices=list(SIZES.keys()))
    parser.add_argument("--layer", default="backbone.0.projector.stages.0.0.cv2.conv")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--scale-index", type=int, default=0)
    args = parser.parse_args()

    if not Path(args.image).exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    # 1. Model Init
    wrapper = load_model(args.model, args.size)
    core = get_core_module(wrapper)
    names_dict = getattr(wrapper, 'names', DEFAULT_NAMES)

    target = dict(core.named_modules()).get(args.layer)
    if target is None:
        raise ValueError(f"Layer '{args.layer}' not found in the graph.")

    # 2. Strict Input Pipeline
    bgr_frame = cv2.imread(args.image)
    if bgr_frame is None:
        raise ValueError(f"Failed to decode image at {args.image}")
        
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB) # RGB Conversion Forced

    # 3. Hook & Infer
    with FeatureExtractor(target) as fx:
        det = wrapper.predict(rgb_frame, threshold=args.conf)
        
        if not fx.candidates:
            raise RuntimeError(f"Hook fired no usable tensor for layer '{args.layer}'.")
            
        feat = fx.candidates[min(args.scale_index, len(fx.candidates) - 1)]

    # 4. Process Outputs
    pca_heatmap = compute_pca_rgb(feat)
    pca_heatmap = cv2.resize(pca_heatmap, (bgr_frame.shape[1], bgr_frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    pca_heatmap = cv2.cvtColor(pca_heatmap, cv2.COLOR_RGB2BGR) # Convert back to BGR for concatenation

    annotated = annotate_frame(bgr_frame, det, names_dict)
    
    # 5. Save Side-by-Side
    combined_img = np.hstack((annotated, pca_heatmap))
    cv2.imwrite(args.output, combined_img)
    print(f"Visualization saved to {args.output}")


if __name__ == "__main__":
    main()