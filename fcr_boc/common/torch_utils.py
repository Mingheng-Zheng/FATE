# -*- coding: utf-8 -*-

import random
import numpy as np
from PIL import Image
import torch

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def safe_open_rgb(path: str):
    return Image.open(path).convert("RGB")


def label_to_id(label):
    if isinstance(label, int): return int(label)
    s = str(label).lower()
    if s in {"0", "real", "0_real"}: return 0
    if s in {"1", "fake", "1_fake"}: return 1
    return 0 if s == "real" else 1


def autocast_cuda(enabled=True, dtype=torch.bfloat16):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda", dtype=dtype, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def make_scaler(use_amp: bool, amp_dtype: str):
    enabled = bool(use_amp and amp_dtype == "fp16")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def amp_dtype_from_name(name: str):
    return torch.float16 if name == "fp16" else torch.bfloat16



def require_finite_scalar(name: str, value: torch.Tensor, step: int):
    """Fail fast before NaNs can poison all trainable parameters."""
    detached = value.detach()
    if not bool(torch.isfinite(detached).all().item()):
        raise FloatingPointError(
            f"[NonFinite] step={step} {name} is not finite: "
            f"{detached.float().cpu().item() if detached.numel() == 1 else 'tensor'}"
        )


def require_finite_grad_norm(total_norm: torch.Tensor, step: int):
    if not bool(torch.isfinite(total_norm.detach()).all().item()):
        raise FloatingPointError(
            f"[NonFinite] step={step} gradient norm is not finite: "
            f"{total_norm.detach().float().cpu().item()}"
        )
