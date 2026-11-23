# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from fcr_boc.common.torch_utils import autocast_cuda

try:
    from pytorch_wavelets import DWTForward
except Exception as e:
    raise RuntimeError('Please install: uv pip install pytorch_wavelets PyWavelets') from e

# ---------------- HH SAFE encoder / teacher ----------------
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SafeTrunk(nn.Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def _make_layer(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            down = nn.Sequential(conv1x1(self.inplanes, planes * Bottleneck.expansion, stride), nn.BatchNorm2d(planes * Bottleneck.expansion))
        layers = [Bottleneck(self.inplanes, planes, stride, down)]
        self.inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.avgpool(x).flatten(1)


class HHSAFE(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3")
        self.trunk = SafeTrunk(3)
        self.fc = nn.Linear(512, 2)
    def hh_band(self, x):
        with autocast_cuda(enabled=False):
            x32 = x.float()
            _, yh = self.dwt(x32)
            hh = yh[0][:, :, 2]
            hh = F.interpolate(hh, size=x32.shape[-2:], mode="bilinear", align_corners=False)
        return hh
    def forward(self, x):
        z = self.trunk(self.hh_band(x))
        logits = self.fc(z)
        return {"logits": logits, "z": z}


def load_hh_checkpoint(model: HHSAFE, ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"HH checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    new_state = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        # HH-only checkpoint variants
        if nk.startswith("hh_trunk."):
            nk = "trunk." + nk[len("hh_trunk."):]
        elif nk.startswith("hh_expert."):
            nk = "trunk." + nk[len("hh_expert."):]
        elif nk.startswith("fc_hh."):
            nk = "fc." + nk[len("fc_hh."):]
        elif nk.startswith("fc1."):
            nk = "fc." + nk[len("fc1."):]
        elif nk == "fc1.weight":
            nk = "fc.weight"
        elif nk == "fc1.bias":
            nk = "fc.bias"
        # original SAFE style conv/bn/layer directly under root
        root_prefixes = ("conv1.", "bn1.", "layer1.", "layer2.")
        if nk.startswith(root_prefixes):
            nk = "trunk." + nk
        if nk.startswith("trunk.") or nk.startswith("fc."):
            new_state[nk] = v
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[HH] loaded ckpt={ckpt_path}")
    print(f"[HH] tensors={len(new_state)} missing={missing[:8]} unexpected={unexpected[:8]}")

