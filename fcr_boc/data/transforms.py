# -*- coding: utf-8 -*-

from torchvision import transforms
from torchvision.transforms import InterpolationMode

class DeterministicFrequencyTransform:
    def __init__(self, size=256, view="center_crop"):
        self.size = int(size); self.view = view
        if view == "center_crop":
            self.tf = transforms.Compose([transforms.CenterCrop([self.size, self.size]), transforms.ToTensor()])
        elif view == "resize":
            self.tf = transforms.Compose([transforms.Resize([self.size, self.size], interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        else:
            raise ValueError(view)
    def __call__(self, img):
        return self.tf(img.convert("RGB"))

