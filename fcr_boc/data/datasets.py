# -*- coding: utf-8 -*-

import json
import os
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import Dataset

from fcr_boc.common.torch_utils import safe_open_rgb, label_to_id
from fcr_boc.data.constants import IMAGE_EXTS, FIELDS, DEFAULT_DESC

# ---------------- data ----------------
def valid_desc(o: Dict) -> bool:
    return bool(o.get("parse_ok", True)) and all(str(o.get(k, "")).strip() for k in FIELDS)


def read_jsonl_train(path: str, max_samples=-1, seed=42, balance=True):
    real, fake = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not valid_desc(o):
                # fallback rather than discard, so training learns no-text cases too.
                for k, v in DEFAULT_DESC.items():
                    o[k] = v
                o["parse_ok"] = True
                o["desc_source"] = "default_fallback_train"
            p = o.get("image_path") or o.get("path") or o.get("image")
            if not p or not os.path.exists(p):
                continue
            o["image_path"] = p
            y = label_to_id(o.get("label_id", o.get("label", 0)))
            o["label"] = "real" if y == 0 else "fake"
            if y == 0: real.append(o)
            else: fake.append(o)
    rng = random.Random(seed); rng.shuffle(real); rng.shuffle(fake)
    if balance:
        n = min(len(real), len(fake))
        real, fake = real[:n], fake[:n]
    samples = real + fake
    rng.shuffle(samples)
    if max_samples and max_samples > 0:
        samples = samples[:max_samples]
    print(f"[TrainJsonl] real={len(real)} fake={len(fake)} total={len(samples)} from={path}")
    return samples


class ForensicDescDataset(Dataset):
    def __init__(self, jsonl, max_samples=-1, seed=42):
        self.samples = read_jsonl_train(jsonl, max_samples, seed, True)
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def load_desc_map(jsonl_path: str):
    desc = {}
    if not jsonl_path:
        return desc
    p = Path(jsonl_path)
    if not p.exists():
        print(f"[EvalDesc] missing: {jsonl_path}")
        return desc
    cnt = 0; fallback = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try: o = json.loads(line)
            except Exception: continue
            if not valid_desc(o):
                for k, v in DEFAULT_DESC.items(): o[k] = v
                o["parse_ok"] = True
                o["desc_source"] = "default_fallback_loader"
                fallback += 1
            ip = o.get("image_path") or o.get("path") or o.get("image")
            if not ip: continue
            d = {k: str(o.get(k, DEFAULT_DESC[k])).strip() or DEFAULT_DESC[k] for k in FIELDS}
            d["desc_source"] = str(o.get("desc_source", "qwen_generated"))
            keys = [str(Path(ip)), Path(ip).name]
            try: keys.append(str(Path(ip).resolve()))
            except Exception: pass
            if o.get("dataset_name") and o.get("relative_path"):
                keys.append(f"{o.get('dataset_name')}::{o.get('relative_path')}")
            for k in keys: desc[k] = d
            cnt += 1
    print(f"[EvalDesc] loaded records={cnt}, keys={len(desc)}, fallback_seen={fallback} from {jsonl_path}")
    return desc


class EvalImageDataset(Dataset):
    def __init__(self, root, max_samples=-1, seed=42, desc_map=None):
        self.samples = []
        self.desc_map = desc_map or {}
        rp = Path(root)
        if not rp.exists():
            print(f"[EvalDataset] missing root: {root}")
            return
        real = self._list(rp, "0_real"); fake = self._list(rp, "1_fake")
        rt, ft = len(real), len(fake)
        rng = random.Random(seed); rng.shuffle(real); rng.shuffle(fake)
        if max_samples and max_samples > 0:
            half = max_samples // 2
            real = real[:min(rt, half)]; fake = fake[:min(ft, max_samples - len(real))]
        self.samples = [{"image_path": str(p), "label": "real", "dataset_name": rp.name} for p in real] + \
                       [{"image_path": str(p), "label": "fake", "dataset_name": rp.name} for p in fake]
        hit = fallback = 0
        for s in self.samples:
            ip = s["image_path"]
            rel = ""
            try: rel = str(Path(ip).relative_to(rp))
            except Exception: rel = Path(ip).name
            keys = [ip, str(Path(ip).resolve()), Path(ip).name, f"{rp.name}::{rel}"]
            d = None
            for k in keys:
                if k in self.desc_map:
                    d = self.desc_map[k]; break
            if d:
                for kk in FIELDS: s[kk] = d.get(kk, DEFAULT_DESC[kk])
                s["desc_source"] = d.get("desc_source", "qwen_generated")
                hit += 1
                if "fallback" in s["desc_source"]: fallback += 1
            else:
                for kk, vv in DEFAULT_DESC.items(): s[kk] = vv
                s["desc_source"] = "default_fallback_missing_in_eval_dataset"
                fallback += 1
        rng.shuffle(self.samples)
        print(f"[EvalDataset] {root}: real={len(real)}/{rt} fake={len(fake)}/{ft} total={len(self.samples)} desc_hit={hit}/{len(self.samples)} fallback={fallback}")
    @staticmethod
    def _list(root: Path, label: str):
        files = []
        for cur, _, names in os.walk(root, followlinks=True):
            if os.path.basename(cur) != label:
                continue
            for name in names:
                p = Path(cur) / name
                if p.suffix.lower() in IMAGE_EXTS:
                    files.append(p)
        return sorted(files, key=lambda x: str(x))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


class Collator:
    def __init__(self, processor, freq_transform, train_mode=True, text_dropout_prob=0.0):
        self.processor = processor
        self.freq_transform = freq_transform
        self.train_mode = train_mode
        self.text_dropout_prob = float(text_dropout_prob)
    def make_prompt(self, s):
        has_desc = all(str(s.get(k, "")).strip() for k in FIELDS)
        if self.train_mode and self.text_dropout_prob > 0 and random.random() < self.text_dropout_prob:
            desc = "未提供有效自生成法证描述。\n"
        elif has_desc:
            desc = (
                f"场景语义：{s['scene_semantics']}\n"
                f"结构几何：{s['structure_geometry']}\n"
                f"局部纹理：{s['local_texture_artifacts']}\n"
                f"成像光照：{s['imaging_lighting']}\n"
            )
        else:
            desc = "未提供有效自生成法证描述。\n"
        return (
            "你是一名图像真实性法证检测模型。以下文字由同一多模态大模型根据当前图像自动生成，不包含真假标签。"
            "不要依赖物体类别语义，重点关注局部纹理、高频细节、结构几何和成像一致性。\n"
            + desc +
            "请结合输入图像和自生成四维法证描述进行真实性二分类。最终只需内部输出分类特征，不需要生成文本。"
        )
    def __call__(self, batch):
        texts, imgs, freq, labels, paths = [], [], [], [], []
        for s in batch:
            p = s["image_path"]
            img = safe_open_rgb(p)
            imgs.append(img)
            freq.append(self.freq_transform(img))
            labels.append(label_to_id(s["label"]))
            paths.append(p)
            msg = [{"role": "user", "content": [{"type": "image", "image": p}, {"type": "text", "text": self.make_prompt(s)}]}]
            texts.append(self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        qi = self.processor(text=texts, images=imgs, padding=True, return_tensors="pt")
        return {"qwen_inputs": qi, "freq_x": torch.stack(freq), "labels": torch.tensor(labels, dtype=torch.long), "paths": paths}

