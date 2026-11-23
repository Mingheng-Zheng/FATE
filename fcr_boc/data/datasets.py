"""Evidence/image collation; split validation lives in release_utils.read_rows."""
import random
import torch
from fcr_boc.common.torch_utils import safe_open_rgb, label_to_id
from fcr_boc.data.constants import FIELDS

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
