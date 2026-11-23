"""Data validation, metrics, and checkpoint I/O shared by the public entry points."""
import csv
import json
from pathlib import Path

import numpy as np
import torch

FIELDS = ('scene_semantics', 'structure_geometry', 'local_texture_artifacts', 'imaging_lighting')


def read_rows(filename, image_root=None, progan_only=False, limit=-1):
    rows = []
    for line in Path(filename).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row['image_path'])
        if not path.is_absolute():
            if image_root is None:
                raise ValueError('Relative image paths require --image-root')
            root = Path(image_root).resolve()
            path = (root / path).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Image path escapes --image-root')
        if not path.is_file():
            raise FileNotFoundError(path)
        if progan_only and not ('progan' in {p.lower() for p in path.parts} or row.get('generator', '').lower() == 'progan'):
            raise ValueError(f'Only GAN-Test/ProGAN evaluation is supported: {path.name}')
        if row.get('label') not in ('real', 'fake'):
            raise ValueError(f'Invalid label: {row.get("label")}')
        if not row.get('parse_ok', True) or not all(str(row.get(k, '')).strip() for k in FIELDS):
            raise ValueError(f'Missing forensic evidence: {path.name}')
        row['image_path'] = str(path)
        rows.append(row)
    if limit > 0:
        # A deterministic balanced subset for smoke tests, never relabel samples.
        half = limit // 2
        rows = [r for r in rows if r['label'] == 'real'][:half] + [r for r in rows if r['label'] == 'fake'][:limit-half]
    if not rows or len({r['image_path'] for r in rows}) != len(rows):
        raise ValueError('Empty manifest or duplicate image paths')
    if {r['label'] for r in rows} != {'real', 'fake'}:
        raise ValueError('Both classes are required')
    return rows


def metrics(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite prediction scores')
    pred = scores > 0
    real = labels == 0
    fake = labels == 1
    if not real.any() or not fake.any():
        raise ValueError('Both classes are required for metrics')
    ra = float((~pred[real]).mean() * 100)
    fa = float(pred[fake].mean() * 100)
    order = np.argsort(-scores, kind='stable')
    ordered = labels[order]
    # Group equal scores before integrating precision over recall increments.
    ends = np.r_[np.where(np.diff(scores[order]) != 0)[0], len(scores)-1]
    tp = np.cumsum(ordered)[ends]
    ap = np.sum(np.diff(np.r_[0, tp]) * tp / (ends+1)) / fake.sum()
    return dict(n=len(labels), acc=float((pred == labels).mean()*100),
                balanced_acc=(ra+fa)/2, ap=float(ap*100),
                real_acc=ra, fake_acc=fa)


def append_json(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def save_checkpoint(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def write_predictions(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=('image_path', 'label', 'fake_margin'))
        writer.writeheader()
        writer.writerows(rows)
