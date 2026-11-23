# FATE

Frequency-Anchored Textual Evidence Learning for AI-Generated Image Detection.

This repository contains the HH teacher, frequency-anchored and corrective LoRA modules,
bounded orthogonal correction (BOC), and two-stage training and evaluation entry points.

## Release contents

| Entry point | Purpose |
| --- | --- |
| `train_hh.py` | Stage 1: train the bior1.3 HH teacher from scratch or initialize an HH checkpoint |
| `train_fate.py` | Stage 2: freeze HH and Qwen, train frequency router, anchor/corrective LoRA and classifier |
| `evaluate.py` | Evaluate one checkpoint on GAN-Test/ProGAN |
| `tools/prepare_evidence.py` | Resolve FORTE annotation paths for a local image installation |
| `configs/fate.json` | Portable hyperparameters for the two-epoch 8B configuration |
| `fcr_boc/models/` | Released architecture; the internal package name is retained for compatibility |

**The public training and evaluation entry points use GAN-Test/ProGAN only for validation
and checkpoint selection.** They do not automatically scan other test datasets.
An evaluation manifest containing another generator is rejected. Validation uses balanced
accuracy, with AP as a tie-breaker. Training with this validation protocol can select a
different checkpoint from other experimental protocols. No cross-dataset performance is
claimed by the provided ProGAN-only evaluation command.

## Installation

```bash
conda create -n fate python=3.11 -y
conda activate fate
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build compatible with your GPU driver. The server smoke
checks use Python 3.11, PyTorch 2.12.0, torchvision 0.27.0, transformers 5.10.2,
accelerate 1.13.0, pytorch-wavelets 1.3.0, numpy 2.4.6 and Pillow 12.2.0.
Stage 1 does not load Qwen. Stage 2 requires a CUDA GPU with sufficient memory for
Qwen3-VL-8B-Instruct; effective batch size 4 and microbatch size 2 are configured by default.

## Models and data

- Backbone: obtain `Qwen/Qwen3-VL-8B-Instruct` from its official model repository.
- Dataset and English forensic descriptions: `mh031228/FORTE` on Hugging Face.
- HH and FATE adapter checkpoints: `mh031228/FATE` on Hugging Face.
- Image sources remain subject to their original dataset terms. See the FORTE dataset
  card for the precise files available in the release and source download links.
- Model weights and training images are not stored in this Git repository.

FORTE stores `image_path`, `label` (`real`/`fake`), `generator` and four evidence fields:
`scene_semantics`, `structure_geometry`, `local_texture_artifacts`, `imaging_lighting`.
Relative paths are rooted in your local image installation. All paths must resolve before
training. Original image pixels are not resized or recompressed by the data packaging code.

Example directory layout:

```text
data/images/
  train/progan/{car,cat,chair,horse}/{0_real,1_fake}/...
  test/GAN-Test/progan/{category}/{0_real,1_fake}/...
data/annotations/
  train.jsonl
  progan_test.jsonl
weights/hh_teacher.pth
weights/fate_step62000.pth
```

```bash
python tools/prepare_evidence.py \
  --annotations /path/to/FORTE/annotations/train.jsonl.gz \
  --image-root data/images --output data/annotations/train.jsonl
python tools/prepare_evidence.py \
  --annotations /path/to/FORTE/annotations/progan_test.jsonl.gz \
  --image-root data/images --output data/annotations/progan_test.jsonl --progan-only
```

The four evidence descriptions are English. The instruction template is preserved from
the trained model, including its original Chinese instructions; changing that template is
a separate modeling change and may change predictions.

## Stage 1: HH teacher

```bash
CUDA_VISIBLE_DEVICES=0 python train_hh.py \
  --train-root data/images/train/progan \
  --progan-root data/images/test/GAN-Test/progan \
  --output-dir outputs/hh --epochs 30 --eval-every 4
```

The trainer applies random crop and horizontal flip during training and deterministic
center crop during validation. It uses AdamW with separate encoder/head and decay/no-decay
parameter groups, warm-up/cosine learning rates, smoothed cross-entropy and an optional
fake-margin term. The standalone stage-1 trainer uses the released HH architecture and
hyperparameters recovered from the HH checkpoint; it is a portable implementation rather
than the unavailable original external training project. Set `--margin-weight 0` for CE only.

Checkpoints include `checkpoint-last.pth` for resume, `checkpoint-best.pth` selected on
ProGAN, and checkpoints at evaluation epochs. Resume with `--resume outputs/hh/checkpoint-last.pth`.

## Stage 2: FATE

```bash
CUDA_VISIBLE_DEVICES=0 python train_fate.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --hh-checkpoint outputs/hh/checkpoint-best.pth \
  --train-jsonl data/annotations/train.jsonl \
  --progan-jsonl data/annotations/progan_test.jsonl \
  --output-dir outputs/fate --epochs 2
```

The anchor/full two-pass objective, restored RNG between passes, frozen HH BatchNorm and
last-valid-token pooling follow the English-evidence research implementation. Checkpoints
are saved every 500 optimizer steps and validation runs every 1,000 steps and at epoch ends.
The learning rates are halved in epoch two. Logs include losses, diagnostics, optimizer-group
learning rates and ProGAN metrics. Inference checkpoints contain trainable parameters only;
Qwen and HH must be loaded separately. `resume-last.pth` also contains optimizer/scaler states.

```bash
CUDA_VISIBLE_DEVICES=0 python train_fate.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --hh-checkpoint outputs/hh/checkpoint-best.pth \
  --train-jsonl data/annotations/train.jsonl \
  --progan-jsonl data/annotations/progan_test.jsonl \
  --output-dir outputs/fate --resume outputs/fate/resume-last.pth
```

## ProGAN evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
  --model-path /path/to/Qwen3-VL-8B-Instruct \
  --hh-checkpoint weights/hh_teacher.pth \
  --checkpoint weights/fate_step62000.pth \
  --progan-jsonl data/annotations/progan_test.jsonl \
  --output-dir outputs/progan
```

Outputs: `progan_metrics.json` (Acc, balanced Acc, AP and class recalls in percent) and
`progan_predictions.csv` (one score per image). The decision rule is fake when
`fake_logit - real_logit > 0`; AP is computed from continuous margins with tied scores grouped.

## Acknowledgments and licenses

This project uses Qwen3-VL, PyTorch/torchvision, pytorch_wavelets and the SAFE-style HH
encoder. Image sources include CNNDetection/ForenSynths, diffusion detection benchmarks
and GenImage. Cite and comply with the relevant source datasets and model licenses.
The MIT license in this repository covers the released project code, not third-party
images, external models or dependencies. A manuscript citation will be added upon publication.
