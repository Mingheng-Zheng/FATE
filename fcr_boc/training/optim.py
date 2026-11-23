# -*- coding: utf-8 -*-

from pathlib import Path
import torch

def build_optimizer(model, args):
    groups = []
    base_lora_params = []
    corrector_params = []
    router_params = []
    head_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if ".A_corr." in n or ".B_corr." in n:
            corrector_params.append(p)
        elif ".A." in n or ".B." in n:
            base_lora_params.append(p)
        elif n.startswith("freq_router."):
            router_params.append(p)
        elif n.startswith("cls_head."):
            head_params.append(p)
        else:
            head_params.append(p)

    if base_lora_params:
        groups.append({
            "params": base_lora_params,
            "lr": args.lora_lr,
            "weight_decay": args.weight_decay,
            "name": "frequency_anchor_lora",
        })
    if corrector_params:
        groups.append({
            "params": corrector_params,
            "lr": args.fcr_corrector_lr,
            "weight_decay": args.weight_decay,
            "name": "teacher_free_corrector",
        })
    if router_params:
        groups.append({
            "params": router_params,
            "lr": args.router_lr,
            "weight_decay": args.weight_decay,
            "name": "freq_router",
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "name": "cls_head",
        })

    print(
        f"[Optim] groups={[(g['name'], len(g['params']), g['lr']) for g in groups]}",
        flush=True,
    )
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def save_trainable(model, out_dir, args, epoch, step, tag, metric):
    p = Path(out_dir) / f"checkpoint-{tag}.pth"
    keep = {}
    for k, v in model.state_dict().items():
        if (
            ".A.weight" in k or ".B.weight" in k
            or ".A_corr.weight" in k or ".B_corr.weight" in k
            or "active_rank_tensor" in k or "rank_score_ema" in k or "rank_score_last" in k
            or k.startswith("freq_router.") or k.startswith("cls_head.")
        ):
            keep[k] = v.detach().cpu()
    torch.save({"model": keep, "args": vars(args), "epoch": epoch, "step": step, "metric": metric}, p)
    print(f"[Save] {p}", flush=True)

