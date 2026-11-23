# -*- coding: utf-8 -*-

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration

from fcr_boc.models.hh_teacher import HHSAFE, load_hh_checkpoint
from fcr_boc.models.gated_lora import GatedLoRALinear, inject_gated_lora, FrequencyGateRouter

class HHGuidedQwenLoRAClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        torch.cuda.set_device(args.gpu)
        self.device = torch.device(f"cuda:{args.gpu}")
        print("[Qwen] loading base model...")
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": args.gpu},
        )
        self.qwen.config.use_cache = False
        for p in self.qwen.parameters():
            p.requires_grad_(False)
        targets = [x.strip() for x in args.lora_targets.split(',') if x.strip()]
        self.gated_loras = inject_gated_lora(
            self.qwen, targets, args.lora_rank, args.lora_alpha, args.lora_dropout, args.max_lora_modules,
            pdr_enable=bool(args.pdr_enable),
            pdr_rank_floor=args.pdr_rank_floor,
            pdr_mask_min_rank=args.pdr_mask_min_rank,
            fcr_enable=bool(args.fcr_enable),
            corrector_rank=args.fcr_corrector_rank,
            corrector_alpha=args.fcr_corrector_alpha,
            corrector_dropout=args.fcr_corrector_dropout,
            corrector_scale=args.fcr_corrector_scale,
            fcr_orthogonalize=bool(args.fcr_orthogonalize),
            boc_enable=bool(args.boc_enable),
            boc_budget_ratio=args.boc_budget_ratio,
            boc_eps=args.boc_eps,
        )
        if len(self.gated_loras) == 0:
            raise RuntimeError(f"No LoRA modules injected. Check --lora_targets={args.lora_targets}")
        print(f"[GatedLoRA] injected modules={len(self.gated_loras)}")
        print(f"[GatedLoRA] first modules={[m.lora_name for m in self.gated_loras[:8]]}")
        if bool(args.fcr_enable):
            print(
                f"[FCR] enabled base_rank={args.lora_rank} corrector_rank={args.fcr_corrector_rank} "
                f"corrector_alpha={args.fcr_corrector_alpha} scale={args.fcr_corrector_scale} "
                f"orthogonalize={bool(args.fcr_orthogonalize)} "
                f"BOC={bool(args.boc_enable)} budget_ratio={args.boc_budget_ratio}",
                flush=True,
            )
        if bool(args.pdr_enable):
            print(f"[PDR] enabled rank_init={args.lora_rank} rank_floor={args.pdr_rank_floor} mask_min={args.pdr_mask_min_rank} coverage={args.pdr_coverage} ema={args.pdr_score_ema}", flush=True)

        hs = getattr(self.qwen.config, "hidden_size", None)
        if hs is None and getattr(self.qwen.config, "text_config", None) is not None:
            hs = self.qwen.config.text_config.hidden_size
        if hs is None:
            raise RuntimeError("Cannot infer Qwen hidden_size")
        self.cls_head = nn.Sequential(nn.LayerNorm(hs), nn.Linear(hs, 2)).to(self.device)
        nn.init.zeros_(self.cls_head[-1].weight)
        nn.init.zeros_(self.cls_head[-1].bias)

        self.hh = HHSAFE().to(self.device)
        load_hh_checkpoint(self.hh, args.hh_ckpt)
        for p in self.hh.parameters(): p.requires_grad_(False)
        self.hh.eval()

        self.freq_router = FrequencyGateRouter(
            512,
            len(self.gated_loras),
            hidden=args.freq_gate_hidden,
            gate_scale=args.freq_gate_scale,
            gate_mode=args.freq_gate_mode,
            residual_beta=args.freq_residual_beta,
            faar_clamp_min=args.faar_clamp_min,
            faar_clamp_max=args.faar_clamp_max,
            faar_renorm_after_clamp=bool(args.faar_renorm_after_clamp),
        ).to(self.device)



    def train(self, mode: bool = True):
        """Keep the frozen HH teacher in eval mode even when the whole model is set to train().

        Critical: HHSAFE contains BatchNorm layers. If model.train() switches HH to training mode,
        BatchNorm running_mean/running_var will be updated by ProGAN train batches even though
        requires_grad=False. Then teacher_ref drifts during training and KD learns a corrupted
        teacher. This override freezes HH statistics.
        """
        super().train(mode)
        if hasattr(self, "hh"):
            self.hh.eval()
        return self

    def freeze_hh_teacher_mode(self):
        """Force HH teacher and all BatchNorm layers to eval mode."""
        self.hh.eval()
        for m in self.hh.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    # ---------------- PDR rank utilities ----------------
    def pdr_enabled(self) -> bool:
        return bool(getattr(self.args, "pdr_enable", False))

    @torch.no_grad()
    def update_pdr_scores_from_grads(self):
        if not self.pdr_enabled():
            return
        beta = float(getattr(self.args, "pdr_score_ema", 0.98))
        for m in self.gated_loras:
            m.update_rank_scores_from_grads(beta=beta)

    @torch.no_grad()
    def pdr_rank_summary(self) -> Dict:
        ranks = [m.active_rank for m in self.gated_loras]
        if not ranks:
            return {}
        rank_arr = np.array(ranks, dtype=np.float32)
        stored = 0
        active = 0
        for m in self.gated_loras:
            stored += m.rank * (m.in_features + m.out_features)
            active += m.active_rank * (m.in_features + m.out_features)
        return {
            "num_modules": len(self.gated_loras),
            "rank_min": int(rank_arr.min()),
            "rank_max": int(rank_arr.max()),
            "rank_mean": float(rank_arr.mean()),
            "rank_median": float(np.median(rank_arr)),
            "rank_init": int(self.args.lora_rank),
            "rank_floor": int(self.args.pdr_rank_floor),
            "active_lora_params_est": int(active),
            "stored_lora_params": int(stored),
            "active_ratio": float(active / max(stored, 1)),
            "shrink_events_total": int(sum(getattr(m, "shrink_events", 0) for m in self.gated_loras)),
        }

    @torch.no_grad()
    def write_pdr_summary(self, out_dir: str, step: int, tag: str):
        if not self.pdr_enabled():
            return
        out = Path(out_dir)
        summary = self.pdr_rank_summary()
        summary.update({"step": int(step), "tag": str(tag)})
        with (out / "pdr_summary.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(
            f"[PDRSummary:{tag}] step={step} rank_mean={summary['rank_mean']:.2f} "
            f"min/max={summary['rank_min']}/{summary['rank_max']} "
            f"active_ratio={summary['active_ratio']:.3f} events={summary['shrink_events_total']}",
            flush=True,
        )

    @torch.no_grad()
    def pdr_maybe_shrink(self, out_dir: str, step: int, force: bool = False, tag: str = "periodic"):
        if not self.pdr_enabled():
            return []
        warmup = int(getattr(self.args, "pdr_warmup_steps", 0))
        every = int(getattr(self.args, "pdr_shrink_every_steps", 0))
        if not force:
            if step < warmup:
                return []
            if every <= 0 or step % every != 0:
                return []

        events = []
        out = Path(out_dir)
        hist_path = out / "pdr_rank_history.csv"
        new_file = not hist_path.exists()
        with hist_path.open("a", encoding="utf-8") as f:
            if new_file:
                f.write("step,tag,module_idx,module_name,old_rank,new_rank,changed,reason,coverage,actual_coverage,score_sum,top_score,score_entropy,active_ratio_after\n")
            for idx, m in enumerate(self.gated_loras):
                ev = m.shrink_active_rank(
                    coverage=float(self.args.pdr_coverage),
                    rank_floor=int(self.args.pdr_rank_floor),
                    max_shrink_per_event=int(self.args.pdr_max_shrink_per_event),
                    reorder=bool(self.args.pdr_reorder),
                    step=int(step),
                )
                if ev is None:
                    continue
                active_ratio = m.active_rank / max(m.rank, 1)
                row = [
                    int(step), str(tag), idx, m.lora_name,
                    ev.get("old_rank", m.active_rank), ev.get("new_rank", m.active_rank),
                    int(bool(ev.get("changed", False))), ev.get("reason", ""),
                    ev.get("coverage", ""), ev.get("actual_coverage", ""),
                    ev.get("score_sum", ""), ev.get("top_score", ""),
                    ev.get("score_entropy", ""), active_ratio,
                ]
                f.write(",".join(str(x) for x in row) + "\n")
                if ev.get("changed", False):
                    events.append({"module_idx": idx, "module_name": m.lora_name, **ev})
        if events:
            print(f"[PDRShrink:{tag}] step={step} changed_modules={len(events)}", flush=True)
            for ev in events[:12]:
                print(f"  - {ev['module_idx']:03d} {ev['module_name']}: {ev['old_rank']} -> {ev['new_rank']} cov={ev.get('actual_coverage',0):.3f} entropy={ev.get('score_entropy',0):.3f}", flush=True)
            if len(events) > 12:
                print(f"  ... {len(events)-12} more modules changed", flush=True)
        else:
            print(f"[PDRShrink:{tag}] step={step} no rank changed", flush=True)
        self.write_pdr_summary(out_dir, step, tag=f"after_shrink_{tag}")
        return events

    @torch.no_grad()
    def dump_pdr_rank_table(self, out_dir: str, step: int, tag: str):
        if not self.pdr_enabled():
            return
        out = Path(out_dir)
        path = out / f"pdr_rank_table_{tag}_step{step}.csv"
        with path.open("w", encoding="utf-8") as f:
            f.write("module_idx,module_name,active_rank,rank_init,rank_floor,score_sum,score_top,score_entropy,last_rank_used,shrink_events,last_shrink_step\n")
            for idx, m in enumerate(self.gated_loras):
                scores = m.rank_score_ema[:m.active_rank].detach().float()
                score_sum = float(scores.sum().cpu()) if scores.numel() else 0.0
                score_top = float(scores.max().cpu()) if scores.numel() else 0.0
                score_entropy = GatedLoRALinear._score_entropy(scores) if scores.numel() else 0.0
                row = [
                    idx, m.lora_name, m.active_rank, m.rank, m.pdr_rank_floor,
                    score_sum, score_top, score_entropy, m.last_rank_used,
                    getattr(m, "shrink_events", 0), getattr(m, "last_shrink_step", -1)
                ]
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"[PDRDump] wrote {path}", flush=True)


    # ---------------- FCR utilities ----------------
    def fcr_enabled(self) -> bool:
        return bool(getattr(self.args, "fcr_enable", False))

    def set_fcr_mode(self, mode: str):
        for module in self.gated_loras:
            module.set_fcr_mode(mode)

    def set_fcr_diagnostics(self, enabled: bool):
        for module in self.gated_loras:
            module.set_collect_fcr_stats(enabled)

    def clear_fcr_runtime_cache(self):
        for module in self.gated_loras:
            module.last_corr_energy_tensor = None
            module.last_fcr_stats = {}

    def collect_fcr_regularizer_and_stats(self):
        device = self.device
        energy_terms = []
        stat_buckets = {}
        active_stats = 0
        for module in self.gated_loras:
            if module.last_corr_energy_tensor is not None:
                energy_terms.append(module.last_corr_energy_tensor)
            d = module.last_fcr_stats
            if d:
                active_stats += 1
                for k, v in d.items():
                    if torch.is_tensor(v):
                        stat_buckets.setdefault(k, []).append(v)
        corr_reg = torch.stack(energy_terms).mean() if energy_terms else torch.zeros((), device=device)
        out = {}
        for k, vals in stat_buckets.items():
            out[f"fcr_{k}"] = float(torch.stack([v.float() for v in vals]).mean().item())
        out["fcr_num_active_modules"] = int(active_stats)
        return corr_reg, out

    def qwen_device(self):
        return self.device

    def set_lora_gates(self, gates: torch.Tensor):
        for i, module in enumerate(self.gated_loras):
            module.set_gate(gates[:, i])

    def clear_lora_gates(self):
        for module in self.gated_loras:
            module.set_gate(None)

    def teacher_trust_score(self, teacher_logits: torch.Tensor, labels: Optional[torch.Tensor] = None):
        """Sample-wise HH teacher trust score in [0, 1].

        The confidence is only used to modulate KD and/or gate residuals;
        it is detached and does not train the HH teacher.
        """
        prob = F.softmax(teacher_logits.float(), dim=-1)
        conf, pred = prob.max(dim=-1)
        tau = float(self.args.trust_tau)
        sigma = max(float(self.args.trust_sigma), 1e-6)
        trust = torch.sigmoid((conf - tau) / sigma)
        if bool(getattr(self.args, "trust_label_aware", False)) and labels is not None:
            wrong = (pred != labels.to(pred.device)).float()
            wrong_scale = float(getattr(self.args, "trust_wrong_scale", 0.0))
            trust = trust * (1.0 - wrong) + trust * wrong * wrong_scale
        return trust.detach(), conf.detach(), pred.detach()

    def prepare_qwen_inputs(self, qi: Dict[str, torch.Tensor]):
        return {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in qi.items()}

    def qwen_forward_hidden(self, qi: Dict[str, torch.Tensor], already_on_device: bool = False):
        qi = dict(qi) if already_on_device else self.prepare_qwen_inputs(qi)
        qi["use_cache"] = False
        qi["return_dict"] = True
        if hasattr(self.qwen, "model"):
            out = self.qwen.model(**qi)
            hidden = getattr(out, "last_hidden_state", None)
            if hidden is None:
                hidden = out[0]
        else:
            out = self.qwen(**qi, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1]
        attn = qi.get("attention_mask", None)
        if attn is None:
            h = hidden[:, -1, :]
        else:
            # Robust last-valid-token pooling for both left and right padding.
            # The previous `attention_mask.sum()-1` formula is only correct for
            # right padding; this project explicitly uses left padding.
            mask = attn.to(device=hidden.device, dtype=torch.bool)
            positions = torch.arange(mask.size(1), device=hidden.device).unsqueeze(0)
            idx = positions.masked_fill(~mask, -1).max(dim=1).values
            if bool((idx < 0).any().item()):
                raise RuntimeError("Found an input sample with no valid token in attention_mask")
            h = hidden[torch.arange(hidden.size(0), device=hidden.device), idx]
        return h

    def _prepare_fcr_train_context(self, qwen_inputs, freq_x, labels):
        dev = self.device
        labels = labels.to(dev, non_blocking=True)
        freq_x = freq_x.to(dev, dtype=torch.float32, non_blocking=True)
        self.freeze_hh_teacher_mode()
        with torch.no_grad():
            hh_out = self.hh(freq_x)
            z_hh = hh_out["z"].detach()
            teacher_logits = hh_out["logits"].detach()
        trust_score, teacher_conf, teacher_pred = self.teacher_trust_score(teacher_logits, labels)
        qi_gpu = self.prepare_qwen_inputs(qwen_inputs)
        return {"qwen_inputs": qi_gpu, "labels": labels, "z_hh": z_hh, "teacher_logits": teacher_logits, "trust_score": trust_score, "teacher_conf": teacher_conf, "teacher_pred": teacher_pred}

    def _gates_from_fcr_context(self, ctx):
        gates = self.freq_router(ctx["z_hh"])
        if bool(getattr(self.args, "freq_gate_use_trust", False)):
            trust = ctx["trust_score"]
            gates = 1.0 + trust[:, None].to(gates.device, gates.dtype) * (gates - 1.0)
        if self.training and float(getattr(self.args, "freq_dropout_prob", 0.0)) > 0:
            drop = torch.rand(gates.size(0), 1, device=gates.device) < float(self.args.freq_dropout_prob)
            gates = torch.where(drop, torch.ones_like(gates), gates)
        return gates

    def _gate_regularizer(self, gates):
        gate_float = gates.float()
        gate_reg_type = str(getattr(self.args, "gate_reg_type", "neutral_mse"))
        if gate_reg_type == "neutral_mse":
            return (gate_float - 1.0).pow(2).mean()
        if gate_reg_type == "mean_only":
            return (gate_float.mean(dim=1) - 1.0).pow(2).mean()
        if gate_reg_type == "none":
            return torch.zeros((), device=self.device)
        raise ValueError(f"Unknown gate_reg_type={gate_reg_type}")

    def fcr_anchor_stage(self, ctx, collect_details: bool = False):
        labels = ctx["labels"]
        gates = self._gates_from_fcr_context(ctx)
        self.set_fcr_diagnostics(False)
        self.set_lora_gates(gates)
        self.set_fcr_mode("freq_only")
        h_anchor = self.qwen_forward_hidden(ctx["qwen_inputs"], already_on_device=True)
        logits_anchor = self.cls_head(h_anchor.float())
        self.clear_lora_gates(); self.set_fcr_mode("full")
        loss_ce_anchor = F.cross_entropy(logits_anchor.float(), labels)
        loss_kd = torch.zeros((), device=self.device)
        trust_mean = torch.ones((), device=self.device)
        if self.args.kd_weight > 0:
            T = float(self.args.kd_temperature)
            teacher_prob = F.softmax(ctx["teacher_logits"].float() / T, dim=-1)
            anchor_logp = F.log_softmax(logits_anchor.float() / T, dim=-1)
            if bool(getattr(self.args, "trust_kd", False)):
                kd_each = F.kl_div(anchor_logp, teacher_prob, reduction="none").sum(dim=-1) * (T * T)
                kd_weight_each = ctx["trust_score"].to(self.device, dtype=kd_each.dtype)
                loss_kd = (kd_each * kd_weight_each).mean(); trust_mean = kd_weight_each.mean()
            else:
                loss_kd = F.kl_div(anchor_logp, teacher_prob, reduction="batchmean") * (T * T)
        loss_anchor_stage = float(self.args.fcr_anchor_ce_weight) * loss_ce_anchor + float(self.args.kd_weight) * loss_kd
        details = {}; anchor_logits_for_log = None
        if collect_details:
            pred = logits_anchor.detach().argmax(dim=1)
            details = {"loss_ce_anchor": float(loss_ce_anchor.detach().item()), "loss_kd": float(loss_kd.detach().item()), "fcr_anchor_acc": float((pred == labels).float().mean().item()), "trust_mean": float(trust_mean.detach().item())}
            anchor_logits_for_log = logits_anchor.detach()
        return loss_anchor_stage, details, anchor_logits_for_log

    def fcr_full_stage(self, ctx, collect_details: bool = False, anchor_logits_for_log=None):
        labels = ctx["labels"]
        gates = self._gates_from_fcr_context(ctx)
        self.set_fcr_diagnostics(collect_details)
        self.set_lora_gates(gates); self.set_fcr_mode("full")
        h_full = self.qwen_forward_hidden(ctx["qwen_inputs"], already_on_device=True)
        logits = self.cls_head(h_full.float())
        corr_reg, fcr_stats = self.collect_fcr_regularizer_and_stats()
        self.clear_lora_gates(); self.set_fcr_mode("full")
        loss_ce_final = F.cross_entropy(logits.float(), labels)
        gate_reg = self._gate_regularizer(gates)
        loss_full_stage = float(self.args.fcr_final_ce_weight) * loss_ce_final + float(self.args.gate_reg_weight) * gate_reg + float(self.args.fcr_corrector_reg_weight) * corr_reg
        details = {}
        if collect_details:
            final_pred = logits.detach().argmax(dim=1)
            teacher_acc = (ctx["teacher_pred"].to(self.device) == labels).float().mean()
            gate_float = gates.detach().float()
            details = {"loss_ce_final": float(loss_ce_final.detach().item()), "gate_reg": float(gate_reg.detach().item()), "fcr_corr_reg": float(corr_reg.detach().item()), "fcr_final_acc": float((final_pred == labels).float().mean().item()), "gate_mean": float(gate_float.mean().item()), "gate_std": float(gate_float.std().item()), "gate_min": float(gate_float.min().item()), "gate_max": float(gate_float.max().item()), "gate_sample_mean_std": float(gate_float.mean(dim=1).std().item()), "teacher_conf": float(ctx["teacher_conf"].float().mean().item()), "teacher_acc_batch": float(teacher_acc.item())}
            if anchor_logits_for_log is not None:
                details["fcr_logit_delta"] = float((logits.detach().float() - anchor_logits_for_log.float()).abs().mean().item())
            details.update(fcr_stats)
        return loss_full_stage, details, logits

    def forward(self, qwen_inputs, freq_x, labels=None):
        dev = self.device
        freq_x = freq_x.to(dev, dtype=torch.float32, non_blocking=True)

        # Frozen single HH teacher.
        self.freeze_hh_teacher_mode()
        with torch.no_grad():
            hh_out = self.hh(freq_x)
            z_hh = hh_out["z"].detach()
            teacher_logits = hh_out["logits"].detach()

        gates = self.freq_router(z_hh)

        trust_score, teacher_conf, teacher_pred = self.teacher_trust_score(
            teacher_logits,
            labels if labels is not None else None,
        )
        if bool(getattr(self.args, "freq_gate_use_trust", False)):
            gates = 1.0 + trust_score[:, None].to(gates.device, gates.dtype) * (gates - 1.0)

        if labels is not None and self.training and float(getattr(self.args, "freq_dropout_prob", 0.0)) > 0:
            drop = torch.rand(gates.size(0), 1, device=gates.device) < float(self.args.freq_dropout_prob)
            gates = torch.where(drop, torch.ones_like(gates), gates)

        self.set_lora_gates(gates)

        # ------------------------------------------------------------------------------------------
        # Inference/evaluation: one full forward only.
        # ------------------------------------------------------------------------------------------
        if labels is None or not self.fcr_enabled():
            self.set_fcr_mode("full" if self.fcr_enabled() else "freq_only")
            h = self.qwen_forward_hidden(qwen_inputs)
            logits = self.cls_head(h.float())
            corr_reg, fcr_stats = self.collect_fcr_regularizer_and_stats()
            self.clear_lora_gates()
            self.set_fcr_mode("full")

            od = {
                "teacher_logits": teacher_logits,
                "gates": gates,
                "trust_score": trust_score,
                "teacher_conf": teacher_conf,
                "anchor_logits": None,
                "fcr_stats": fcr_stats,
            }

            # Baseline-compatible training if FCR is disabled.
            if labels is None:
                return logits, od, None, {}

            labels = labels.to(dev)
            loss_ce = F.cross_entropy(logits.float(), labels)
            loss_kd = torch.tensor(0.0, device=dev)
            trust_mean = torch.tensor(1.0, device=dev)

            if self.args.kd_weight > 0:
                T = float(self.args.kd_temperature)
                teacher_prob = F.softmax(teacher_logits.float() / T, dim=-1)
                student_logp = F.log_softmax(logits.float() / T, dim=-1)
                if bool(getattr(self.args, "trust_kd", False)):
                    kd_each = F.kl_div(
                        student_logp, teacher_prob, reduction="none"
                    ).sum(dim=-1) * (T * T)
                    kd_weight_each = trust_score.to(dev, dtype=kd_each.dtype)
                    loss_kd = (kd_each * kd_weight_each).mean()
                    trust_mean = kd_weight_each.mean()
                else:
                    loss_kd = F.kl_div(
                        student_logp, teacher_prob, reduction="batchmean"
                    ) * (T * T)

            gate_float = gates.float()
            gate_reg_type = str(getattr(self.args, "gate_reg_type", "neutral_mse"))
            if gate_reg_type == "neutral_mse":
                gate_reg = (gate_float - 1.0).pow(2).mean()
            elif gate_reg_type == "mean_only":
                gate_reg = (gate_float.mean(dim=1) - 1.0).pow(2).mean()
            elif gate_reg_type == "none":
                gate_reg = torch.tensor(0.0, device=dev)
            else:
                raise ValueError(f"Unknown gate_reg_type={gate_reg_type}")

            loss = (
                loss_ce
                + float(self.args.kd_weight) * loss_kd
                + float(self.args.gate_reg_weight) * gate_reg
            )
            teacher_acc = (teacher_pred.to(dev) == labels).float().mean()
            details = {
                "loss_ce": float(loss_ce.detach().cpu()),
                "loss_ce_anchor": float(loss_ce.detach().cpu()),
                "loss_ce_final": float(loss_ce.detach().cpu()),
                "loss_kd": float(loss_kd.detach().cpu()),
                "gate_reg": float(gate_reg.detach().cpu()),
                "gate_mean": float(gate_float.detach().mean().cpu()),
                "gate_std": float(gate_float.detach().std().cpu()),
                "gate_min": float(gate_float.detach().min().cpu()),
                "gate_max": float(gate_float.detach().max().cpu()),
                "gate_sample_mean_std": float(gate_float.detach().mean(dim=1).std().cpu()),
                "trust_mean": float(trust_mean.detach().cpu()),
                "teacher_conf": float(teacher_conf.detach().float().mean().cpu()),
                "teacher_acc_batch": float(teacher_acc.detach().cpu()),
                "loss_total": float(loss.detach().cpu()),
            }
            return logits, od, loss, details

        # ------------------------------------------------------------------------------------------
        # FCR training:
        #   Pass A: frequency anchor only; KD is confined here.
        #   Pass B: frequency anchor + teacher-free orthogonal corrector; label CE only.
        #
        # RNG synchronization is important. Since Corrective LoRA B starts at zero and its dropout
        # defaults to 0, both passes begin from nearly identical stochastic trajectories.
        # ------------------------------------------------------------------------------------------
        labels = labels.to(dev)

        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state(self.device)

        # Anchor path.
        self.set_fcr_mode("freq_only")
        h_anchor = self.qwen_forward_hidden(qwen_inputs)
        logits_anchor = self.cls_head(h_anchor.float())

        # Reuse the same dropout/random trajectory for the full path.
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, self.device)

        # Full path.
        self.set_fcr_mode("full")
        h_full = self.qwen_forward_hidden(qwen_inputs)
        logits = self.cls_head(h_full.float())

        corr_reg, fcr_stats = self.collect_fcr_regularizer_and_stats()

        self.clear_lora_gates()
        self.set_fcr_mode("full")

        # Two CE terms whose weights sum to one. At zero-init they reduce to the old CE.
        loss_ce_anchor = F.cross_entropy(logits_anchor.float(), labels)
        loss_ce_final = F.cross_entropy(logits.float(), labels)
        loss_ce = (
            float(self.args.fcr_anchor_ce_weight) * loss_ce_anchor
            + float(self.args.fcr_final_ce_weight) * loss_ce_final
        )

        # KD is applied ONLY to the frequency anchor logits.
        loss_kd = torch.tensor(0.0, device=dev)
        trust_mean = torch.tensor(1.0, device=dev)
        if self.args.kd_weight > 0:
            T = float(self.args.kd_temperature)
            teacher_prob = F.softmax(teacher_logits.float() / T, dim=-1)
            anchor_logp = F.log_softmax(logits_anchor.float() / T, dim=-1)

            if bool(getattr(self.args, "trust_kd", False)):
                kd_each = F.kl_div(
                    anchor_logp, teacher_prob, reduction="none"
                ).sum(dim=-1) * (T * T)
                kd_weight_each = trust_score.to(dev, dtype=kd_each.dtype)
                loss_kd = (kd_each * kd_weight_each).mean()
                trust_mean = kd_weight_each.mean()
            else:
                loss_kd = F.kl_div(
                    anchor_logp, teacher_prob, reduction="batchmean"
                ) * (T * T)

        gate_float = gates.float()
        gate_reg_type = str(getattr(self.args, "gate_reg_type", "neutral_mse"))
        if gate_reg_type == "neutral_mse":
            gate_reg = (gate_float - 1.0).pow(2).mean()
        elif gate_reg_type == "mean_only":
            gate_reg = (gate_float.mean(dim=1) - 1.0).pow(2).mean()
        elif gate_reg_type == "none":
            gate_reg = torch.tensor(0.0, device=dev)
        else:
            raise ValueError(f"Unknown gate_reg_type={gate_reg_type}")

        loss = (
            loss_ce
            + float(self.args.kd_weight) * loss_kd
            + float(self.args.gate_reg_weight) * gate_reg
            + float(self.args.fcr_corrector_reg_weight) * corr_reg
        )

        teacher_acc = (teacher_pred.to(dev) == labels).float().mean()
        anchor_pred = logits_anchor.detach().argmax(dim=1)
        final_pred = logits.detach().argmax(dim=1)

        details = {
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_ce_anchor": float(loss_ce_anchor.detach().cpu()),
            "loss_ce_final": float(loss_ce_final.detach().cpu()),
            "loss_kd": float(loss_kd.detach().cpu()),
            "gate_reg": float(gate_reg.detach().cpu()),
            "fcr_corr_reg": float(corr_reg.detach().cpu()),
            "fcr_anchor_acc": float((anchor_pred == labels).float().mean().cpu()),
            "fcr_final_acc": float((final_pred == labels).float().mean().cpu()),
            "fcr_logit_delta": float(
                (logits.detach().float() - logits_anchor.detach().float()).abs().mean().cpu()
            ),
            "gate_mean": float(gate_float.detach().mean().cpu()),
            "gate_std": float(gate_float.detach().std().cpu()),
            "gate_min": float(gate_float.detach().min().cpu()),
            "gate_max": float(gate_float.detach().max().cpu()),
            "gate_sample_mean_std": float(gate_float.detach().mean(dim=1).std().cpu()),
            "trust_mean": float(trust_mean.detach().cpu()),
            "teacher_conf": float(teacher_conf.detach().float().mean().cpu()),
            "teacher_acc_batch": float(teacher_acc.detach().cpu()),
            "loss_total": float(loss.detach().cpu()),
        }
        details.update(fcr_stats)

        if bool(getattr(self.args, "pdr_enable", False)):
            ranks = [m.active_rank for m in self.gated_loras]
            if ranks:
                details["pdr_rank_mean"] = float(np.mean(ranks))
                details["pdr_rank_min"] = float(np.min(ranks))
                details["pdr_rank_max"] = float(np.max(ranks))
                details["pdr_last_rank_used_mean"] = float(
                    np.mean([m.last_rank_used for m in self.gated_loras])
                )

        od = {
            "teacher_logits": teacher_logits,
            "gates": gates,
            "trust_score": trust_score,
            "teacher_conf": teacher_conf,
            "anchor_logits": logits_anchor.detach(),
            "fcr_stats": fcr_stats,
        }
        return logits, od, loss, details
