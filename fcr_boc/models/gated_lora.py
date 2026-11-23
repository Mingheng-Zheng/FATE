# -*- coding: utf-8 -*-

import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Frequency-guided LoRA ----------------
class GatedLoRALinear(nn.Module):
    """Frozen base Linear + trainable LoRA branch gated per sample by HH feature.

    PDR-LoRA additions:
      - rank_init is the actual stored LoRA rank.
      - active_rank is the current usable prefix rank for this module.
      - during training, a random prefix rank b is sampled in [pdr_mask_min_rank, active_rank],
        encouraging important rank-1 updates to move toward low indices.
      - after backward, each rank component receives an EMA sensitivity score:
            0.5 * ( |<A_i, grad A_i>| + |<B_i, grad B_i>| )
      - periodically, active ranks are sorted by EMA score and shrunk using a coverage rule.

    Device/dtype rule:
      Qwen is loaded on GPU through device_map, but newly created LoRA A/B are CPU tensors
      by default. Therefore A/B must follow the wrapped base layer device. LoRA math is fp32.
    """
    def __init__(
        self,
        base: nn.Linear,
        rank: int = 4,
        alpha: float = 16.0,
        dropout: float = 0.05,
        pdr_enable: bool = False,
        pdr_rank_floor: int = 4,
        pdr_mask_min_rank: int = 4,
        fcr_enable: bool = False,
        corrector_rank: int = 2,
        corrector_alpha: float = 8.0,
        corrector_dropout: float = 0.0,
        corrector_scale: float = 1.0,
        fcr_orthogonalize: bool = True,
        boc_enable: bool = False,
        boc_budget_ratio: float = 1.0,
        boc_eps: float = 1e-6,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("GatedLoRALinear wraps nn.Linear only")
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = int(rank)
        self.scaling = float(alpha) / max(self.rank, 1)
        self.dropout = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        self.A = nn.Linear(self.in_features, self.rank, bias=False)
        self.B = nn.Linear(self.rank, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

        base_param = next(self.base.parameters())
        self.A.to(device=base_param.device, dtype=torch.float32)
        self.B.to(device=base_param.device, dtype=torch.float32)

        # ---------------- FCR teacher-free corrective LoRA ----------------
        self.fcr_enable = bool(fcr_enable)
        self.corrector_rank = max(1, int(corrector_rank))
        self.corrector_scaling = float(corrector_alpha) / max(self.corrector_rank, 1)
        self.corrector_scale = float(corrector_scale)
        self.fcr_orthogonalize = bool(fcr_orthogonalize)

        # BOC: cap orthogonal correction energy relative to frequency residual energy.
        self.boc_enable = bool(boc_enable)
        self.boc_budget_ratio = max(0.0, float(boc_budget_ratio))
        self.boc_eps = max(1e-12, float(boc_eps))

        self.corrector_dropout = (
            nn.Dropout(float(corrector_dropout))
            if corrector_dropout and corrector_dropout > 0
            else nn.Identity()
        )

        if self.fcr_enable:
            self.A_corr = nn.Linear(self.in_features, self.corrector_rank, bias=False)
            self.B_corr = nn.Linear(self.corrector_rank, self.out_features, bias=False)
            nn.init.kaiming_uniform_(self.A_corr.weight, a=math.sqrt(5))
            # Critical: exact zero-init makes the initial full path equal to the anchor path.
            nn.init.zeros_(self.B_corr.weight)
            self.A_corr.to(device=base_param.device, dtype=torch.float32)
            self.B_corr.to(device=base_param.device, dtype=torch.float32)
        else:
            self.A_corr = None
            self.B_corr = None

        self._fcr_mode = "full"
        self.last_corr_energy_tensor = None
        self.last_fcr_stats = {}
        self.collect_fcr_stats = False

        self._gate = None
        self.lora_name = ""

        # PDR state. active_rank_tensor is a buffer so it can be saved in state_dict.
        self.pdr_enable = bool(pdr_enable)
        floor = max(1, min(int(pdr_rank_floor), self.rank))
        mask_min = max(1, min(int(pdr_mask_min_rank), self.rank))
        self.pdr_rank_floor = floor
        self.pdr_mask_min_rank = min(mask_min, self.rank)
        self.register_buffer("active_rank_tensor", torch.tensor(self.rank, dtype=torch.long, device=base_param.device))
        self.register_buffer("rank_score_ema", torch.zeros(self.rank, dtype=torch.float32, device=base_param.device))
        self.register_buffer("rank_score_last", torch.zeros(self.rank, dtype=torch.float32, device=base_param.device))
        self.register_buffer("rank_update_count", torch.zeros((), dtype=torch.long, device=base_param.device))
        self.last_rank_used = self.rank
        self.last_prefix_mask = None
        self.shrink_events = 0
        self.last_shrink_step = -1

    @property
    def active_rank(self) -> int:
        return int(self.active_rank_tensor.detach().item())

    def set_gate(self, gate: Optional[torch.Tensor]):
        self._gate = gate

    def set_fcr_mode(self, mode: str):
        if mode not in {"freq_only", "full"}:
            raise ValueError(f"Unknown FCR mode={mode}")
        self._fcr_mode = mode

    def set_collect_fcr_stats(self, enabled: bool):
        self.collect_fcr_stats = bool(enabled)

    def _make_prefix_mask(self, device, dtype):
        """Return prefix mask of length rank. Evaluation uses full active rank."""
        active = max(1, min(self.active_rank, self.rank))
        if (not self.pdr_enable) or (not self.training):
            b = active
        else:
            lo = max(1, min(self.pdr_mask_min_rank, active))
            # Using Python random keeps behavior tied to global seed.
            b = random.randint(lo, active)
        mask = torch.zeros(self.rank, device=device, dtype=dtype)
        mask[:b] = 1.0
        self.last_rank_used = int(b)
        self.last_prefix_mask = mask.detach()
        return mask

    def forward(self, x):
        y = self.base(x)
        x_fp = x.to(device=self.A.weight.device, dtype=self.A.weight.dtype)

        x_lora = self.dropout(x_fp)
        a = self.A(x_lora)
        if self.pdr_enable:
            mask = self._make_prefix_mask(a.device, a.dtype)
            while mask.dim() < a.dim():
                mask = mask.unsqueeze(0)
            a = a * mask
        else:
            self.last_rank_used = self.rank
            self.last_prefix_mask = None

        delta_freq = self.B(a) * self.scaling
        gate = self._gate
        if gate is not None:
            if gate.dim() == 1:
                gate = gate[:, None]
            while gate.dim() < delta_freq.dim():
                gate = gate.unsqueeze(1)
            gate = gate.to(device=delta_freq.device, dtype=delta_freq.dtype)
            delta_freq = delta_freq * gate

        if (not self.fcr_enable) or self._fcr_mode == "freq_only":
            self.last_corr_energy_tensor = None
            self.last_fcr_stats = {}
            return y + delta_freq.to(device=y.device, dtype=y.dtype)

        x_corr = self.corrector_dropout(x_fp)
        delta_corr_raw = self.B_corr(self.A_corr(x_corr)) * self.corrector_scaling

        f_ref = delta_freq.detach()
        c_raw = delta_corr_raw
        if self.fcr_orthogonalize:
            dot = torch.sum(c_raw * f_ref, dim=-1, keepdim=True, dtype=torch.float32)
            denom = torch.sum(f_ref * f_ref, dim=-1, keepdim=True, dtype=torch.float32).clamp_min(self.boc_eps)
            proj_coeff = (dot / denom).to(dtype=c_raw.dtype)
            delta_corr_perp = c_raw - proj_coeff * f_ref
        else:
            delta_corr_perp = c_raw
        delta_corr_perp = delta_corr_perp * self.corrector_scale

        reduce_dims = tuple(range(1, delta_corr_perp.dim()))
        elems = 1
        for d in delta_corr_perp.shape[1:]:
            elems *= int(d)

        freq_ms = torch.sum(
            f_ref * f_ref,
            dim=reduce_dims,
            keepdim=True,
            dtype=torch.float32,
        ) / max(elems, 1)
        corr_ms_pre = torch.sum(
            delta_corr_perp * delta_corr_perp,
            dim=reduce_dims,
            keepdim=True,
            dtype=torch.float32,
        ) / max(elems, 1)

        # IMPORTANT NUMERICAL FIX:
        # Both frequency LoRA B and Corrector B are zero-initialized.
        # Using sqrt(ms) gives an infinite derivative at ms=0 and can produce
        # 0 * inf -> NaN on the very first backward.
        #
        # Clamp the mean-square to eps^2 BEFORE sqrt. This gives:
        #   - finite derivative at zero;
        #   - neutral initial BOC scale = 1 when both residuals are zero;
        #   - direct gradient to the zero-initialized Corrector B matrix.
        rms_floor_sq = self.boc_eps * self.boc_eps
        freq_rms = torch.sqrt(freq_ms.clamp_min(rms_floor_sq))
        corr_rms_pre = torch.sqrt(corr_ms_pre.clamp_min(rms_floor_sq))

        if self.boc_enable:
            budget = self.boc_budget_ratio * freq_rms
            boc_scale_fp32 = torch.clamp(
                budget / corr_rms_pre.clamp_min(self.boc_eps),
                max=1.0,
            )
            boc_scale = boc_scale_fp32.to(dtype=delta_corr_perp.dtype)
            delta_corr = delta_corr_perp * boc_scale
        else:
            boc_scale = torch.ones_like(freq_rms, dtype=delta_corr_perp.dtype)
            delta_corr = delta_corr_perp

        self.last_corr_energy_tensor = torch.mean(delta_corr * delta_corr, dtype=torch.float32)

        if self.collect_fcr_stats:
            with torch.no_grad():
                corr_ms_post = torch.sum(
                    delta_corr * delta_corr,
                    dim=reduce_dims,
                    keepdim=True,
                    dtype=torch.float32,
                ) / max(elems, 1)
                corr_rms_post = torch.sqrt(
                    corr_ms_post.clamp_min(rms_floor_sq)
                )
                f_norm = torch.linalg.vector_norm(f_ref, dim=-1).float().mean()
                c_raw_norm = torch.linalg.vector_norm(c_raw, dim=-1).float().mean()
                c_pre_norm = torch.linalg.vector_norm(delta_corr_perp, dim=-1).float().mean()
                c_post_norm = torch.linalg.vector_norm(delta_corr, dim=-1).float().mean()
                cos_raw = F.cosine_similarity(c_raw, f_ref, dim=-1, eps=1e-8).abs().float().mean()
                cos_perp = F.cosine_similarity(delta_corr, f_ref, dim=-1, eps=1e-8).abs().float().mean()
                pre_ratio = corr_rms_pre / (freq_rms + self.boc_eps)
                post_ratio = corr_rms_post / (freq_rms + self.boc_eps)
                if self.boc_enable:
                    budget_util = corr_rms_post / (self.boc_budget_ratio * freq_rms + self.boc_eps)
                    clip_frac = (boc_scale.float() < 0.999).float().mean()
                else:
                    budget_util = torch.zeros_like(post_ratio)
                    clip_frac = torch.zeros((), device=post_ratio.device)
                self.last_fcr_stats = {
                    "freq_norm": f_norm.detach(),
                    "corr_raw_norm": c_raw_norm.detach(),
                    "corr_perp_pre_norm": c_pre_norm.detach(),
                    "corr_perp_norm": c_post_norm.detach(),
                    "corr_to_freq_ratio": post_ratio.mean().detach(),
                    "corr_to_freq_ratio_pre": pre_ratio.mean().detach(),
                    "cos_raw_abs": cos_raw.detach(),
                    "cos_perp_abs": cos_perp.detach(),
                    "boc_scale_mean": boc_scale.float().mean().detach(),
                    "boc_scale_min": boc_scale.float().min().detach(),
                    "boc_clip_frac": clip_frac.detach(),
                    "boc_budget_util": budget_util.mean().detach(),
                }
        else:
            self.last_fcr_stats = {}

        return y + delta_freq.to(device=y.device, dtype=y.dtype) + delta_corr.to(device=y.device, dtype=y.dtype)

    @torch.no_grad()
    def update_rank_scores_from_grads(self, beta: float = 0.98):
        """Update EMA loss-sensitivity score for each rank-1 component.

        Should be called after backward and after GradScaler unscale_ if AMP is used.
        """
        if not self.pdr_enable:
            return None
        if self.A.weight.grad is None or self.B.weight.grad is None:
            return None

        # A: [rank, in_features], B: [out_features, rank]
        score_A = (self.A.weight.detach() * self.A.weight.grad.detach()).abs().sum(dim=1)
        score_B = (self.B.weight.detach() * self.B.weight.grad.detach()).abs().sum(dim=0)
        score = 0.5 * (score_A + score_B)

        # Only update currently active prefix. Inactive ranks should not dominate because their grads are zero.
        active = max(1, min(self.active_rank, self.rank))
        active_mask = torch.zeros_like(score)
        active_mask[:active] = 1.0
        score = score * active_mask

        b = float(beta)
        self.rank_score_ema.mul_(b).add_(score.float(), alpha=(1.0 - b))
        self.rank_score_last.copy_(score.float())
        self.rank_update_count.add_(1)
        return score

    @torch.no_grad()
    def shrink_active_rank(
        self,
        coverage: float = 0.95,
        rank_floor: Optional[int] = None,
        max_shrink_per_event: int = 2,
        reorder: bool = True,
        step: int = -1,
    ):
        """Shrink active prefix rank according to EMA coverage.

        Returns an event dict for logging.
        """
        if not self.pdr_enable:
            return None

        old = self.active_rank
        floor = self.pdr_rank_floor if rank_floor is None else int(rank_floor)
        floor = max(1, min(floor, self.rank))
        if old <= floor:
            return {
                "changed": False,
                "old_rank": old,
                "new_rank": old,
                "reason": "already_at_floor",
                "coverage": float(coverage),
                "score_sum": float(self.rank_score_ema[:old].sum().detach().cpu()),
            }

        scores = self.rank_score_ema[:old].float().clone()
        score_sum = float(scores.sum().detach().cpu())
        if score_sum <= 0 or not torch.isfinite(scores).all():
            return {
                "changed": False,
                "old_rank": old,
                "new_rank": old,
                "reason": "invalid_or_zero_scores",
                "coverage": float(coverage),
                "score_sum": score_sum,
            }

        order = torch.argsort(scores, descending=True)
        sorted_scores = scores[order]
        cumsum = torch.cumsum(sorted_scores, dim=0)
        cov = cumsum / (cumsum[-1] + 1e-12)
        keep = int((cov >= float(coverage)).nonzero(as_tuple=False)[0].item() + 1)
        keep = max(floor, min(keep, old))

        if max_shrink_per_event and max_shrink_per_event > 0:
            keep = max(keep, old - int(max_shrink_per_event))

        changed = keep < old

        if reorder:
            # Reorder active ranks so important rank-1 directions become prefix slots.
            # Keep selected important ranks first, then remaining active ranks, then inactive ranks.
            keep_order = order[:keep]
            rest_active = order[keep:]
            inactive = torch.arange(old, self.rank, device=order.device)
            perm = torch.cat([keep_order, rest_active, inactive], dim=0).long()

            self.A.weight.data.copy_(self.A.weight.data[perm, :])
            self.B.weight.data.copy_(self.B.weight.data[:, perm])
            self.rank_score_ema.data.copy_(self.rank_score_ema.data[perm])
            self.rank_score_last.data.copy_(self.rank_score_last.data[perm])

        # Permanently erase ranks outside new active prefix from optimization contribution.
        if changed:
            self.A.weight.data[keep:, :].zero_()
            self.B.weight.data[:, keep:].zero_()
            self.rank_score_ema.data[keep:].zero_()
            self.rank_score_last.data[keep:].zero_()
            self.active_rank_tensor.fill_(int(keep))
            self.shrink_events += 1
            self.last_shrink_step = int(step)

        top_score = float(sorted_scores[0].detach().cpu()) if sorted_scores.numel() else 0.0
        keep_cov = float(cov[keep - 1].detach().cpu()) if cov.numel() else 0.0
        return {
            "changed": bool(changed),
            "old_rank": int(old),
            "new_rank": int(keep),
            "reason": "coverage",
            "coverage": float(coverage),
            "actual_coverage": keep_cov,
            "score_sum": score_sum,
            "top_score": top_score,
            "score_entropy": self._score_entropy(scores),
            "shrink_events": int(self.shrink_events),
        }

    @staticmethod
    def _score_entropy(scores: torch.Tensor) -> float:
        s = scores.float().clamp_min(0)
        total = s.sum()
        if float(total.detach().cpu()) <= 0:
            return 0.0
        p = s / (total + 1e-12)
        ent = -(p * (p + 1e-12).log()).sum()
        return float(ent.detach().cpu())


def _get_parent_module(root: nn.Module, module_name: str):
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_gated_lora(
    model: nn.Module,
    target_names: List[str],
    rank: int,
    alpha: float,
    dropout: float,
    max_modules: int = -1,
    pdr_enable: bool = False,
    pdr_rank_floor: int = 4,
    pdr_mask_min_rank: int = 4,
    fcr_enable: bool = False,
    corrector_rank: int = 2,
    corrector_alpha: float = 8.0,
    corrector_dropout: float = 0.0,
    corrector_scale: float = 1.0,
    fcr_orthogonalize: bool = True,
    boc_enable: bool = False,
    boc_budget_ratio: float = 1.0,
    boc_eps: float = 1e-6,
) -> List[GatedLoRALinear]:
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if any(t == leaf or t in name for t in target_names):
            candidates.append(name)
    # Replace after collecting names.
    wrapped: List[GatedLoRALinear] = []
    for name in candidates:
        if max_modules > 0 and len(wrapped) >= max_modules:
            break
        parent, child_name = _get_parent_module(model, name)
        child = getattr(parent, child_name)
        if not isinstance(child, nn.Linear):
            continue
        gl = GatedLoRALinear(
            child,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            pdr_enable=pdr_enable,
            pdr_rank_floor=pdr_rank_floor,
            pdr_mask_min_rank=pdr_mask_min_rank,
            fcr_enable=fcr_enable,
            corrector_rank=corrector_rank,
            corrector_alpha=corrector_alpha,
            corrector_dropout=corrector_dropout,
            corrector_scale=corrector_scale,
            fcr_orthogonalize=fcr_orthogonalize,
            boc_enable=boc_enable,
            boc_budget_ratio=boc_budget_ratio,
            boc_eps=boc_eps,
        )
        gl.lora_name = name
        setattr(parent, child_name, gl)
        wrapped.append(gl)
    return wrapped


class FrequencyGateRouter(nn.Module):
    """
    Frequency-conditioned LoRA gate router.

    Modes:
      sigmoid_scale:
        g = gate_scale * sigmoid(router(z)), default old behavior.
        With gate_scale=2.0 and zero init, g starts at 1.

      bounded_residual:
        g = 1 + beta * tanh(router(z)).
        With zero init, g starts at 1; with beta=0.3, g is bounded in [0.7, 1.3].

      faar_mean_norm:
        Frequency-Adaptive Adapter Reallocation (FAAR).
        First compute the old strong gate:
            g0 = gate_scale * sigmoid(router(z))
        Then normalize the per-sample mean over all LoRA modules:
            g = g0 / mean(g0)
        Therefore mean(g)=1 for each sample. Frequency cues cannot globally amplify
        the whole MLLM adaptation; they can only reallocate adapter capacity across
        internal LoRA modules. This keeps the original dynamic range better than
        bounded_residual while avoiding frequency-dominant global scaling.
    """
    def __init__(
        self,
        in_dim: int,
        num_gates: int,
        hidden: int = 256,
        gate_scale: float = 2.0,
        gate_mode: str = "sigmoid_scale",
        residual_beta: float = 0.3,
        faar_clamp_min: float = 0.25,
        faar_clamp_max: float = 2.50,
        faar_renorm_after_clamp: bool = True,
    ):
        super().__init__()
        self.num_gates = int(num_gates)
        self.gate_scale = float(gate_scale)
        self.gate_mode = str(gate_mode)
        self.residual_beta = float(residual_beta)
        self.faar_clamp_min = float(faar_clamp_min)
        self.faar_clamp_max = float(faar_clamp_max)
        self.faar_renorm_after_clamp = bool(faar_renorm_after_clamp)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_gates),
        )
        # zero init: both old sigmoid gate and bounded residual gate start at neutral gate=1.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        raw = self.net(z.float())

        if self.gate_mode in {"bounded_residual", "residual_tanh", "tanh"}:
            return 1.0 + self.residual_beta * torch.tanh(raw)

        if self.gate_mode in {"sigmoid_scale", "old", "sigmoid"}:
            return torch.sigmoid(raw) * self.gate_scale

        if self.gate_mode in {"faar", "faar_mean_norm", "mean_norm", "adapter_reallocation"}:
            g0 = torch.sigmoid(raw) * self.gate_scale
            g = g0 / (g0.mean(dim=1, keepdim=True) + 1e-6)

            # Clamp prevents extremely small/large individual gates.
            if self.faar_clamp_min > 0 or self.faar_clamp_max > 0:
                lo = self.faar_clamp_min if self.faar_clamp_min > 0 else None
                hi = self.faar_clamp_max if self.faar_clamp_max > 0 else None
                g = torch.clamp(g, min=lo, max=hi)

            # Re-normalize after clamp so frequency still cannot globally amplify LoRA.
            if self.faar_renorm_after_clamp:
                g = g / (g.mean(dim=1, keepdim=True) + 1e-6)
            return g

        raise ValueError(f"Unknown freq_gate_mode={self.gate_mode}")

