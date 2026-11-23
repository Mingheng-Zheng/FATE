import warnings
from collections import defaultdict
from statistics import mean, stdev
import torch

def accumulate_microbatch(project, args, model, scaler, dtype, batch, step, collect, divisor):
    device = model.qwen_device()
    labels = batch['labels'].to(device, non_blocking=True)
    ctx = model._prepare_fcr_train_context(batch['qwen_inputs'], batch['freq_x'], labels)
    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state(device)
    with project.autocast_cuda(args.use_amp, dtype=dtype):
        loss_a, details_a, anchor_logits = model.fcr_anchor_stage(ctx, collect_details=collect)
    project.require_finite_scalar('loss_anchor_stage', loss_a, step)
    scaler.scale(loss_a / divisor).backward()
    anchor_loss = float(loss_a.detach())
    del loss_a
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state(gpu_rng, device)
    with project.autocast_cuda(args.use_amp, dtype=dtype), warnings.catch_warnings():
        if collect and args.micro_batch_size == 1:
            # The native singleton diagnostic is recomputed across the effective batch.
            warnings.filterwarnings('ignore', message=r'std\(\): degrees of freedom is <= 0')
        loss_f, details_f, logits = model.fcr_full_stage(
            ctx, collect_details=collect, anchor_logits_for_log=anchor_logits)
    project.require_finite_scalar('loss_full_stage', loss_f, step)
    scaler.scale(loss_f / divisor).backward()
    full_loss = float(loss_f.detach())
    del loss_f, ctx, anchor_logits
    model.clear_fcr_runtime_cache()
    return anchor_loss, full_loss, int((logits.detach().argmax(1) == labels).sum()), details_a | details_f

def train_step(project, args, model, opt, scaler, dtype, microbatches, step, collect):
    assert len(microbatches) == args.batch_size // args.micro_batch_size
    assert all(len(batch['labels']) == args.micro_batch_size for batch in microbatches)
    opt.zero_grad(set_to_none=True)
    anchor_losses, full_losses = [], []
    correct = 0
    details = defaultdict(list)
    for batch in microbatches:
        a, f, hits, diagnostics = accumulate_microbatch(
            project, args, model, scaler, dtype, batch, step, collect, len(microbatches))
        anchor_losses.append(a)
        full_losses.append(f)
        correct += hits
        for name, value in diagnostics.items():
            details[name].append(value)
    scaler.unscale_(opt)
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], args.grad_clip)
    project.require_finite_grad_norm(norm, step)
    scaler.step(opt)
    scaler.update()
    if collect and args.micro_batch_size == 1:
        details['gate_sample_mean_std'] = [stdev(details['gate_mean'])]
    return mean(anchor_losses), mean(full_losses), correct, {k: mean(v) for k, v in details.items()}
