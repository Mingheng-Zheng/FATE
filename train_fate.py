"""Stage 2 FATE adaptation and ProGAN-only checkpoint selection."""
import argparse
import gc
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from fcr_boc.common import torch_utils as project
from fcr_boc.data.datasets import Collator
from fcr_boc.data.transforms import DeterministicFrequencyTransform
from fcr_boc.models.fcr_boc_qwen import HHGuidedQwenLoRAClassifier
from fcr_boc.training.optim import build_optimizer
from fcr_boc.training.step import train_step
from release_utils import read_rows, metrics, append_json, save_checkpoint, write_predictions


class MicroCollator:
    def __init__(self, collator, size):
        self.collator, self.size = collator, size

    def __call__(self, rows):
        return [self.collator(rows[i:i+self.size]) for i in range(0,len(rows),self.size)]


def arguments(evaluation=False):
    p = argparse.ArgumentParser(description='FATE: GAN-Test/ProGAN evaluation only')
    p.add_argument('--config', default=str(Path(__file__).parent/'configs/fate.json'))
    p.add_argument('--model-path', required=True)
    p.add_argument('--hh-checkpoint', required=True)
    p.add_argument('--progan-jsonl', required=True)
    p.add_argument('--image-root')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-eval-samples', type=int, default=-1)
    if evaluation:
        p.add_argument('--checkpoint', required=True)
    else:
        p.add_argument('--train-jsonl', required=True)
        p.add_argument('--resume')
        p.add_argument('--max-train-samples', type=int, default=-1)
        p.add_argument('--max-steps', type=int, default=-1)
        p.add_argument('--epochs', type=int)
        p.add_argument('--batch-size', type=int)
        p.add_argument('--micro-batch-size', type=int)
        p.add_argument('--eval-every-steps', type=int)
        p.add_argument('--save-every-steps', type=int)
    cli = vars(p.parse_args())
    cfg = json.loads(Path(cli['config']).read_text())
    cfg.update({k.replace('-','_'):v for k,v in cli.items() if v is not None})
    cfg.setdefault('image_root', None)
    cfg.setdefault('resume', None)
    cfg['hh_ckpt'] = cfg.pop('hh_checkpoint')
    a = SimpleNamespace(**cfg)
    if a.batch_size < 1 or a.micro_batch_size < 1 or a.batch_size % a.micro_batch_size:
        p.error('batch_size must be a positive multiple of micro_batch_size')
    return a


def initialize(a):
    project.set_seed(a.seed)
    processor = AutoProcessor.from_pretrained(a.model_path,min_pixels=a.min_pixels,max_pixels=a.max_pixels)
    processor.tokenizer.padding_side = 'left'
    model = HHGuidedQwenLoRAClassifier(a)
    # Classification uses hidden states, not autoregressive vocabulary logits.
    model.qwen.lm_head = torch.nn.Identity()
    freq = DeterministicFrequencyTransform(a.freq_size,a.freq_view)
    return model,processor,freq


def load_weights(model, path):
    obj = torch.load(path,map_location='cpu',weights_only=False)
    state = obj.get('model',obj)
    expected = {name for name,p in model.named_parameters() if p.requires_grad}
    missing = expected-set(state)
    unexpected = set(state)-set(model.state_dict())
    if missing or unexpected:
        raise ValueError(f'Checkpoint incompatibility: missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]}')
    model.load_state_dict(state,strict=False)
    return obj


@torch.no_grad()
def evaluate(model, loader, a, output):
    model.eval()
    labels, scores, rows = [], [], []
    for batch in loader:
        with project.autocast_cuda(a.use_amp,dtype=project.amp_dtype_from_name(a.amp_dtype)):
            logits,_,_,_ = model(batch['qwen_inputs'],batch['freq_x'],None)
        margin = (logits.float()[:,1]-logits.float()[:,0]).cpu().tolist()
        target = batch['labels'].tolist()
        labels.extend(target);scores.extend(margin)
        rows.extend(dict(image_path=p,label='fake' if y else 'real',fake_margin=m)
                    for p,y,m in zip(batch['paths'],target,margin))
    result = dict(dataset='GAN-Test/ProGAN',**metrics(labels,scores))
    write_predictions(output,rows)
    return result


def snapshot(model,a,step,epoch,opt,best):
    return dict(model={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad},
                args=vars(a),step=step,epoch=epoch,optimizer=opt.state_dict(),best=best,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(model.qwen_device()),
                python_rng=random.getstate(),numpy_rng=np.random.get_state())


def main():
    a = arguments()
    out = Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a),indent=2))
    train_rows = read_rows(a.train_jsonl,a.image_root,limit=a.max_train_samples)
    val_rows = read_rows(a.progan_jsonl,a.image_root,progan_only=True,limit=a.max_eval_samples)
    overlap = {r['image_path'] for r in train_rows}&{r['image_path'] for r in val_rows}
    if overlap:
        raise ValueError('Training and validation manifests overlap')
    model,processor,freq = initialize(a)
    val = DataLoader(val_rows,batch_size=a.eval_batch_size,num_workers=a.num_workers,
                     collate_fn=Collator(processor,freq,False,0.))
    opt = build_optimizer(model,a)
    scaler = project.make_scaler(a.use_amp,a.amp_dtype)
    dtype = project.amp_dtype_from_name(a.amp_dtype)
    steps_per_epoch = len(train_rows)//a.batch_size
    if steps_per_epoch < 1:
        raise ValueError('Training set is smaller than batch_size')
    target = a.epochs*steps_per_epoch if a.max_steps < 0 else min(a.max_steps,a.epochs*steps_per_epoch)
    step,best = 0,(-1.,-1.)
    if a.resume:
        saved=load_weights(model,a.resume)
        opt.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler'])
        step=saved['step'];best=tuple(saved['best'])
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'],model.qwen_device())
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        del saved
    base_rates=[g['lr'] for g in opt.param_groups]
    if a.resume:
        # Reconstruct base rates rather than compounding epoch-two LR reduction.
        rates=dict(frequency_anchor_lora=a.lora_lr,teacher_free_corrector=a.fcr_corrector_lr,freq_router=a.router_lr,cls_head=a.head_lr)
        base_rates=[rates[g['name']] for g in opt.param_groups]
    for epoch in range(step//steps_per_epoch,a.epochs):
        generator=torch.Generator().manual_seed(a.seed+epoch)
        train=DataLoader(train_rows,batch_size=a.batch_size,shuffle=True,drop_last=True,
                         generator=generator,num_workers=a.num_workers,
                         collate_fn=MicroCollator(Collator(processor,freq,True,a.text_dropout_prob),a.micro_batch_size))
        for g,rate in zip(opt.param_groups,base_rates):g['lr']=rate*(.5 if epoch>=1 else 1.)
        model.train()
        for batch_i,batch in enumerate(train):
            if epoch*steps_per_epoch+batch_i < step:
                continue
            if step>=target:
                return
            step+=1
            before=time.monotonic()
            la,lf,correct,details=train_step(project,a,model,opt,scaler,dtype,batch,step,True)
            append_json(out/'training.jsonl',dict(step=step,epoch=epoch+1,anchor_loss=la,full_loss=lf,
                        acc=100*correct/a.batch_size,seconds=time.monotonic()-before,
                        learning_rates={g['name']:g['lr'] for g in opt.param_groups},diagnostics=details))
            final=step==target or step%steps_per_epoch==0
            need_eval=final or step%a.eval_every_steps==0
            need_save=need_eval or step%a.save_every_steps==0
            opt.zero_grad(set_to_none=True)
            if need_eval:
                gc.collect();torch.cuda.empty_cache()
                result=evaluate(model,val,a,out/f'progan-step{step}.csv')
                append_json(out/'validation.jsonl',dict(step=step,epoch=epoch+1,**result))
                improved=(result['balanced_acc'],result['ap'])>best
                if improved:best=(result['balanced_acc'],result['ap'])
                model.train()
            if need_save:
                saved=snapshot(model,a,step,epoch+1,opt,best);saved['scaler']=scaler.state_dict()
                save_checkpoint(out/'resume-last.pth',saved)
                # Inference weights deliberately omit frozen Qwen/HH weights and optimizer state.
                weights={k:saved[k] for k in ('model','args','step','epoch','best')}
                save_checkpoint(out/f'checkpoint-step{step}.pth',weights)
                if need_eval and improved:save_checkpoint(out/'checkpoint-best.pth',weights)
                del saved,weights
            if step>=target:
                return


if __name__=='__main__':
    main()
