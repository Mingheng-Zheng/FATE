"""Stage 1: train the HH teacher; validate and select checkpoints on ProGAN only."""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fcr_boc.models.hh_teacher import HHSAFE, load_hh_checkpoint
from release_utils import metrics, append_json, save_checkpoint


class Images(Dataset):
    def __init__(self, root, size, training, limit=-1):
        root = Path(root)
        self.rows = [(p, label) for label, folder in enumerate(('0_real', '1_fake'))
                     for p in sorted(root.rglob('*'))
                     if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.webp') and folder in p.parts]
        if limit > 0:
            self.rows = [r for r in self.rows if r[1] == 0][:limit//2] + [r for r in self.rows if r[1] == 1][:limit-limit//2]
        if {label for _, label in self.rows} != {0,1}:
            raise ValueError(f'Both classes must exist under {root}')
        ops = [transforms.RandomCrop(size, pad_if_needed=True), transforms.RandomHorizontalFlip()] if training else [transforms.CenterCrop(size)]
        self.tf = transforms.Compose(ops + [transforms.ToTensor()])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        p, label = self.rows[i]
        with Image.open(p) as image:
            return self.tf(image.convert('RGB')), label


def optimizer_groups(model, lr, wd):
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Keep normalization and bias tensors out of weight decay.
        section = 'head' if name.startswith('fc.') else 'encoder'
        decay = p.ndim > 1 and not name.endswith('bias')
        groups.setdefault((section, decay), []).append(p)
    return [dict(params=ps, lr=lr, weight_decay=wd if decay else 0., name=f'{section}_{"decay" if decay else "no_decay"}')
            for (section,decay),ps in groups.items()]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    for x,y in loader:
        logits = model(x.to(device))['logits'].float()
        labels.extend(y.tolist())
        scores.extend((logits[:,1]-logits[:,0]).cpu().tolist())
    return metrics(labels, scores)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-root', required=True)
    p.add_argument('--progan-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--init-checkpoint')
    p.add_argument('--resume')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--eval-every', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--size', type=int, default=256)
    p.add_argument('--lr', type=float, default=.001875)
    p.add_argument('--min-lr', type=float, default=1e-6)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--label-smoothing', type=float, default=.1)
    p.add_argument('--fake-margin', type=float, default=.5)
    p.add_argument('--margin-weight', type=float, default=.2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max-train-samples', type=int, default=-1)
    p.add_argument('--max-eval-samples', type=int, default=-1)
    a = p.parse_args()
    if Path(a.progan_root).name.lower() != 'progan':
        p.error('--progan-root must name the GAN-Test/progan directory')
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out/'args.json').write_text(json.dumps(vars(a),indent=2))
    model = HHSAFE().to(device)
    if a.init_checkpoint:
        load_hh_checkpoint(model, a.init_checkpoint)
    opt = torch.optim.AdamW(optimizer_groups(model,a.lr,a.weight_decay),eps=1e-8)
    train = DataLoader(Images(a.train_root,a.size,True,a.max_train_samples),batch_size=a.batch_size,shuffle=True,num_workers=a.workers)
    val = DataLoader(Images(a.progan_root,a.size,False,a.max_eval_samples),batch_size=a.eval_batch_size,num_workers=a.workers)
    if len(train) == 0:
        raise ValueError('No training batches')
    best, start = (-1.,-1.), 0
    if a.resume:
        ck = torch.load(a.resume,map_location='cpu',weights_only=False)
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        start=ck['epoch'];best=tuple(ck['best'])
    for epoch in range(start,a.epochs):
        model.train(); total=0.; count=0
        for i,(x,y) in enumerate(train):
            # Linear warm-up for one epoch, then cosine decay.
            progress=epoch+i/len(train)
            lr=a.lr*min(1.,(progress+1/len(train))) if progress<1 else a.min_lr+(a.lr-a.min_lr)*.5*(1+math.cos(math.pi*(progress-1)/max(1,a.epochs-1)))
            for g in opt.param_groups:g['lr']=lr
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            logits=model(x)['logits']
            loss=F.cross_entropy(logits,y,label_smoothing=a.label_smoothing)
            if (y==1).any():loss=loss+a.margin_weight*F.relu(a.fake_margin-(logits[y==1,1]-logits[y==1,0])).mean()
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite HH loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(y);count+=len(y)
            if i%100==0:print(f'epoch={epoch+1} batch={i}/{len(train)} loss={float(loss.detach()):.5f}',flush=True)
        result=None
        if (epoch+1)%a.eval_every==0 or epoch+1==a.epochs:
            result=evaluate(model,val,device)
            append_json(out/'validation.jsonl',dict(epoch=epoch+1,dataset='GAN-Test/ProGAN',**result))
        state=dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch+1,args=vars(a),best=best)
        if result is not None and (result['balanced_acc'],result['ap'])>best:
            best=(result['balanced_acc'],result['ap']);state['best']=best;state['metric']=result
            save_checkpoint(out/'checkpoint-best.pth',state)
        state['best']=best
        save_checkpoint(out/'checkpoint-last.pth',state)
        if result is not None:save_checkpoint(out/f'checkpoint-epoch{epoch+1}.pth',state)
        append_json(out/'train.jsonl',dict(epoch=epoch+1,loss=total/count,lr=lr,best=best))


if __name__ == '__main__':
    main()
