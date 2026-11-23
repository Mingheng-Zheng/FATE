"""Evaluate a FATE checkpoint on GAN-Test/ProGAN only."""
import json
from pathlib import Path
from torch.utils.data import DataLoader

from fcr_boc.data.datasets import Collator
from release_utils import read_rows
from train_fate import arguments, initialize, load_weights, evaluate


def main():
    a=arguments(evaluation=True)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=read_rows(a.progan_jsonl,a.image_root,progan_only=True,limit=a.max_eval_samples)
    model,processor,freq=initialize(a)
    load_weights(model,a.checkpoint)
    loader=DataLoader(rows,batch_size=a.eval_batch_size,num_workers=a.num_workers,
                      collate_fn=Collator(processor,freq,False,0.))
    result=evaluate(model,loader,a,out/'progan_predictions.csv')
    (out/'progan_metrics.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
