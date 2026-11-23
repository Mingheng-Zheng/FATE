"""Resolve released relative image paths against a local dataset directory."""
import argparse
import gzip
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotations',required=True)
    p.add_argument('--image-root',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--progan-only',action='store_true')
    a=p.parse_args()
    root=Path(a.image_root).resolve()
    opener=gzip.open if a.annotations.endswith('.gz') else open
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with opener(a.annotations,'rt',encoding='utf-8') as src, output.open('w',encoding='utf-8') as dst:
        for line in src:
            row=json.loads(line)
            if a.progan_only and row.get('generator','').lower()!='progan':continue
            image=(root/row['image_path']).resolve()
            if not image.is_relative_to(root):raise ValueError('Image outside root')
            if not image.is_file():raise FileNotFoundError(image)
            row['image_path']=str(image)
            dst.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
    if not count:raise ValueError('No records exported')
    print(f'Prepared {count} records: {output}')


if __name__=='__main__':
    main()
