"""Verify and extract FORTE tar shards under one explicit image root."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    source=Path(a.dataset_dir).resolve();root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    for record in json.loads((source/'image_shards.json').read_text()):
        archive=(source/record['file']).resolve()
        if not archive.is_relative_to(source):raise ValueError('Invalid archive path')
        with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=record['sha256']:raise ValueError(f'Checksum mismatch: {archive.name}')
        with tarfile.open(archive,'r') as tar:
            for member in tar:
                target=(root/member.name).resolve()
                if not member.isfile() or not target.is_relative_to(root):raise ValueError('Unsafe tar entry')
                if target.exists():
                    if target.is_file() and target.stat().st_size==member.size:continue
                    raise FileExistsError(target)
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        print(f'Extracted {archive.name}',flush=True)


if __name__=='__main__':
    main()
