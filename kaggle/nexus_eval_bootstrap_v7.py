#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import runpy
import subprocess
import sys
import urllib.request

CODE_COMMIT = '1c5d93e7c5d289ef62030527e77d845215647fd5'
FILE = 'nexus_eval_v7_17b.py'
EXPECTED_BLOB = '9f9a7a8ee4e423fb96e9fde15a8989fafa1f5f15'


def git_blob_sha(data: bytes) -> str:
    h = hashlib.sha1(); h.update(f'blob {len(data)}\0'.encode()); h.update(data); return h.hexdigest()


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    try:
        print('V7_BOOTSTRAP torchao_before=' + importlib.metadata.version('torchao'), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('V7_BOOTSTRAP torchao_before=absent', flush=True)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=False)
    subprocess.check_call([sys.executable,'-m','pip','install','--disable-pip-version-check','-q','transformers==5.14.1','peft==0.19.1','accelerate==1.14.0'])
    url=f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{CODE_COMMIT}/kaggle/{FILE}'
    with urllib.request.urlopen(url,timeout=60) as r: payload=r.read()
    if git_blob_sha(payload)!=EXPECTED_BLOB: raise RuntimeError('v7 evaluator blob mismatch')
    p=Path('/kaggle/working')/FILE; p.write_bytes(payload)
    print(f'V7_BOOTSTRAP verified={FILE} blob={EXPECTED_BLOB} bytes={len(payload)}',flush=True)
    runpy.run_path(str(p),run_name='__main__')

if __name__=='__main__': main()
