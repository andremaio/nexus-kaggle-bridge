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

SCRIPT_COMMIT = '569b3b7f728884106e7943d8d72a2c782185ae47'
SCRIPT_BLOB = '4802f38f4e9dff8eed5f59aebf73e69eed5ee85f'
SCRIPT_URL = 'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/' + SCRIPT_COMMIT + '/kaggle/eval_qwen3_4b_instruct2507_lora_v3.py'


def blob_sha(payload: bytes) -> str:
    return hashlib.sha1(f'blob {len(payload)}\0'.encode() + payload).hexdigest()


def remove_torchao() -> None:
    try: before=importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_V3_EVAL torchao=absent',flush=True); return
    print(f'NEXUS_QWEN4B_V3_EVAL torchao_before={before}',flush=True)
    subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=False)
    try: after=importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError: return
    raise RuntimeError(f'torchao still installed: {after}')


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['HF_HOME']='/kaggle/temp/hf-cache'
    os.environ['XDG_CACHE_HOME']='/kaggle/temp/cache'
    os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    remove_torchao()
    import torch
    names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(f'NEXUS_QWEN4B_V3_EVAL_CUDA torch={torch.__version__} available={torch.cuda.is_available()} count={torch.cuda.device_count()} devices={names}',flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('exactly one CUDA GPU required')
    req=urllib.request.Request(SCRIPT_URL,headers={'User-Agent':'nexus-qwen4b-v3-eval-bootstrap'})
    with urllib.request.urlopen(req,timeout=90) as response: payload=response.read()
    actual=blob_sha(payload)
    if actual!=SCRIPT_BLOB: raise RuntimeError(f'eval script blob mismatch: {actual} != {SCRIPT_BLOB}')
    target=Path('/kaggle/working/eval_qwen3_4b_instruct2507_lora_v3.py')
    target.write_bytes(payload); compile(payload.decode('utf-8'),str(target),'exec')
    print(f'NEXUS_QWEN4B_V3_EVAL_SCRIPT_OK commit={SCRIPT_COMMIT} blob={SCRIPT_BLOB}',flush=True)
    runpy.run_path(str(target),run_name='__main__')


if __name__=='__main__': main()
