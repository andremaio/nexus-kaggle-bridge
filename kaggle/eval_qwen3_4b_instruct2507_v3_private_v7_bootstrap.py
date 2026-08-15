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

SCRIPT_COMMIT = '6b12acdc48bfdf31d89eca8a4ea6c16365574c6b'
SCRIPT_BLOB = '0a8ed965a9526597b1da23b614c5f2e007a79501'
SCRIPT_URL = (
    'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/'
    + SCRIPT_COMMIT
    + '/kaggle/eval_qwen3_4b_instruct2507_lora_v3_private_v7.py'
)


def _blob_sha(payload: bytes) -> str:
    return hashlib.sha1(f'blob {len(payload)}\0'.encode('ascii') + payload).hexdigest()


def _remove_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V7_PRIVATE_EVAL torchao=absent', flush=True)
        return
    print(f'NEXUS_V7_PRIVATE_EVAL torchao_before={before}', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        return
    raise RuntimeError(f'torchao still installed: {after}')


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    _remove_torchao()

    import torch
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print(
        'NEXUS_V7_PRIVATE_EVAL_CUDA '
        f'torch={torch.__version__} available={torch.cuda.is_available()} '
        f'count={torch.cuda.device_count()} devices={names}',
        flush=True,
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA GPU required')

    request = urllib.request.Request(
        SCRIPT_URL,
        headers={'User-Agent': 'nexus-v7-private-eval-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    actual = _blob_sha(payload)
    if actual != SCRIPT_BLOB:
        raise RuntimeError(f'private evaluator blob mismatch: {actual} != {SCRIPT_BLOB}')
    target = Path('/kaggle/working/eval_qwen3_4b_instruct2507_lora_v3_private_v7.py')
    target.write_bytes(payload)
    compile(payload.decode('utf-8'), str(target), 'exec')
    print(
        f'NEXUS_V7_PRIVATE_EVAL_SCRIPT_OK commit={SCRIPT_COMMIT} blob={SCRIPT_BLOB}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
