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

SCRIPT_COMMIT = '21f40a2171278ad026d042a45759d87969c4b177'
SCRIPT_BLOB = '74be780ec921417915aeb40c7468f0fd1660ac5c'
SCRIPT_URL = (
    'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/'
    f'{SCRIPT_COMMIT}/kaggle/train_qwen3_4b_instruct2507_lora_v2.py'
)


def _git_blob_sha1(payload: bytes) -> str:
    header = f'blob {len(payload)}\0'.encode('ascii')
    return hashlib.sha1(header + payload).hexdigest()


def _remove_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_V2_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN4B_V2_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_V2_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def _materialize_training_script() -> Path:
    request = urllib.request.Request(
        SCRIPT_URL,
        headers={'User-Agent':'nexus-qwen4b-v2-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError('downloaded empty pinned v2 training script')
    actual = _git_blob_sha1(payload)
    if actual != SCRIPT_BLOB:
        raise RuntimeError(
            f'pinned training script blob mismatch: {actual} != {SCRIPT_BLOB}'
        )
    target = Path('/kaggle/working/train_qwen3_4b_instruct2507_lora_v2.py')
    target.write_bytes(payload)
    compile(payload.decode('utf-8'), str(target), 'exec')
    print(
        'NEXUS_QWEN4B_V2_SCRIPT_OK '
        f'commit={SCRIPT_COMMIT} blob={SCRIPT_BLOB} bytes={len(payload)}',
        flush=True,
    )
    return target


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

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    names = []
    for index in range(count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except Exception as exc:
            names.append(f'ERROR:{type(exc).__name__}')
    print(
        'NEXUS_QWEN4B_V2_CUDA_DIAGNOSTIC '
        f'torch={torch.__version__} visible={visible!r} '
        f'available={available} count={count} devices={names}',
        flush=True,
    )
    if not available or count != 1:
        raise RuntimeError(
            f'exactly one CUDA GPU required after bootstrap: '
            f'available={available} count={count} visible={visible!r}'
        )

    target = _materialize_training_script()
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
