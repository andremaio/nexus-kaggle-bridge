#!/usr/bin/env python3
from __future__ import annotations

import base64
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import urllib.request

TRAIN_COMMIT = 'bbe0991067b328e545baf2b41edecb8452e3a142'
TRAIN_BLOB_SHA = '5a20df29a6dcc85466686b41c005a72ababe04bf'
TRAIN_API = (
    'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
    f'kaggle/train_qwen3_1_7b_lora.py?ref={TRAIN_COMMIT}'
)


def _remove_incompatible_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_1_7B_TRAIN_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN3_1_7B_TRAIN_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_1_7B_TRAIN_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    _remove_incompatible_torchao()

    request = urllib.request.Request(
        TRAIN_API,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'nexus-qwen3-1.7b-train-bootstrap',
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if payload.get('sha') != TRAIN_BLOB_SHA:
        raise RuntimeError(
            f'trainer blob mismatch: {payload.get("sha")} != {TRAIN_BLOB_SHA}'
        )
    encoded = payload.get('content')
    if not isinstance(encoded, str) or payload.get('encoding') != 'base64':
        raise RuntimeError('GitHub trainer payload is not base64 content')
    trainer_bytes = base64.b64decode(encoded, validate=False)
    if not trainer_bytes:
        raise RuntimeError('downloaded empty trainer')
    target = Path('/kaggle/working/train_qwen3_1_7b_lora.py')
    target.write_bytes(trainer_bytes)
    print(
        'NEXUS_QWEN3_1_7B_TRAIN_BOOTSTRAP '
        f'CUDA_VISIBLE_DEVICES=0 commit={TRAIN_COMMIT} blob={TRAIN_BLOB_SHA} '
        f'bytes={len(trainer_bytes)}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
