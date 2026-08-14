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

PROBE_COMMIT = '59fe1fc89a95b769b7e982a6b1f570aea5b90d4f'
PROBE_BLOB_SHA = 'dee14013ce41ad8baa0032004c4761fd8b6239e3'
PROBE_API = (
    'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
    f'kaggle/probe_qwen3_4b.py?ref={PROBE_COMMIT}'
)


def _remove_incompatible_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_4B_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN3_4B_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_4B_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    _remove_incompatible_torchao()

    request = urllib.request.Request(
        PROBE_API,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'nexus-qwen3-4b-probe-bootstrap',
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if payload.get('sha') != PROBE_BLOB_SHA:
        raise RuntimeError(
            f'probe blob mismatch: {payload.get("sha")} != {PROBE_BLOB_SHA}'
        )
    encoded = payload.get('content')
    if not isinstance(encoded, str) or payload.get('encoding') != 'base64':
        raise RuntimeError('GitHub probe payload is not base64 content')
    probe_bytes = base64.b64decode(encoded, validate=False)
    if not probe_bytes:
        raise RuntimeError('downloaded empty probe script')

    target = Path('/kaggle/working/probe_qwen3_4b.py')
    target.write_bytes(probe_bytes)
    print(
        'NEXUS_QWEN3_4B_BOOTSTRAP '
        f'CUDA_VISIBLE_DEVICES=0 HF_HOME={os.environ["HF_HOME"]} '
        f'commit={PROBE_COMMIT} blob={PROBE_BLOB_SHA} bytes={len(probe_bytes)}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
