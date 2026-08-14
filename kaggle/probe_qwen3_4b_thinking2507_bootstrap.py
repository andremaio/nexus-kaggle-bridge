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

PROBE_COMMIT = '54f31547ca3750b2d22715ffc6e5a4c7c1c91e7b'
PROBE_BLOB_SHA = '33c5c2f23acf065b3cfae417785a10764774378a'
PROBE_API = (
    'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
    f'kaggle/probe_qwen3_4b_thinking2507.py?ref={PROBE_COMMIT}'
)


def remove_incompatible_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_THINKING2507_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_THINKING2507_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_THINKING2507_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache-thinking2507'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache-thinking2507'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    remove_incompatible_torchao()
    request = urllib.request.Request(
        PROBE_API,
        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'nexus-thinking2507-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if payload.get('sha') != PROBE_BLOB_SHA:
        raise RuntimeError('thinking2507 probe blob mismatch')
    if payload.get('encoding') != 'base64' or not isinstance(payload.get('content'), str):
        raise RuntimeError('unexpected GitHub probe payload')
    probe_bytes = base64.b64decode(payload['content'], validate=False)
    if not probe_bytes:
        raise RuntimeError('empty thinking2507 probe')
    target = Path('/kaggle/working/probe_qwen3_4b_thinking2507.py')
    target.write_bytes(probe_bytes)
    print(
        f'NEXUS_THINKING2507_BOOTSTRAP commit={PROBE_COMMIT} blob={PROBE_BLOB_SHA} bytes={len(probe_bytes)}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
