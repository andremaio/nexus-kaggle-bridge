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

PROBE_URL = (
    'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/'
    'main/kaggle/probe_qwen3_4b_decisions_v2.py'
)


def _remove_incompatible_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_4B_DECISION_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN3_4B_DECISION_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN3_4B_DECISION_BOOTSTRAP torchao_after=absent', flush=True)
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
        PROBE_URL,
        headers={'User-Agent': 'nexus-qwen3-4b-decision-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError('downloaded empty decision probe')
    digest = hashlib.sha256(payload).hexdigest()
    target = Path('/kaggle/working/probe_qwen3_4b_decisions_v2.py')
    target.write_bytes(payload)
    print(
        'NEXUS_QWEN3_4B_DECISION_BOOTSTRAP '
        f'CUDA_VISIBLE_DEVICES=0 HF_HOME={os.environ["HF_HOME"]} '
        f'script_sha256={digest} bytes={len(payload)}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
