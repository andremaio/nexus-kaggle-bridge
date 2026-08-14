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

SCRIPT_COMMIT = 'e9a53cfada5ea4feeafcff4ef6feccb8dabd2f5e'
BAD_DATA_COMMIT = 'e2a87ce2c74a7c3e24e4a6d651f4560947081ef1'
DATA_COMMIT = 'e2a87cebd9336ecde0c6939df30d5d6071285be2'
API = ('https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
       f'kaggle/train_qwen3_4b_instruct2507_lora.py?ref={SCRIPT_COMMIT}')


def remove_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        return
    print('NEXUS_QWEN4B_BOOTSTRAP torchao_before=' + before, flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError('torchao still installed: ' + after)


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    remove_torchao()

    req = urllib.request.Request(API, headers={'Accept':'application/vnd.github+json','User-Agent':'nexus-qwen4b-train-bootstrap'})
    with urllib.request.urlopen(req, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    encoded = payload.get('content')
    if payload.get('encoding') != 'base64' or not isinstance(encoded, str):
        raise RuntimeError('invalid GitHub training-script payload')
    source = base64.b64decode(encoded).decode('utf-8')
    old = f"DATA_COMMIT = '{BAD_DATA_COMMIT}'"
    new = f"DATA_COMMIT = '{DATA_COMMIT}'"
    if source.count(old) != 1:
        raise RuntimeError('training data commit patch invariant failed')
    source = source.replace(old, new)
    target = Path('/kaggle/working/train_qwen3_4b_instruct2507_lora.py')
    target.write_text(source, encoding='utf-8')
    compile(source, str(target), 'exec')
    print(f'NEXUS_QWEN4B_BOOTSTRAP script_commit={SCRIPT_COMMIT} data_commit={DATA_COMMIT} blob={payload.get("sha")}', flush=True)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
