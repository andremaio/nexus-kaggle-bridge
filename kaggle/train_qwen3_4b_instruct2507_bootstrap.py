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

SCRIPT_COMMIT = 'd322cea46bf710f832d0c7d3a5463acc9d5f7452'
SCRIPT_BLOB_SHA = '6d6453e1bfa3a8b68daad9db005250126751a6bc'
SCRIPT_API = (
    'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
    f'kaggle/train_qwen3_4b_instruct2507_lora.py?ref={SCRIPT_COMMIT}'
)
BAD_DATA_COMMIT = 'e2a87ce2c74a7c3e24e4a6d651f4560947081ef1'
DATA_COMMIT = 'e2a87cebd9336ecde0c6939df30d5d6071285be2'
EXPECTED_TRAIN_BLOBS = {
    'seed_sft_v1.jsonl': 'b9baa9ca58c241c47ab41cb59eb4ece312991d37',
    'seed_sft_v2.jsonl': '7b84dd2cce420eabbe903fc26258f2c2db7774db',
    'seed_sft_v4.jsonl': '9624614dab1200842aa27f5989fb6c6b38fdd31f',
    'seed_sft_v5.jsonl': '764f37ab8a4f1889ab49cc2966766156227ff004',
}


def remove_incompatible_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_TRAIN_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN4B_TRAIN_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_TRAIN_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def github_content(path: str, ref: str) -> dict:
    url = f'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/{path}?ref={ref}'
    request = urllib.request.Request(
        url,
        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'nexus-qwen4b-train-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if not isinstance(payload, dict):
        raise RuntimeError('unexpected GitHub payload')
    return payload


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache-qwen4b-lora'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache-qwen4b-lora'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    remove_incompatible_torchao()

    for name, expected_blob in EXPECTED_TRAIN_BLOBS.items():
        meta = github_content(f'training/{name}', DATA_COMMIT)
        if meta.get('sha') != expected_blob:
            raise RuntimeError(f'training blob mismatch for {name}: {meta.get("sha")} != {expected_blob}')
        print(f'NEXUS_QWEN4B_TRAIN_INPUT_OK {name} blob={expected_blob}', flush=True)

    request = urllib.request.Request(
        SCRIPT_API,
        headers={'Accept': 'application/vnd.github+json', 'User-Agent': 'nexus-qwen4b-train-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode('utf-8'))
    if payload.get('sha') != SCRIPT_BLOB_SHA:
        raise RuntimeError(f'trainer blob mismatch: {payload.get("sha")} != {SCRIPT_BLOB_SHA}')
    if payload.get('encoding') != 'base64' or not isinstance(payload.get('content'), str):
        raise RuntimeError('unexpected trainer payload')
    source = base64.b64decode(payload['content'], validate=False).decode('utf-8')
    old = f"DATA_COMMIT = '{BAD_DATA_COMMIT}'"
    new = f"DATA_COMMIT = '{DATA_COMMIT}'"
    if source.count(old) != 1:
        raise RuntimeError('training data commit patch invariant failed')
    source = source.replace(old, new, 1)
    if "num_train_epochs=1.0" not in source or "learning_rate=2e-5" not in source or "r=8" not in source:
        raise RuntimeError('fixed conservative hyperparameter contract missing')
    if "automatic_promotion_authorized':False" not in source or "automatic_activation_authorized':False" not in source:
        raise RuntimeError('authority contract missing')
    target = Path('/kaggle/working/train_qwen3_4b_instruct2507_lora.py')
    target.write_text(source, encoding='utf-8')
    compile(source, str(target), 'exec')
    print(
        f'NEXUS_QWEN4B_TRAIN_BOOTSTRAP script={SCRIPT_COMMIT}:{SCRIPT_BLOB_SHA} '
        f'data={DATA_COMMIT} bytes={len(source.encode("utf-8"))}', flush=True
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
