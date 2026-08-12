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

TRAIN_COMMIT = 'abeb3156c81dc76ff144121e976b7cae6fb018e7'
TRAIN_SHA256 = 'da8aad5e0efb27119ca7dfb503214e8661a9777535d3d4dab931898a96c262a4'
TRAIN_URL = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{TRAIN_COMMIT}/kaggle/nexus_train_v4.py'


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('NEXUS_V4_BOOTSTRAP CUDA_VISIBLE_DEVICES=0', flush=True)

    try:
        version = importlib.metadata.version('torchao')
        print(f'NEXUS_V4_BOOTSTRAP torchao_before={version}', flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V4_BOOTSTRAP torchao_before=absent', flush=True)

    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    try:
        version = importlib.metadata.version('torchao')
        raise RuntimeError(f'torchao still installed after cleanup: {version}')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V4_BOOTSTRAP torchao_after=absent', flush=True)

    train_path = Path('/kaggle/working/nexus_train_v4.py')
    with urllib.request.urlopen(TRAIN_URL, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError('downloaded empty nexus_train_v4.py')
    digest = hashlib.sha256(payload).hexdigest()
    if digest != TRAIN_SHA256:
        raise RuntimeError(f'trainer hash mismatch: {digest}')
    train_path.write_bytes(payload)
    print(f'NEXUS_V4_BOOTSTRAP train_commit={TRAIN_COMMIT} sha256={digest} bytes={len(payload)}', flush=True)
    runpy.run_path(str(train_path), run_name='__main__')


if __name__ == '__main__':
    main()
