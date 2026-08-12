#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import runpy
import subprocess
import sys
import urllib.request
from pathlib import Path

TRAIN_COMMIT = '3ebe3b1bd59767a41c00e008b86d19377151bebc'
TRAIN_URL = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{TRAIN_COMMIT}/kaggle/nexus_train.py'


def main() -> None:
    try:
        version = importlib.metadata.version('torchao')
        print(f'NEXUS_BOOTSTRAP torchao_before={version}', flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_BOOTSTRAP torchao_before=absent', flush=True)

    # Standard FP16 LoRA does not require TorchAO. Kaggle currently ships an
    # old TorchAO build that PEFT 0.19 rejects during adapter injection.
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
        print('NEXUS_BOOTSTRAP torchao_after=absent', flush=True)

    train_path = Path('/kaggle/working/nexus_train.py')
    with urllib.request.urlopen(TRAIN_URL, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError('downloaded empty nexus_train.py')
    train_path.write_bytes(payload)
    print(f'NEXUS_BOOTSTRAP train_commit={TRAIN_COMMIT} bytes={len(payload)}', flush=True)

    runpy.run_path(str(train_path), run_name='__main__')


if __name__ == '__main__':
    main()
