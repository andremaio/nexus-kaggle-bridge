#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import runpy
import subprocess
import sys


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

    runpy.run_path('nexus_train.py', run_name='__main__')


if __name__ == '__main__':
    main()
