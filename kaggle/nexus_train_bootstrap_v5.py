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

TRAIN_COMMIT = '6ec90236df66707df1c240b8cbee93a1ab510ec1'
TRAIN_GIT_BLOB = 'd50b1100d4ec05e7c8fccde65bde7bc3b0a15b98'
TRAIN_URL = (
    f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/'
    f'{TRAIN_COMMIT}/kaggle/nexus_train_v5.py'
)


def _git_blob_sha1(data: bytes) -> str:
    header = f'blob {len(data)}\0'.encode('ascii')
    return hashlib.sha1(header + data).hexdigest()


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('NEXUS_V5_BOOTSTRAP CUDA_VISIBLE_DEVICES=0', flush=True)

    try:
        version = importlib.metadata.version('torchao')
        print(f'NEXUS_V5_BOOTSTRAP torchao_before={version}', flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V5_BOOTSTRAP torchao_before=absent', flush=True)

    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    train_path = Path('/kaggle/working/nexus_train_v5.py')
    with urllib.request.urlopen(TRAIN_URL, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError('downloaded empty nexus_train_v5.py')
    digest = _git_blob_sha1(payload)
    if digest != TRAIN_GIT_BLOB:
        raise RuntimeError(f'v5 trainer git-blob mismatch: {digest}')
    train_path.write_bytes(payload)
    print(
        f'NEXUS_V5_BOOTSTRAP train_commit={TRAIN_COMMIT} '
        f'git_blob={digest} bytes={len(payload)}',
        flush=True,
    )
    runpy.run_path(str(train_path), run_name='__main__')


if __name__ == '__main__':
    main()
