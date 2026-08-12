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

CODE_COMMIT = '80e6100a1a766d5bad8b08d3e88bf1731d3b9701'
FILES = {
    'v5_common.py': 'b51c5f8b9d2aab419c5d91c45bc3705465e3191f',
    'nexus_train_v5.py': '6de2f7ad1da5adb9d13c1c58e63c5da51dcc8568',
}


def git_blob_sha(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(f'blob {len(data)}\0'.encode())
    h.update(data)
    return h.hexdigest()


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print('NEXUS_V5_BOOTSTRAP CUDA_VISIBLE_DEVICES=0', flush=True)
    try:
        print('NEXUS_V5_BOOTSTRAP torchao_before=' + importlib.metadata.version('torchao'), flush=True)
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V5_BOOTSTRAP torchao_before=absent', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'], check=False)
    try:
        version = importlib.metadata.version('torchao')
        raise RuntimeError(f'torchao still installed after cleanup: {version}')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_V5_BOOTSTRAP torchao_after=absent', flush=True)

    work = Path('/kaggle/working')
    for name, expected_blob in FILES.items():
        url = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{CODE_COMMIT}/kaggle/{name}'
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        if not payload:
            raise RuntimeError(f'empty download: {name}')
        digest = git_blob_sha(payload)
        if digest != expected_blob:
            raise RuntimeError(f'code blob mismatch for {name}: {digest}')
        (work / name).write_bytes(payload)
        print(f'NEXUS_V5_BOOTSTRAP verified={name} blob={digest} bytes={len(payload)}', flush=True)

    sys.path.insert(0, str(work))
    runpy.run_path(str(work / 'nexus_train_v5.py'), run_name='__main__')


if __name__ == '__main__':
    main()
