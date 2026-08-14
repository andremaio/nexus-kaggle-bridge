#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import runpy
import urllib.request

PROBE_COMMIT = '3a2f9527ce32b3960a784b971b6ad58bcfbe855f'
PROBE_BLOB_SHA = '4bd29ee0d4d7a7e37d8b829bb1971bc0b02ef01f'
PROBE_API = (
    'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/'
    f'kaggle/probe_qwen3_1_7b.py?ref={PROBE_COMMIT}'
)


def main() -> None:
    # Must be set before the downloaded probe imports torch.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')

    request = urllib.request.Request(
        PROBE_API,
        headers={
            'Accept': 'application/vnd.github+json',
            'User-Agent': 'nexus-qwen3-1.7b-probe-bootstrap',
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

    target = Path('/kaggle/working/probe_qwen3_1_7b.py')
    target.write_bytes(probe_bytes)
    print(
        'NEXUS_QWEN3_1_7B_BOOTSTRAP '
        f'CUDA_VISIBLE_DEVICES=0 commit={PROBE_COMMIT} blob={PROBE_BLOB_SHA} '
        f'bytes={len(probe_bytes)}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
