#!/usr/bin/env python3
from __future__ import annotations

import hashlib
from pathlib import Path
import runpy
import urllib.request

EVALUATOR_COMMIT = '37c442ad8368fc3424c4622aa72893a9a5aab9d6'
EVALUATOR_BLOB = '70aa0e987391e5ccb55f8120e3adb4dc3f3d6d96'
EVALUATOR_URL = (
    'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/'
    + EVALUATOR_COMMIT
    + '/kaggle/eval_decision_policy_v1_private_v7.py'
)


def git_blob_sha1(payload: bytes) -> str:
    return hashlib.sha1(f'blob {len(payload)}\0'.encode('ascii') + payload).hexdigest()


def main() -> None:
    request = urllib.request.Request(
        EVALUATOR_URL,
        headers={'User-Agent': 'nexus-decision-v7-bootstrap'},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    actual = git_blob_sha1(payload)
    if actual != EVALUATOR_BLOB:
        raise RuntimeError(f'private V7 evaluator blob mismatch: {actual} != {EVALUATOR_BLOB}')
    target = Path('/kaggle/working/eval_decision_policy_v1_private_v7.py')
    target.write_bytes(payload)
    compile(payload.decode('utf-8'), str(target), 'exec')
    print(
        f'NEXUS_DECISION_V7_EVALUATOR_OK commit={EVALUATOR_COMMIT} blob={EVALUATOR_BLOB}',
        flush=True,
    )
    try:
        runpy.run_path(str(target), run_name='__main__')
    except SystemExit as exc:
        # Exit code 2 means the scientific gate rejected the candidate. Preserve
        # the JSON evidence and let the outer workflow fail closed from the report.
        if exc.code not in (0, None, 2):
            raise


if __name__ == '__main__':
    main()
