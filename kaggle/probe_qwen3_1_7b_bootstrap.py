#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import runpy


def main() -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    target = Path(__file__).with_name('probe_qwen3_1_7b.py')
    if not target.is_file():
        raise RuntimeError(f'missing probe script: {target}')
    print('NEXUS_QWEN3_1_7B_BOOTSTRAP CUDA_VISIBLE_DEVICES=0', flush=True)
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
