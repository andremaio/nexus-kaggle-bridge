#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import os
from pathlib import Path
import runpy
import subprocess
import sys


def _remove_torchao() -> None:
    try:
        before = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_V2_BOOTSTRAP torchao_before=absent', flush=True)
        return
    print(f'NEXUS_QWEN4B_V2_BOOTSTRAP torchao_before={before}', flush=True)
    subprocess.run(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao'],
        check=False,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        after = importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:
        print('NEXUS_QWEN4B_V2_BOOTSTRAP torchao_after=absent', flush=True)
        return
    raise RuntimeError(f'torchao still installed after cleanup: {after}')


def main() -> None:
    # Must be set before torch is imported. Kaggle occasionally exposes the T4
    # only after the accelerator allocation while inheriting an empty/default
    # visibility value in the script environment.
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['HF_HOME'] = '/kaggle/temp/hf-cache'
    os.environ['XDG_CACHE_HOME'] = '/kaggle/temp/cache'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('DO_NOT_TRACK', '1')
    os.environ.setdefault('WANDB_DISABLED', 'true')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

    _remove_torchao()

    import torch

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    names = []
    for index in range(count):
        try:
            names.append(torch.cuda.get_device_name(index))
        except Exception as exc:
            names.append(f'ERROR:{type(exc).__name__}')
    print(
        'NEXUS_QWEN4B_V2_CUDA_DIAGNOSTIC '
        f'torch={torch.__version__} visible={visible!r} '
        f'available={available} count={count} devices={names}',
        flush=True,
    )
    if not available or count != 1:
        raise RuntimeError(
            f'exactly one CUDA GPU required after bootstrap: '
            f'available={available} count={count} visible={visible!r}'
        )

    target = Path('/kaggle/working/train_qwen3_4b_instruct2507_lora_v2.py')
    if not target.is_file():
        # Kaggle normally places both uploaded kernel files in /kaggle/working,
        # but keep a relative fallback for compatibility with script kernels.
        candidate = Path.cwd() / 'train_qwen3_4b_instruct2507_lora_v2.py'
        if candidate.is_file():
            target = candidate
        else:
            raise RuntimeError('pinned v2 training script is missing from kernel')
    source = target.read_text(encoding='utf-8')
    compile(source, str(target), 'exec')
    print(
        f'NEXUS_QWEN4B_V2_BOOTSTRAP script={target} bytes={target.stat().st_size}',
        flush=True,
    )
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
