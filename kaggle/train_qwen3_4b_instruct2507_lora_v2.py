#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

OUT = Path('/kaggle/working')
ADAPTER = OUT / 'nexus-qwen3-4b-instruct2507-adapter-v2'
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
MODEL_REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'
DATA_COMMIT = 'abfc1851b447b5d227fd7e80a69a8c33227f725f'
RAW = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{DATA_COMMIT}/training'
TRAIN_FILES = [
    'seed_sft_v1.jsonl',
    'seed_sft_v2.jsonl',
    'seed_sft_v4.jsonl',
    'seed_sft_v5.jsonl',
    'seed_sft_v6_decision_balance.jsonl',
]
EXPECTED_BLOBS = {
    'seed_sft_v1.jsonl': 'b9baa9ca58c241c47ab41cb59eb4ece312991d37',
    'seed_sft_v2.jsonl': '7b84dd2cce420eabbe903fc26258f2c2db7774db',
    'seed_sft_v4.jsonl': '9624614dab1200842aa27f5989fb6c6b38fdd31f',
    'seed_sft_v5.jsonl': '764f37ab8a4f1889ab49cc2966766156227ff004',
    'seed_sft_v6_decision_balance.jsonl': '73e2da7b6926f948f16e7a9e3dbf9311ea94147c',
}
EXPECTED_COUNTS = {
    'seed_sft_v1.jsonl': 24,
    'seed_sft_v2.jsonl': 64,
    'seed_sft_v4.jsonl': 400,
    'seed_sft_v5.jsonl': 40,
    'seed_sft_v6_decision_balance.jsonl': 64,
}
SEED = 20260814

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('DO_NOT_TRACK', '1')
os.environ.setdefault('WANDB_DISABLED', 'true')
os.environ.setdefault('DISABLE_MLFLOW_INTEGRATION', 'TRUE')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HOME', '/kaggle/temp/hf-cache')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def dump(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + '\n', encoding='utf-8')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f'blob {len(payload)}\0'.encode('ascii')
    return hashlib.sha1(header + payload).hexdigest()


def download(name: str) -> Path:
    dest = OUT / name
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f'{RAW}/{name}', timeout=90) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError('empty payload')
            actual_blob = git_blob_sha1(payload)
            expected_blob = EXPECTED_BLOBS[name]
            if actual_blob != expected_blob:
                raise RuntimeError(
                    f'immutable Git blob mismatch for {name}: '
                    f'{actual_blob} != {expected_blob}'
                )
            dest.write_bytes(payload)
            print(
                f'NEXUS_QWEN4B_V2_SOURCE_OK name={name} '
                f'git_blob={actual_blob} bytes={len(payload)}',
                flush=True,
            )
            return dest
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'failed to download {name}: {type(last).__name__}: {last}')


def jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f'invalid jsonl: {path.name}')
    return rows


def validate_rows(by_file: dict[str, list[dict]]) -> dict:
    ids: set[str] = set()
    prompts: set[str] = set()
    counts: dict[str, int] = {}
    decision_counts = {'BLOCK':0,'VERIFY':0,'ALLOW':0,'DEFER':0}
    for name, rows in by_file.items():
        if len(rows) != EXPECTED_COUNTS[name]:
            raise RuntimeError(f'{name}: count {len(rows)} != {EXPECTED_COUNTS[name]}')
        counts[name] = len(rows)
        for row in rows:
            rid = str(row.get('id','')).strip()
            messages = row.get('messages')
            if not rid or rid in ids:
                raise RuntimeError(f'duplicate/missing id: {rid}')
            if not isinstance(messages, list) or len(messages) < 2 or messages[-1].get('role') != 'assistant':
                raise RuntimeError(f'invalid messages: {rid}')
            prompt = json.dumps(messages[:-1], ensure_ascii=False, sort_keys=True)
            if prompt in prompts:
                raise RuntimeError(f'duplicate prompt: {rid}')
            ids.add(rid); prompts.add(prompt)
            if name == 'seed_sft_v6_decision_balance.jsonl':
                label = str(messages[-1].get('content','')).strip().upper()
                if label not in decision_counts:
                    raise RuntimeError(f'invalid decision label: {rid}')
                decision_counts[label] += 1
    if decision_counts != {'BLOCK':16,'VERIFY':16,'ALLOW':16,'DEFER':16}:
        raise RuntimeError(f'unbalanced decision curriculum: {decision_counts}')
    total = sum(counts.values())
    if total != 592:
        raise RuntimeError(f'expected 592 examples, got {total}')
    return {
        'schema':'nexus.training.dataset.qwen3-4b-instruct2507.v2',
        'source_commit':DATA_COMMIT,
        'examples':total,
        'files':counts,
        'decision_balance':decision_counts,
        'contains_private_user_conversations':False,
        'contains_credentials':False,
        'holdouts_in_training':False,
    }


def prompt_completion(row: dict) -> dict:
    messages = list(row['messages'])
    return {'prompt': messages[:-1], 'completion': [messages[-1]]}


def install_stack() -> None:
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-q',
        'transformers==5.14.1', 'datasets==5.0.0', 'peft==0.19.1',
        'trl==1.9.0', 'accelerate==1.14.0',
    ])


def main() -> None:
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('exactly one CUDA GPU required')
    gpu = torch.cuda.get_device_name(0)
    print('NEXUS_QWEN4B_V2_TRAIN_START', gpu, flush=True)

    paths = {name: download(name) for name in TRAIN_FILES}
    by_file = {name: jsonl(path) for name, path in paths.items()}
    dataset_manifest = validate_rows(by_file)
    dataset_manifest['source_sha256'] = {name: sha256(path) for name, path in paths.items()}
    dataset_manifest['source_git_blobs'] = dict(EXPECTED_BLOBS)
    dump(OUT / 'qwen3-4b-instruct2507-dataset-manifest-v2.json', dataset_manifest)

    install_stack()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(SEED)
    rows = []
    for name in TRAIN_FILES:
        rows.extend(by_file[name])
    dataset = Dataset.from_list([prompt_completion(row) for row in rows]).shuffle(seed=SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REV, trust_remote_code=False, token=False, dtype=torch.float16
    )
    base.config.use_cache = False
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    )
    args = SFTConfig(
        output_dir=str(OUT / 'trainer-qwen3-4b-instruct2507-v2'),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=1e-5,
        warmup_steps=8,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy='no',
        max_length=1024,
        completion_only_loss=True,
        loss_type='chunked_nll',
        report_to='none',
        push_to_hub=False,
        seed=SEED,
        data_seed=SEED,
        fp16=True,
        bf16=False,
    )
    trainer = SFTTrainer(
        model=base,
        args=args,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(str(ADAPTER))
    tokenizer.save_pretrained(str(ADAPTER))

    adapter_files = {}
    for path in sorted(ADAPTER.iterdir()):
        if path.is_file():
            adapter_files[path.name] = {'sha256':sha256(path),'bytes':path.stat().st_size}
    report = {
        'schema':'nexus.training.qwen3-4b-instruct2507-lora.v2',
        'model_id':MODEL_ID,
        'model_revision':MODEL_REV,
        'gpu':gpu,
        'seed':SEED,
        'dataset':dataset_manifest,
        'training':{
            'epochs':1.0,
            'learning_rate':1e-5,
            'lora_r':16,
            'lora_alpha':32,
            'training_loss':float(result.training_loss),
        },
        'adapter_files':adapter_files,
        'evaluation_performed_in_training_kernel':False,
        'baseline_loaded_after_training':False,
        'human_review_required':True,
        'automatic_promotion_authorized':False,
        'automatic_activation_authorized':False,
        'paid_service_used':False,
    }
    dump(OUT / 'qwen3-4b-instruct2507-lora-v2.json', report)
    print('NEXUS_QWEN4B_V2_TRAIN_COMPLETE', flush=True)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        dump(OUT / 'qwen3-4b-instruct2507-lora-v2-failure.json', {
            'schema':'nexus.training.qwen3-4b-instruct2507-lora.v2.failure',
            'error_type':type(exc).__name__,
            'error':str(exc),
            'traceback':traceback.format_exc()[-20000:],
            'automatic_promotion_authorized':False,
            'automatic_activation_authorized':False,
        })
        raise
