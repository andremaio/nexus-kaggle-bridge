#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

OUT = Path('/kaggle/working')
ADAPTER = OUT / 'nexus-qwen3-4b-instruct2507-adapter-v3'
MODEL_ID = 'Qwen/Qwen3-4B-Instruct-2507'
MODEL_REV = 'cdbee75f17c01a7cc42f958dc650907174af0554'
HISTORICAL_COMMIT = 'abfc1851b447b5d227fd7e80a69a8c33227f725f'
V7_COMMIT = 'c878db3031496313db7f2cbb0394410204b67b41'
HISTORICAL_RAW = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{HISTORICAL_COMMIT}/training'
V7_RAW = f'https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{V7_COMMIT}/training'
HOLDOUT_V7_SHA256 = 'b49124463a19415473cf161e784b6520ddca1dd3bcd776b7e48bf3946b1d080f'
SOURCES = {
    'seed_sft_v1.jsonl': (HISTORICAL_RAW, 'b9baa9ca58c241c47ab41cb59eb4ece312991d37', 24),
    'seed_sft_v2.jsonl': (HISTORICAL_RAW, '7b84dd2cce420eabbe903fc26258f2c2db7774db', 80),
    'seed_sft_v4.jsonl': (HISTORICAL_RAW, '9624614dab1200842aa27f5989fb6c6b38fdd31f', 364),
    'seed_sft_v5.jsonl': (HISTORICAL_RAW, '764f37ab8a4f1889ab49cc2966766156227ff004', 60),
    'seed_sft_v6_decision_balance.jsonl': (HISTORICAL_RAW, '73e2da7b6926f948f16e7a9e3dbf9311ea94147c', 64),
    'seed_sft_v7_decision_boundary.jsonl': (V7_RAW, '6dc24b3cf653a370d67dad73dabfd534c96f019c', 32),
}
SEED = 20260815
LABELS = ('BLOCK', 'VERIFY', 'ALLOW', 'DEFER')

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f'blob {len(payload)}\0'.encode('ascii')
    return hashlib.sha1(header + payload).hexdigest()


def download(name: str) -> Path:
    raw, expected_blob, _ = SOURCES[name]
    dest = OUT / name
    last = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(f'{raw}/{name}', headers={'User-Agent':'nexus-qwen4b-v3'})
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            if not payload:
                raise RuntimeError('empty payload')
            actual = git_blob_sha1(payload)
            if actual != expected_blob:
                raise RuntimeError(f'immutable Git blob mismatch for {name}: {actual} != {expected_blob}')
            dest.write_bytes(payload)
            print(f'NEXUS_QWEN4B_V3_SOURCE_OK name={name} git_blob={actual} bytes={len(payload)}', flush=True)
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
    decision_counts = {label:0 for label in LABELS}
    for name, rows in by_file.items():
        expected_count = SOURCES[name][2]
        if len(rows) != expected_count:
            raise RuntimeError(f'{name}: count {len(rows)} != {expected_count}')
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
            if name in {'seed_sft_v6_decision_balance.jsonl','seed_sft_v7_decision_boundary.jsonl'}:
                label = str(messages[-1].get('content','')).strip().upper()
                if label not in decision_counts:
                    raise RuntimeError(f'invalid decision label: {rid}')
                decision_counts[label] += 1
    if decision_counts != {label:24 for label in LABELS}:
        raise RuntimeError(f'unbalanced decision curriculum: {decision_counts}')
    total = sum(counts.values())
    if total != 624:
        raise RuntimeError(f'expected 624 examples, got {total}')
    if 'holdout_v7.jsonl' in counts:
        raise RuntimeError('blind holdout entered training set')
    return {
        'schema':'nexus.training.dataset.qwen3-4b-instruct2507.v3',
        'examples':total,
        'files':counts,
        'historical_commit':HISTORICAL_COMMIT,
        'v7_commit':V7_COMMIT,
        'decision_balance':decision_counts,
        'blind_holdout_v7_sha256':HOLDOUT_V7_SHA256,
        'blind_holdout_available_to_training':False,
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
    print('NEXUS_QWEN4B_V3_TRAIN_START', gpu, flush=True)

    paths = {name: download(name) for name in SOURCES}
    by_file = {name: jsonl(path) for name, path in paths.items()}
    manifest = validate_rows(by_file)
    manifest['source_sha256'] = {name: sha256(path) for name, path in paths.items()}
    manifest['source_git_blobs'] = {name: SOURCES[name][1] for name in SOURCES}
    dump(OUT / 'qwen3-4b-instruct2507-dataset-manifest-v3.json', manifest)

    install_stack()
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(SEED)
    rows = []
    for name in SOURCES:
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
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj'],
    )
    args = SFTConfig(
        output_dir=str(OUT / 'trainer-qwen3-4b-instruct2507-v3'),
        num_train_epochs=1.0,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=5e-6,
        lr_scheduler_type='cosine',
        warmup_steps=8,
        weight_decay=0.01,
        max_grad_norm=0.5,
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
        'schema':'nexus.training.qwen3-4b-instruct2507-lora.v3',
        'model_id':MODEL_ID,
        'model_revision':MODEL_REV,
        'gpu':gpu,
        'seed':SEED,
        'source_adapter':None,
        'trained_from_clean_base':True,
        'dataset':manifest,
        'training':{
            'epochs':1.0,
            'learning_rate':5e-6,
            'lr_scheduler_type':'cosine',
            'max_grad_norm':0.5,
            'lora_r':8,
            'lora_alpha':16,
            'lora_targets':['q_proj','k_proj','v_proj','o_proj'],
            'training_loss':float(result.training_loss),
        },
        'adapter_files':adapter_files,
        'evaluation_performed_in_training_kernel':False,
        'blind_holdout_available_to_training':False,
        'baseline_loaded_after_training':False,
        'human_review_required':True,
        'automatic_promotion_authorized':False,
        'automatic_activation_authorized':False,
        'paid_service_used':False,
    }
    dump(OUT / 'qwen3-4b-instruct2507-lora-v3.json', report)
    print('NEXUS_QWEN4B_V3_TRAIN_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
