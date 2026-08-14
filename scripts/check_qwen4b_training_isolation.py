#!/usr/bin/env python3
from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
import sys
import unicodedata
import urllib.request

DATA_COMMIT = "a10104c50eb4320acda30592c424e75848698df1"
RAW = f"https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{DATA_COMMIT}/training"
API = f"https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/training"
TRAIN_FILES = {
    "seed_sft_v1.jsonl": "b9baa9ca58c241c47ab41cb59eb4ece312991d37",
    "seed_sft_v2.jsonl": "7b84dd2cce420eabbe903fc26258f2c2db7774db",
    "seed_sft_v4.jsonl": "9624614dab1200842aa27f5989fb6c6b38fdd31f",
    "seed_sft_v5.jsonl": "764f37ab8a4f1889ab49cc2966766156227ff004",
}
EVAL_FILES = {
    "benchmark_fixed_v1.jsonl": "01992f104fb1945a31a99130006ad1b1579aa62f",
    "benchmark_adversarial_v1.jsonl": "d3eab85cde1807b986d8b8e5246022b2079958cc",
    "holdout_v2.jsonl": "e2c44c002ccc5ae8ffb0c40b7d4bbc7964c3c2f5",
    "holdout_v3.jsonl": "46ba39ad8ed8398a9092535fcd1563b162aa69aa",
}
SEQUENCE_LIMIT = 0.90
JACCARD_LIMIT = 0.82


def _get_json(url: str) -> object:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "nexus-qwen4b-isolation-gate"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_bytes(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nexus-qwen4b-isolation-gate"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError("empty source payload")
    return payload


def _jsonl(payload: bytes) -> list[dict]:
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("invalid jsonl source")
    return rows


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text).casefold())
    plain = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(plain.split())


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", _fold(text), flags=re.UNICODE))


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _train_prompt(row: dict) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        raise RuntimeError(f"invalid training messages for {row.get('id')}")
    if messages[-1].get("role") != "assistant":
        raise RuntimeError(f"training row does not end in assistant completion: {row.get('id')}")
    prompt = "\n".join(str(item.get("content", "")) for item in messages[:-1]).strip()
    if not prompt:
        raise RuntimeError(f"empty training prompt: {row.get('id')}")
    return prompt


def main() -> int:
    expected = {**TRAIN_FILES, **EVAL_FILES}
    for name, blob in expected.items():
        metadata = _get_json(f"{API}/{name}?ref={DATA_COMMIT}")
        if not isinstance(metadata, dict) or metadata.get("sha") != blob:
            raise RuntimeError(f"immutable blob mismatch for {name}")

    train_rows: list[tuple[str, str, str]] = []
    for name in TRAIN_FILES:
        for row in _jsonl(_get_bytes(f"{RAW}/{name}")):
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                raise RuntimeError(f"missing training id in {name}")
            train_rows.append((name, row_id, _train_prompt(row)))

    eval_rows: list[tuple[str, str, str]] = []
    for name in EVAL_FILES:
        for row in _jsonl(_get_bytes(f"{RAW}/{name}")):
            row_id = str(row.get("id", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            if not row_id or not prompt:
                raise RuntimeError(f"invalid evaluation row in {name}")
            eval_rows.append((name, row_id, prompt))

    train_ids = [row_id for _, row_id, _ in train_rows]
    if len(train_ids) != len(set(train_ids)):
        raise RuntimeError("duplicate training ids")
    normalized_train = [_fold(prompt) for _, _, prompt in train_rows]
    if len(normalized_train) != len(set(normalized_train)):
        raise RuntimeError("duplicate normalized training prompts")

    nearest = {"sequence_ratio": 0.0, "token_jaccard": 0.0, "train_id": None, "eval_id": None}
    violations: list[dict] = []
    per_eval_file = {name: {"cases": 0, "near_matches": 0} for name in EVAL_FILES}
    for eval_name, eval_id, eval_prompt in eval_rows:
        per_eval_file[eval_name]["cases"] += 1
        folded_eval = _fold(eval_prompt)
        for train_name, train_id, train_prompt in train_rows:
            folded_train = _fold(train_prompt)
            if folded_train == folded_eval:
                seq = jac = 1.0
            else:
                jac = _jaccard(train_prompt, eval_prompt)
                seq = SequenceMatcher(None, folded_train, folded_eval).ratio()
            if seq > nearest["sequence_ratio"]:
                nearest.update(sequence_ratio=round(seq, 6), train_id=train_id, eval_id=eval_id)
            if jac > nearest["token_jaccard"]:
                nearest["token_jaccard"] = round(jac, 6)
            if seq >= SEQUENCE_LIMIT or jac >= JACCARD_LIMIT:
                per_eval_file[eval_name]["near_matches"] += 1
                violations.append(
                    {
                        "train_file": train_name,
                        "train_id": train_id,
                        "eval_file": eval_name,
                        "eval_id": eval_id,
                        "sequence_ratio": round(seq, 6),
                        "token_jaccard": round(jac, 6),
                    }
                )

    result = {
        "schema": "nexus.training-isolation.qwen3-4b.v1",
        "data_commit": DATA_COMMIT,
        "train_examples": len(train_rows),
        "evaluation_cases": len(eval_rows),
        "evaluation_sets": per_eval_file,
        "sequence_limit": SEQUENCE_LIMIT,
        "token_jaccard_limit": JACCARD_LIMIT,
        "nearest": nearest,
        "violations": violations[:50],
        "violation_count": len(violations),
        "fresh_v3_used_for_training": False,
        "fresh_v3_used_for_hyperparameter_selection": False,
        "training_allowed": not violations,
        "automatic_training_authorized": False,
        "automatic_promotion_authorized": False,
        "automatic_activation_authorized": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if not violations else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"schema": "nexus.training-isolation.qwen3-4b.failure.v1", "error_type": type(exc).__name__, "error": str(exc), "training_allowed": False}, ensure_ascii=False, sort_keys=True))
        raise
