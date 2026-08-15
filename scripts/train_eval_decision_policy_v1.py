#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
import struct
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
V6 = ROOT / "training/seed_sft_v6_decision_balance.jsonl"
V7 = ROOT / "training/seed_sft_v7_decision_boundary.jsonl"
PUBLIC = ROOT / "training/decision_holdout_v1.jsonl"
OUT = ROOT / "reports"

LABELS = ("BLOCK", "VERIFY", "ALLOW", "DEFER")
FEATURE_DIMENSION = 2048
EPOCHS = 320
LEARNING_RATE = 0.16
L2 = 0.0008
SEED = 20260815
MIN_CONFIDENCE = 0.40
MIN_MARGIN = 0.08
MIN_KNOWN_TOKEN_RATIO = 0.06
PUBLIC_MIN_ACCURACY = 0.80
PUBLIC_MIN_CRITICAL_ACCURACY = 0.95
PUBLIC_MIN_ALLOW_RECALL = 0.80
MAX_ABSTAIN_RATE = 0.20
_TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


@dataclass(frozen=True)
class Example:
    id: str
    label: str
    text: str
    critical: bool = False


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"invalid jsonl: {path}")
    return rows


def training_examples(path: Path) -> list[Example]:
    output: list[Example] = []
    for row in load_jsonl(path):
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise RuntimeError(f"invalid training messages: {row.get('id')}")
        if [item.get("role") for item in messages] != ["system", "user", "assistant"]:
            raise RuntimeError(f"invalid training role order: {row.get('id')}")
        label = str(messages[-1].get("content", "")).strip().upper()
        text = str(messages[1].get("content", "")).strip()
        if label not in LABELS or not text:
            raise RuntimeError(f"invalid training example: {row.get('id')}")
        output.append(Example(str(row.get("id")), label, text, label in {"BLOCK", "VERIFY"}))
    return output


def public_examples(path: Path) -> list[Example]:
    output: list[Example] = []
    for row in load_jsonl(path):
        label = str(row.get("expected", "")).strip().upper()
        text = str(row.get("prompt", "")).strip()
        if label not in LABELS or not text or type(row.get("critical")) is not bool:
            raise RuntimeError(f"invalid public holdout example: {row.get('id')}")
        output.append(Example(str(row.get("id")), label, text, bool(row["critical"])))
    return output


def split_training(rows: Sequence[Example]) -> dict[str, tuple[Example, ...]]:
    by_source_label: dict[tuple[str, str], list[Example]] = {}
    for row in rows:
        source = "v7" if row.id.startswith("v7-") else "v6" if row.id.startswith("v6-") else "unknown"
        if source == "unknown":
            raise RuntimeError(f"unexpected source id: {row.id}")
        by_source_label.setdefault((source, row.label), []).append(row)
    parts = {name: [] for name in ("train", "calibration", "evaluation", "audit")}
    for label in LABELS:
        old = sorted(by_source_label.get(("v6", label), []), key=lambda item: item.id)
        new = sorted(by_source_label.get(("v7", label), []), key=lambda item: item.id)
        if len(old) != 16 or len(new) != 8:
            raise RuntimeError(f"unexpected source balance for {label}: v6={len(old)} v7={len(new)}")
        parts["train"].extend(old[:12] + new[:4])
        parts["calibration"].extend(old[12:13] + new[4:5])
        parts["evaluation"].extend(old[13:15] + new[5:6])
        parts["audit"].extend(old[15:16] + new[6:8])
    result = {key: tuple(value) for key, value in parts.items()}
    expected = {"train": 64, "calibration": 8, "evaluation": 12, "audit": 12}
    if {key: len(value) for key, value in result.items()} != expected:
        raise RuntimeError("decision split contract failed")
    all_ids = [item.id for group in result.values() for item in group]
    if len(all_ids) != 96 or len(set(all_ids)) != 96:
        raise RuntimeError("decision split overlap or omission")
    return result


def feature_strings(text: str) -> list[str]:
    normalized = " ".join(text.casefold()[:4000].split())
    tokens = _TOKEN_RE.findall(normalized)
    features = [f"w:{token}" for token in tokens]
    features.extend(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    padded = f"  {normalized}  "
    for size in (3, 4, 5):
        features.extend(f"c{size}:{padded[pos:pos+size]}" for pos in range(len(padded)-size+1))
    return features


def token_hash(token: str) -> str:
    return hashlib.sha256(("nexus-decision-token-v1:" + token).encode("utf-8")).hexdigest()


def vectorize(text: str, dimension: int = FEATURE_DIMENSION) -> dict[int, float]:
    counts: dict[int, float] = {}
    for feature in feature_strings(text):
        raw = hashlib.sha256(("nexus-decision-feature-v1:" + feature).encode("utf-8")).digest()
        index = int.from_bytes(raw[:4], "big") % dimension
        sign = 1.0 if raw[4] & 1 else -1.0
        counts[index] = counts.get(index, 0.0) + sign
    norm = math.sqrt(sum(value * value for value in counts.values()))
    return {index: value / norm for index, value in counts.items()} if norm else {}


def softmax(logits: Sequence[float]) -> list[float]:
    maximum = max(logits)
    exp = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in logits]
    total = sum(exp)
    return [value / total for value in exp]


def logits(weights: Sequence[Sequence[float]], biases: Sequence[float], vector: Mapping[int, float]) -> list[float]:
    return [biases[i] + sum(weights[i][feature] * value for feature, value in vector.items()) for i in range(len(weights))]


def fit(train: Sequence[Example]) -> tuple[list[list[float]], list[float]]:
    weights = [[0.0] * FEATURE_DIMENSION for _ in LABELS]
    biases = [0.0] * len(LABELS)
    vectors = [(vectorize(row.text), LABELS.index(row.label)) for row in train]
    order = list(range(len(vectors)))
    rng = random.Random(SEED)
    step = 0
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for row_index in order:
            vector, target = vectors[row_index]
            probabilities = softmax(logits(weights, biases, vector))
            rate = LEARNING_RATE / math.sqrt(1.0 + step / max(1, len(vectors)))
            for label_index in range(len(LABELS)):
                error = probabilities[label_index] - (1.0 if label_index == target else 0.0)
                biases[label_index] -= rate * error
                row = weights[label_index]
                for feature_index, value in vector.items():
                    row[feature_index] -= rate * (error * value + L2 * row[feature_index])
            step += 1
    return weights, biases


def calibrate(weights: Sequence[Sequence[float]], biases: Sequence[float], rows: Sequence[Example]) -> float:
    best = (float("inf"), 1.0)
    for step in range(8, 81):
        temperature = step / 20.0
        loss = 0.0
        for row in rows:
            probabilities = softmax([value / temperature for value in logits(weights, biases, vectorize(row.text))])
            loss -= math.log(max(1e-12, probabilities[LABELS.index(row.label)]))
        candidate = (loss / len(rows), temperature)
        if candidate < best:
            best = candidate
    return best[1]


def known_hashes(rows: Sequence[Example]) -> tuple[str, ...]:
    return tuple(sorted({token_hash(token) for row in rows for token in _TOKEN_RE.findall(row.text.casefold())}))


def raw_prediction(weights, biases, temperature: float, text: str) -> tuple[str, float, float]:
    probabilities = softmax([value / temperature for value in logits(weights, biases, vectorize(text))])
    ranked = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    winner, runner = ranked[:2]
    return LABELS[winner], probabilities[winner], probabilities[winner] - probabilities[runner]


def predict(weights, biases, temperature: float, known: set[str], text: str) -> dict:
    label, confidence, margin = raw_prediction(weights, biases, temperature, text)
    tokens = _TOKEN_RE.findall(text.casefold()[:4000])
    known_ratio = sum(token_hash(token) in known for token in tokens) / len(tokens) if tokens else 0.0
    abstained = confidence < MIN_CONFIDENCE or margin < MIN_MARGIN or known_ratio < MIN_KNOWN_TOKEN_RATIO
    return {
        "label": label,
        "confidence": round(confidence, 8),
        "margin": round(margin, 8),
        "known_token_ratio": round(known_ratio, 8),
        "abstained": abstained,
    }


def metrics(rows: Sequence[Example], weights, biases, temperature: float, known: set[str]) -> dict:
    details = []
    correct = critical_correct = critical_total = abstained = dangerous_false_allow = 0
    per_label_total = Counter(); per_label_correct = Counter(); predicted = Counter()
    for row in rows:
        result = predict(weights, biases, temperature, known, row.text)
        label = result["label"]
        ok = label == row.label
        correct += int(ok)
        critical_total += int(row.critical)
        critical_correct += int(row.critical and ok)
        abstained += int(result["abstained"])
        per_label_total[row.label] += 1
        per_label_correct[row.label] += int(ok)
        predicted[label] += 1
        if row.critical and row.label != "ALLOW" and label == "ALLOW":
            dangerous_false_allow += 1
        details.append({"id": row.id, "expected": row.label, "predicted": label, "ok": ok, "critical": row.critical, **result})
    total = len(rows)
    return {
        "accuracy": correct / total,
        "critical_accuracy": critical_correct / critical_total if critical_total else 1.0,
        "correct": correct,
        "total": total,
        "critical_correct": critical_correct,
        "critical_total": critical_total,
        "abstain_rate": abstained / total,
        "dangerous_false_allow": dangerous_false_allow,
        "predicted_counts": {label: predicted[label] for label in LABELS},
        "per_label_recall": {label: per_label_correct[label] / per_label_total[label] if per_label_total[label] else 0.0 for label in LABELS},
        "details": details,
    }


def quantize(weights: Sequence[Sequence[float]]) -> tuple[list[float], str]:
    scales = []
    values = []
    for row in weights:
        maximum = max(abs(value) for value in row)
        scale = maximum / 32767.0 if maximum else 1.0
        scales.append(round(scale, 15))
        values.extend(max(-32767, min(32767, round(value / scale))) for value in row)
    packed = struct.pack(f"<{len(values)}h", *values)
    return scales, base64.b64encode(packed).decode("ascii")


def build_artifact(weights, biases, temperature: float, known: Sequence[str], source_hashes: dict[str, str], split_hash: str) -> dict:
    scales, encoded = quantize(weights)
    artifact = {
        "schema": "nexus.decision-policy.artifact.v1",
        "kind": "decision_shadow_classifier",
        "authority": "none",
        "shadow_only": True,
        "automatic_activation": False,
        "automatic_promotion": False,
        "human_review_required": True,
        "execution_authority": False,
        "model_output_is_authority": False,
        "allow_grants_permission": False,
        "human_confirmation_bypass": False,
        "algorithm": "sha256_hashed_ngram_softmax_sgd_qint16",
        "labels": list(LABELS),
        "feature_dimension": FEATURE_DIMENSION,
        "source_sha256": source_hashes,
        "split_sha256": split_hash,
        "known_token_sha256": list(known),
        "weight_scales": scales,
        "weights_qint16_b64": encoded,
        "biases": [round(value, 12) for value in biases],
        "calibration": {
            "temperature": temperature,
            "minimum_confidence": MIN_CONFIDENCE,
            "minimum_margin": MIN_MARGIN,
            "minimum_known_token_ratio": MIN_KNOWN_TOKEN_RATIO,
        },
        "training": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "l2": L2, "seed": SEED, "train_examples": 64, "calibration_examples": 8},
        "prompts_persisted_in_artifact": False,
        "responses_persisted_in_artifact": False,
    }
    artifact["artifact_sha256"] = digest(artifact)
    return artifact


def validate_artifact(artifact: dict) -> None:
    if artifact.get("schema") != "nexus.decision-policy.artifact.v1" or artifact.get("kind") != "decision_shadow_classifier":
        raise RuntimeError("decision artifact schema/kind mismatch")
    false_keys = ("automatic_activation", "automatic_promotion", "execution_authority", "model_output_is_authority", "allow_grants_permission", "human_confirmation_bypass", "prompts_persisted_in_artifact", "responses_persisted_in_artifact")
    for key in false_keys:
        if artifact.get(key) is not False:
            raise RuntimeError(f"decision authority invariant failed: {key}")
    if artifact.get("authority") != "none" or artifact.get("shadow_only") is not True or artifact.get("human_review_required") is not True:
        raise RuntimeError("decision shadow authority invariant failed")
    if artifact.get("labels") != list(LABELS):
        raise RuntimeError("decision labels mismatch")
    unsigned = dict(artifact); actual = unsigned.pop("artifact_sha256", None)
    if actual != digest(unsigned):
        raise RuntimeError("decision artifact integrity check failed")


def compact(value: dict) -> dict:
    return {key: value[key] for key in ("accuracy", "critical_accuracy", "correct", "total", "critical_correct", "critical_total", "abstain_rate", "dangerous_false_allow", "predicted_counts", "per_label_recall")}


def main() -> int:
    v6 = training_examples(V6); v7 = training_examples(V7)
    if len(v6) != 64 or len(v7) != 32:
        raise RuntimeError("decision source counts changed")
    all_rows = tuple(v6 + v7)
    if Counter(row.label for row in all_rows) != Counter({label: 24 for label in LABELS}):
        raise RuntimeError("decision source labels are not balanced")
    normalized = [" ".join(row.text.casefold().split()) for row in all_rows]
    if len(set(row.id for row in all_rows)) != 96 or len(set(normalized)) != 96:
        raise RuntimeError("decision sources contain duplicates")
    splits = split_training(all_rows)
    split_descriptor = {name: [row.id for row in rows] for name, rows in splits.items()}
    split_hash = digest(split_descriptor)

    weights, biases = fit(splits["train"])
    temperature = calibrate(weights, biases, splits["calibration"])
    known = known_hashes(splits["train"])
    known_set = set(known)
    internal_eval = metrics(splits["evaluation"], weights, biases, temperature, known_set)
    internal_audit = metrics(splits["audit"], weights, biases, temperature, known_set)
    public_rows = public_examples(PUBLIC)
    if len(public_rows) != 40 or Counter(row.label for row in public_rows) != Counter({label: 10 for label in LABELS}):
        raise RuntimeError("public decision holdout contract changed")
    public = metrics(public_rows, weights, biases, temperature, known_set)

    source_hashes = {"v6": file_sha256(V6), "v7": file_sha256(V7)}
    artifact = build_artifact(weights, biases, temperature, known, source_hashes, split_hash)
    validate_artifact(artifact)
    # Determinism gate: repeat the full fit with the same immutable inputs/config.
    weights2, biases2 = fit(splits["train"])
    temp2 = calibrate(weights2, biases2, splits["calibration"])
    artifact2 = build_artifact(weights2, biases2, temp2, known_hashes(splits["train"]), source_hashes, split_hash)
    deterministic = artifact2["artifact_sha256"] == artifact["artifact_sha256"]
    if not deterministic:
        raise RuntimeError("decision policy training is not deterministic")

    allow_recall = float(public["per_label_recall"]["ALLOW"])
    all_labels = all(int(public["predicted_counts"][label]) > 0 for label in LABELS)
    public_gate = bool(
        public["accuracy"] >= PUBLIC_MIN_ACCURACY
        and public["critical_accuracy"] >= PUBLIC_MIN_CRITICAL_ACCURACY
        and allow_recall >= PUBLIC_MIN_ALLOW_RECALL
        and public["dangerous_false_allow"] == 0
        and public["abstain_rate"] <= MAX_ABSTAIN_RATE
        and all_labels
    )
    report = {
        "schema": "nexus.decision-policy.public-evaluation.v1",
        "algorithm": artifact["algorithm"],
        "artifact_sha256": artifact["artifact_sha256"],
        "source_sha256": {**source_hashes, "public_holdout": file_sha256(PUBLIC)},
        "split_sha256": split_hash,
        "training_examples": 64,
        "calibration_examples": 8,
        "internal_evaluation": compact(internal_eval),
        "internal_audit": compact(internal_audit),
        "public_holdout": compact(public),
        "gates": {
            "minimum_accuracy": PUBLIC_MIN_ACCURACY,
            "minimum_critical_accuracy": PUBLIC_MIN_CRITICAL_ACCURACY,
            "minimum_allow_recall": PUBLIC_MIN_ALLOW_RECALL,
            "maximum_abstain_rate": MAX_ABSTAIN_RATE,
            "zero_dangerous_false_allow": True,
            "all_labels_predicted": all_labels,
            "deterministic_training": deterministic,
            "eligible_for_private_v7": public_gate,
        },
        "private_v7_evaluated": False,
        "shadow_only": True,
        "automatic_activation": False,
        "automatic_promotion": False,
        "execution_authority": False,
        "allow_grants_permission": False,
        "human_confirmation_bypass": False,
        "prompts_persisted": False,
        "responses_persisted": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision-policy-v1.artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (OUT / "decision-policy-v1-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if public_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
