#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Callable, Sequence

from scripts import decision_policy_v4_data as data
from scripts import train_eval_decision_policy_v1 as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
SCHEMA = "nexus.decision-policy.artifact.v4"
REPORT_SCHEMA = "nexus.decision-policy.public-evaluation.v4"
DIMENSION = 2048
EPOCHS = 180
LEARNING_RATE = 0.14
L2 = 0.0008
SEED = 20260815
MIN_CONFIDENCE = 0.42
MIN_MARGIN = 0.07

AXES = {
    "effect": data.EFFECTS,
    "authority": data.AUTHORITIES,
    "evidence": data.EVIDENCE,
    "actionability": data.ACTIONABILITY,
}


class AxisModel:
    def __init__(self, name: str, labels: tuple[str, ...], weights, biases, temperature: float, known: tuple[str, ...]):
        self.name = name
        self.labels = labels
        self.weights = weights
        self.biases = biases
        self.temperature = temperature
        self.known = known


def _target(row: data.AxisExample, axis: str) -> str:
    return str(getattr(row, axis))


def fit_axis(rows: Sequence[data.AxisExample], axis: str, labels: tuple[str, ...], seed_offset: int = 0) -> tuple[list[list[float]], list[float]]:
    weights = [[0.0] * DIMENSION for _ in labels]
    biases = [0.0] * len(labels)
    vectors = [(base.vectorize(row.text, DIMENSION), labels.index(_target(row, axis))) for row in rows]
    order = list(range(len(vectors)))
    rng = random.Random(SEED + seed_offset)
    step = 0
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for row_index in order:
            vector, target = vectors[row_index]
            probabilities = base.softmax(base.logits(weights, biases, vector))
            rate = LEARNING_RATE / math.sqrt(1.0 + step / max(1, len(vectors)))
            for label_index in range(len(labels)):
                error = probabilities[label_index] - (1.0 if label_index == target else 0.0)
                biases[label_index] -= rate * error
                row_weights = weights[label_index]
                for feature_index, value in vector.items():
                    row_weights[feature_index] -= rate * (error * value + L2 * row_weights[feature_index])
            step += 1
    return weights, biases


def calibrate_axis(weights, biases, rows: Sequence[data.AxisExample], axis: str, labels: tuple[str, ...]) -> float:
    best = (float("inf"), 1.0)
    for step in range(8, 81):
        temperature = step / 20.0
        loss = 0.0
        for row in rows:
            probabilities = base.softmax([value / temperature for value in base.logits(weights, biases, base.vectorize(row.text, DIMENSION))])
            loss -= math.log(max(1e-12, probabilities[labels.index(_target(row, axis))]))
        candidate = (loss / max(1, len(rows)), temperature)
        if candidate < best:
            best = candidate
    return best[1]


def known_hashes(rows: Sequence[data.AxisExample]) -> tuple[str, ...]:
    return tuple(sorted({base.token_hash(token) for row in rows for token in base._TOKEN_RE.findall(row.text.casefold())}))


def train_models() -> dict[str, AxisModel]:
    train = data.train_rows()
    calibration = data.calibration_rows()
    known = known_hashes(train)
    models: dict[str, AxisModel] = {}
    for index, (axis, labels) in enumerate(AXES.items()):
        weights, biases = fit_axis(train, axis, labels, seed_offset=index * 97)
        temperature = calibrate_axis(weights, biases, calibration, axis, labels)
        models[axis] = AxisModel(axis, labels, weights, biases, temperature, known)
    return models


def axis_predict(model: AxisModel, text: str) -> dict:
    vector = base.vectorize(text, DIMENSION)
    probabilities = base.softmax([value / model.temperature for value in base.logits(model.weights, model.biases, vector)])
    ranked = sorted(range(len(probabilities)), key=lambda index: (-probabilities[index], index))
    winner, runner = ranked[:2]
    confidence = probabilities[winner]
    margin = confidence - probabilities[runner]
    tokens = base._TOKEN_RE.findall(text.casefold()[:4000])
    known_set = set(model.known)
    known_ratio = sum(base.token_hash(token) in known_set for token in tokens) / len(tokens) if tokens else 0.0
    return {
        "label": model.labels[winner],
        "confidence": confidence,
        "margin": margin,
        "known_token_ratio": known_ratio,
        "uncertain": confidence < MIN_CONFIDENCE or margin < MIN_MARGIN,
    }


def map_predictions(predictions: dict[str, dict]) -> tuple[str, str]:
    effect = predictions["effect"]
    authority = predictions["authority"]
    evidence = predictions["evidence"]
    actionability = predictions["actionability"]

    # Fail-safe mapping from latent states. The shadow label never changes execution authority.
    if authority["label"] == "INVALID":
        return "BLOCK", "authority_invalid"
    if authority["label"] == "UNKNOWN":
        return "VERIFY", "authority_unknown"
    if evidence["label"] == "CHECKABLE":
        return "VERIFY", "evidence_checkable"
    if evidence["label"] == "UNAVAILABLE":
        return "DEFER", "evidence_unavailable"
    if actionability["label"] == "WAIT":
        return "DEFER", "not_actionable_now"

    # An uncertain authority/evidence prediction on an external effect may not become ALLOW.
    if effect["label"] == "EXTERNAL" and (authority["uncertain"] or evidence["uncertain"]):
        return "VERIFY", "external_uncertainty_fallback"
    # Local tasks with sufficient evidence and no invalid/unknown authority are semantically ALLOW.
    if evidence["label"] == "SUFFICIENT" and actionability["label"] == "NOW" and authority["label"] in {"VALID", "IRRELEVANT"}:
        return "ALLOW", "sufficient_authorized_or_local"
    return "VERIFY", "latent_state_fallback"


def predict(models: dict[str, AxisModel], text: str) -> dict:
    axes = {name: axis_predict(model, text) for name, model in models.items()}
    decision, reason = map_predictions(axes)
    confidence = min(value["confidence"] for value in axes.values())
    margin = min(value["margin"] for value in axes.values())
    return {
        "label": decision,
        "reason": reason,
        "axes": axes,
        "confidence": confidence,
        "margin": margin,
        "abstained": any(value["uncertain"] for value in axes.values()),
        "execution_authority": False,
        "allow_grants_permission": False,
    }


def axis_metrics(rows: Sequence[data.AxisExample], models: dict[str, AxisModel]) -> dict:
    output = {}
    for axis, model in models.items():
        correct = 0
        totals = Counter(); hit = Counter()
        for row in rows:
            result = axis_predict(model, row.text)
            expected = _target(row, axis)
            ok = result["label"] == expected
            correct += int(ok); totals[expected] += 1; hit[expected] += int(ok)
        output[axis] = {
            "accuracy": correct / len(rows),
            "per_label_recall": {label: hit[label] / totals[label] if totals[label] else 0.0 for label in model.labels},
        }
    return output


def decision_metrics_axis_rows(rows: Sequence[data.AxisExample], predictor: Callable[[str], dict]) -> dict:
    correct = critical_correct = critical_total = dangerous_false_allow = abstained = 0
    totals = Counter(); hits = Counter(); predicted = Counter(); failures = Counter(); reasons = Counter()
    for row in rows:
        result = predictor(row.text)
        label = result["label"]
        ok = label == row.decision
        correct += int(ok)
        critical_total += int(row.critical)
        critical_correct += int(row.critical and ok)
        dangerous_false_allow += int(row.critical and row.decision != "ALLOW" and label == "ALLOW")
        abstained += int(result["abstained"])
        totals[row.decision] += 1; hits[row.decision] += int(ok); predicted[label] += 1; reasons[result["reason"]] += 1
        if not ok: failures[f"{row.decision}->{label}"] += 1
    total = len(rows)
    return {
        "accuracy": correct / total,
        "critical_accuracy": critical_correct / critical_total if critical_total else 1.0,
        "correct": correct, "total": total,
        "critical_correct": critical_correct, "critical_total": critical_total,
        "dangerous_false_allow": dangerous_false_allow,
        "abstain_rate": abstained / total,
        "predicted_counts": {label: predicted[label] for label in data.DECISIONS},
        "per_label_recall": {label: hits[label] / totals[label] if totals[label] else 0.0 for label in data.DECISIONS},
        "failure_count": sum(failures.values()),
        "failure_summary": dict(sorted(failures.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def decision_metrics_legacy(rows: Sequence[base.Example], predictor: Callable[[str], dict]) -> dict:
    correct = critical_correct = critical_total = dangerous_false_allow = abstained = 0
    totals = Counter(); hits = Counter(); predicted = Counter(); failures = Counter(); reasons = Counter()
    for row in rows:
        result = predictor(row.text)
        label = result["label"]
        ok = label == row.label
        correct += int(ok); critical_total += int(row.critical); critical_correct += int(row.critical and ok)
        dangerous_false_allow += int(row.critical and row.label != "ALLOW" and label == "ALLOW")
        abstained += int(result["abstained"])
        totals[row.label] += 1; hits[row.label] += int(ok); predicted[label] += 1; reasons[result["reason"]] += 1
        if not ok: failures[f"{row.label}->{label}"] += 1
    total = len(rows)
    return {
        "accuracy": correct / total,
        "critical_accuracy": critical_correct / critical_total if critical_total else 1.0,
        "correct": correct, "total": total,
        "critical_correct": critical_correct, "critical_total": critical_total,
        "dangerous_false_allow": dangerous_false_allow,
        "abstain_rate": abstained / total,
        "predicted_counts": {label: predicted[label] for label in data.DECISIONS},
        "per_label_recall": {label: hits[label] / totals[label] if totals[label] else 0.0 for label in data.DECISIONS},
        "failure_count": sum(failures.values()),
        "failure_summary": dict(sorted(failures.items())),
        "reason_counts": dict(sorted(reasons.items())),
    }


def quantize(weights) -> tuple[list[float], str]:
    scales = []
    values = []
    for row in weights:
        maximum = max(abs(value) for value in row)
        scale = maximum / 32767.0 if maximum else 1.0
        scales.append(round(scale, 15))
        values.extend(max(-32767, min(32767, round(value / scale))) for value in row)
    packed = struct.pack(f"<{len(values)}h", *values)
    return scales, base64.b64encode(packed).decode("ascii")


def build_artifact(models: dict[str, AxisModel]) -> dict:
    model_docs = {}
    for axis, model in models.items():
        scales, encoded = quantize(model.weights)
        model_docs[axis] = {
            "labels": list(model.labels),
            "feature_dimension": DIMENSION,
            "weight_scales": scales,
            "weights_qint16_b64": encoded,
            "biases": [round(value, 12) for value in model.biases],
            "temperature": model.temperature,
            "known_token_sha256": list(model.known),
            "minimum_confidence": MIN_CONFIDENCE,
            "minimum_margin": MIN_MARGIN,
        }
    artifact = {
        "schema": SCHEMA,
        "kind": "latent_axis_decision_shadow_policy",
        "version": "nexus.decision-policy.v4",
        "algorithm": "four_axis_sha256_hashed_ngram_softmax_qint16",
        "axes": {name: list(labels) for name, labels in AXES.items()},
        "models": model_docs,
        "mapping_precedence": ["INVALID_AUTHORITY->BLOCK", "UNKNOWN_AUTHORITY->VERIFY", "CHECKABLE_EVIDENCE->VERIFY", "UNAVAILABLE_EVIDENCE->DEFER", "WAIT->DEFER", "SUFFICIENT_NOW_VALID_OR_LOCAL->ALLOW"],
        "source_sha256": {
            "train": data.digest(data.train_rows()),
            "calibration": data.digest(data.calibration_rows()),
            "legacy_v6": base.file_sha256(base.V6),
            "legacy_v7": base.file_sha256(base.V7),
        },
        "training": {"epochs": EPOCHS, "learning_rate": LEARNING_RATE, "l2": L2, "seed": SEED, "train_examples": len(data.train_rows()), "calibration_examples": len(data.calibration_rows())},
        "private_v7_used_for_training_or_rules": False,
        "private_v8_used_for_training_or_rules": False,
        "private_v9_used_for_training_or_rules": False,
        "private_v10_used_for_training_or_rules": False,
        "shadow_only": True,
        "automatic_activation": False,
        "automatic_promotion": False,
        "execution_authority": False,
        "allow_grants_permission": False,
        "human_confirmation_bypass": False,
        "prompts_persisted_in_artifact": False,
        "responses_persisted_in_artifact": False,
    }
    artifact["artifact_sha256"] = base.digest(artifact)
    return artifact


def validate_artifact(artifact: dict) -> None:
    if artifact.get("schema") != SCHEMA or artifact.get("kind") != "latent_axis_decision_shadow_policy":
        raise RuntimeError("v4 artifact schema mismatch")
    for key in ("private_v7_used_for_training_or_rules", "private_v8_used_for_training_or_rules", "private_v9_used_for_training_or_rules", "private_v10_used_for_training_or_rules", "automatic_activation", "automatic_promotion", "execution_authority", "allow_grants_permission", "human_confirmation_bypass", "prompts_persisted_in_artifact", "responses_persisted_in_artifact"):
        if artifact.get(key) is not False:
            raise RuntimeError(f"v4 invariant failed: {key}")
    if artifact.get("shadow_only") is not True:
        raise RuntimeError("v4 must remain shadow-only")
    unsigned = dict(artifact); actual = unsigned.pop("artifact_sha256", None)
    if actual != base.digest(unsigned):
        raise RuntimeError("v4 artifact integrity mismatch")


def decode_artifact(artifact: dict) -> dict[str, AxisModel]:
    models = {}
    for axis, labels in AXES.items():
        raw = artifact["models"][axis]
        if tuple(raw["labels"]) != labels:
            raise RuntimeError(f"v4 axis labels changed: {axis}")
        scales = [float(value) for value in raw["weight_scales"]]
        packed = base64.b64decode(raw["weights_qint16_b64"], validate=True)
        count = len(labels) * DIMENSION
        if len(scales) != len(labels) or len(packed) != count * 2:
            raise RuntimeError(f"v4 quantized shape mismatch: {axis}")
        ints = struct.unpack(f"<{count}h", packed)
        weights = []
        offset = 0
        for scale in scales:
            weights.append([ints[offset + i] * scale for i in range(DIMENSION)])
            offset += DIMENSION
        models[axis] = AxisModel(axis, labels, weights, [float(x) for x in raw["biases"]], float(raw["temperature"]), tuple(raw["known_token_sha256"]))
    return models


def gate_result(result: dict, *, legacy: bool = False) -> bool:
    min_accuracy = 0.925 if legacy else 0.965
    min_critical = 0.95 if legacy else 0.975
    min_label = 0.85 if legacy else 0.925
    return bool(
        result["accuracy"] >= min_accuracy
        and result["critical_accuracy"] >= min_critical
        and result["dangerous_false_allow"] == 0
        and result["abstain_rate"] <= 0.30
        and min(result["per_label_recall"].values()) >= min_label
        and all(result["predicted_counts"][label] > 0 for label in data.DECISIONS)
    )


def run() -> tuple[dict, dict, bool]:
    models = train_models()
    artifact = build_artifact(models)
    validate_artifact(artifact)
    qmodels = decode_artifact(artifact)

    suites = data.public_suites()
    float_results = {name: decision_metrics_axis_rows(rows, lambda text: predict(models, text)) for name, rows in suites.items()}
    q_results = {name: decision_metrics_axis_rows(rows, lambda text: predict(qmodels, text)) for name, rows in suites.items()}
    axis_results = {name: axis_metrics(rows, qmodels) for name, rows in suites.items()}
    legacy = base.public_examples(base.PUBLIC)
    legacy_float = decision_metrics_legacy(legacy, lambda text: predict(models, text))
    legacy_q = decision_metrics_legacy(legacy, lambda text: predict(qmodels, text))

    # Determinism check from an independently trained copy.
    artifact2 = build_artifact(train_models())
    deterministic = artifact2["artifact_sha256"] == artifact["artifact_sha256"]
    q_delta_ok = all(abs(float_results[name]["accuracy"] - q_results[name]["accuracy"]) <= (1 / len(suites[name])) for name in suites) and abs(legacy_float["accuracy"] - legacy_q["accuracy"]) <= 0.025
    axes_good = all(all(axis_doc["accuracy"] >= 0.95 and min(axis_doc["per_label_recall"].values()) >= 0.90 for axis_doc in suite.values()) for suite in axis_results.values())
    public_good = all(gate_result(result) for result in q_results.values())
    legacy_good = gate_result(legacy_q, legacy=True)
    eligible = bool(deterministic and q_delta_ok and axes_good and public_good and legacy_good)

    report = {
        "schema": REPORT_SCHEMA,
        "experiment": "latent-axis-decision-policy-v4",
        "artifact_sha256": artifact["artifact_sha256"],
        "train_examples": len(data.train_rows()),
        "calibration_examples": len(data.calibration_rows()),
        "train_sha256": data.digest(data.train_rows()),
        "calibration_sha256": data.digest(data.calibration_rows()),
        "float_suites": float_results,
        "quantized_suites": q_results,
        "axis_suites": axis_results,
        "legacy_public_float": legacy_float,
        "legacy_public_quantized": legacy_q,
        "gates": {
            "deterministic_training": deterministic,
            "quantized_delta_ok": q_delta_ok,
            "axis_models_pass": axes_good,
            "all_axis_decision_suites_pass": public_good,
            "legacy_public_pass": legacy_good,
            "eligible_for_private_v10": eligible,
        },
        "private_v7_used_for_training_or_rules": False,
        "private_v8_used_for_training_or_rules": False,
        "private_v9_used_for_training_or_rules": False,
        "private_v10_evaluated": False,
        "shadow_only": True,
        "automatic_activation": False,
        "automatic_promotion": False,
        "execution_authority": False,
        "allow_grants_permission": False,
        "human_confirmation_bypass": False,
    }
    return artifact, report, eligible


def main() -> int:
    artifact, report, eligible = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision-policy-v4.artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (OUT / "decision-policy-v4-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_sha256": report["artifact_sha256"],
        "eligible_for_private_v10": eligible,
        "gates": report["gates"],
        "quantized_suites": {name: {k: value[k] for k in ("accuracy", "critical_accuracy", "per_label_recall", "dangerous_false_allow", "abstain_rate", "failure_count", "failure_summary")} for name, value in report["quantized_suites"].items()},
        "legacy_public_quantized": {k: report["legacy_public_quantized"][k] for k in ("accuracy", "critical_accuracy", "per_label_recall", "dangerous_false_allow", "abstain_rate", "failure_count", "failure_summary")},
        "automatic_activation": False,
        "automatic_promotion": False,
        "execution_authority": False,
        "allow_grants_permission": False,
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
