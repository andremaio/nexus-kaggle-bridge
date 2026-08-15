#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from scripts import train_eval_decision_policy_v1 as base


def split_training(rows):
    by_source_label = {}
    for row in rows:
        source = "v7" if row.id.startswith("v7-") else "v6" if row.id.startswith("v6-") else "unknown"
        if source == "unknown":
            raise RuntimeError(f"unexpected source id: {row.id}")
        by_source_label.setdefault((source, row.label), []).append(row)
    parts = {name: [] for name in ("train", "calibration", "evaluation", "audit")}
    for label in base.LABELS:
        old = sorted(by_source_label.get(("v6", label), []), key=lambda item: item.id)
        new = sorted(by_source_label.get(("v7", label), []), key=lambda item: item.id)
        if len(old) != 16 or len(new) != 8:
            raise RuntimeError(f"unexpected source balance for {label}: v6={len(old)} v7={len(new)}")
        # All new v7 boundary examples belong to training. Public holdout remains external.
        parts["train"].extend(old[:12] + new)
        parts["calibration"].extend(old[12:14])
        parts["evaluation"].append(old[14])
        parts["audit"].append(old[15])
    result = {key: tuple(value) for key, value in parts.items()}
    expected = {"train": 80, "calibration": 8, "evaluation": 4, "audit": 4}
    if {key: len(value) for key, value in result.items()} != expected:
        raise RuntimeError("decision split-fix contract failed")
    all_ids = [item.id for group in result.values() for item in group]
    if len(all_ids) != 96 or len(set(all_ids)) != 96:
        raise RuntimeError("decision split-fix overlap or omission")
    return result


def compact(value: dict) -> dict:
    result = {key: value[key] for key in (
        "accuracy", "critical_accuracy", "correct", "total",
        "critical_correct", "critical_total", "abstain_rate",
        "dangerous_false_allow", "predicted_counts", "per_label_recall",
    )}
    result["failure_case_ids"] = [item["id"] for item in value["details"] if not item["ok"]]
    result["critical_failure_case_ids"] = [
        item["id"] for item in value["details"] if item["critical"] and not item["ok"]
    ]
    result["confusions"] = [
        {"id": item["id"], "expected": item["expected"], "predicted": item["predicted"], "abstained": item["abstained"]}
        for item in value["details"] if not item["ok"]
    ]
    return result


def main() -> int:
    v6 = base.training_examples(base.V6)
    v7 = base.training_examples(base.V7)
    all_rows = tuple(v6 + v7)
    if len(v6) != 64 or len(v7) != 32:
        raise RuntimeError("decision source counts changed")
    if Counter(row.label for row in all_rows) != Counter({label: 24 for label in base.LABELS}):
        raise RuntimeError("decision source labels are not balanced")
    splits = split_training(all_rows)
    descriptor = {name: [row.id for row in values] for name, values in splits.items()}
    split_hash = base.digest(descriptor)

    weights, biases = base.fit(splits["train"])
    temperature = base.calibrate(weights, biases, splits["calibration"])
    known = base.known_hashes(splits["train"])
    known_set = set(known)
    internal_eval = base.metrics(splits["evaluation"], weights, biases, temperature, known_set)
    internal_audit = base.metrics(splits["audit"], weights, biases, temperature, known_set)
    public_rows = base.public_examples(base.PUBLIC)
    public = base.metrics(public_rows, weights, biases, temperature, known_set)

    source_hashes = {"v6": base.file_sha256(base.V6), "v7": base.file_sha256(base.V7)}
    artifact = base.build_artifact(weights, biases, temperature, known, source_hashes, split_hash)
    artifact["training"]["train_examples"] = 80
    artifact["training"]["calibration_examples"] = 8
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = base.digest(artifact)
    base.validate_artifact(artifact)

    weights2, biases2 = base.fit(splits["train"])
    temperature2 = base.calibrate(weights2, biases2, splits["calibration"])
    artifact2 = base.build_artifact(weights2, biases2, temperature2, base.known_hashes(splits["train"]), source_hashes, split_hash)
    artifact2["training"]["train_examples"] = 80
    artifact2["training"]["calibration_examples"] = 8
    artifact2.pop("artifact_sha256", None)
    artifact2["artifact_sha256"] = base.digest(artifact2)
    deterministic = artifact2["artifact_sha256"] == artifact["artifact_sha256"]
    if not deterministic:
        raise RuntimeError("decision split-fix training is not deterministic")

    allow_recall = float(public["per_label_recall"]["ALLOW"])
    all_labels = all(int(public["predicted_counts"][label]) > 0 for label in base.LABELS)
    public_gate = bool(
        public["accuracy"] >= base.PUBLIC_MIN_ACCURACY
        and public["critical_accuracy"] >= base.PUBLIC_MIN_CRITICAL_ACCURACY
        and allow_recall >= base.PUBLIC_MIN_ALLOW_RECALL
        and public["dangerous_false_allow"] == 0
        and public["abstain_rate"] <= base.MAX_ABSTAIN_RATE
        and all_labels
    )
    report = {
        "schema": "nexus.decision-policy.public-evaluation.v1",
        "experiment": "split-fix-all-v7-in-training",
        "artifact_sha256": artifact["artifact_sha256"],
        "source_sha256": {**source_hashes, "public_holdout": base.file_sha256(base.PUBLIC)},
        "split_sha256": split_hash,
        "training_examples": 80,
        "calibration_examples": 8,
        "internal_evaluation": compact(internal_eval),
        "internal_audit": compact(internal_audit),
        "public_holdout": compact(public),
        "gates": {
            "minimum_accuracy": base.PUBLIC_MIN_ACCURACY,
            "minimum_critical_accuracy": base.PUBLIC_MIN_CRITICAL_ACCURACY,
            "minimum_allow_recall": base.PUBLIC_MIN_ALLOW_RECALL,
            "maximum_abstain_rate": base.MAX_ABSTAIN_RATE,
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
    base.OUT.mkdir(parents=True, exist_ok=True)
    (base.OUT / "decision-policy-v1-splitfix.artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (base.OUT / "decision-policy-v1-splitfix-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if public_gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
