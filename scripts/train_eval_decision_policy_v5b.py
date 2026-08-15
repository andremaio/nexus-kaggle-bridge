#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json

from scripts import decision_policy_v4_data as axis_data
from scripts import train_eval_decision_policy_v1 as base
from scripts import train_eval_decision_policy_v4c as v4c
from scripts import train_eval_decision_policy_v5 as v5

OUT = v5.OUT
SCHEMA = "nexus.decision-policy.artifact.v5b"
REPORT_SCHEMA = "nexus.decision-policy.public-evaluation.v5b"


def _as_base(row: axis_data.AxisExample) -> base.Example:
    return base.Example(row.id, row.decision, row.text, row.critical)


def expanded_training_rows() -> tuple[base.Example, ...]:
    rows = list(v5.training_rows())
    # The v5 structured_i/j/k suites are consumed public evidence after run 31884355163.
    # They may train v5b, but are never reused as blind gates for v5b.
    for seed, name in ((46001, "structured_i"), (46002, "structured_j"), (46003, "structured_k")):
        rows.extend(_as_base(row) for row in axis_data.generate(seed, 3, public=True, prefix=f"pv5-{name}"))
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("v5b training ids overlap")
    return tuple(rows)


def fresh_suites_v5b() -> dict[str, tuple[base.Example, ...]]:
    suites: dict[str, tuple[base.Example, ...]] = {}
    for seed, name in ((45001, "natural_m"), (45002, "natural_n"), (45003, "natural_o"), (45004, "natural_p")):
        suites[name] = tuple(_as_base(row) for row in v4c.natural_suite(seed, f"pv5b-{name}"))
    for seed, name in ((47001, "structured_m"), (47002, "structured_n"), (47003, "structured_o"), (47004, "structured_p")):
        suites[name] = tuple(_as_base(row) for row in axis_data.generate(seed, 3, public=True, prefix=f"pv5b-{name}"))
    return suites


def run():
    train = expanded_training_rows()
    suites = fresh_suites_v5b()
    encoder = v5.load_encoder()
    x_train = v5.embed(encoder, train)
    clf = v5.fit_head(x_train, train)
    artifact = v5.build_artifact(clf, int(x_train.shape[1]), train)
    artifact["schema"] = SCHEMA
    artifact["version"] = "nexus.decision-policy.v5b"
    artifact["training"]["parent_candidate"] = "nexus.decision-policy.v5"
    artifact["training"]["parent_public_run_id"] = 31884355163
    artifact["training"]["consumed_structured_suites_added"] = ["structured_i", "structured_j", "structured_k"]
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = base.digest(artifact)

    results = {name: v5.metrics(rows, v5.embed(encoder, rows), artifact) for name, rows in suites.items()}

    # Determinism of the linear head/artifact with the encoder fixed by immutable revision.
    clf2 = v5.fit_head(x_train, train)
    artifact2 = v5.build_artifact(clf2, int(x_train.shape[1]), train)
    artifact2["schema"] = SCHEMA
    artifact2["version"] = "nexus.decision-policy.v5b"
    artifact2["training"]["parent_candidate"] = "nexus.decision-policy.v5"
    artifact2["training"]["parent_public_run_id"] = 31884355163
    artifact2["training"]["consumed_structured_suites_added"] = ["structured_i", "structured_j", "structured_k"]
    artifact2.pop("artifact_sha256", None)
    artifact2["artifact_sha256"] = base.digest(artifact2)
    deterministic = artifact2["artifact_sha256"] == artifact["artifact_sha256"]

    natural_names = ("natural_m", "natural_n", "natural_o", "natural_p")
    structured_names = ("structured_m", "structured_n", "structured_o", "structured_p")
    natural_good = all(v5.suite_gate(results[name], natural=True) for name in natural_names)
    structured_good = all(v5.suite_gate(results[name], natural=False) for name in structured_names)
    eligible = bool(deterministic and natural_good and structured_good)

    report = {
        "schema": REPORT_SCHEMA,
        "experiment": "semantic-embedding-decision-policy-v5b",
        "artifact_sha256": artifact["artifact_sha256"],
        "encoder_model": v5.MODEL_ID,
        "encoder_revision": v5.MODEL_REVISION,
        "train_examples": len(train),
        "train_sha256": v5.dataset_sha(train),
        "suites": results,
        "gates": {
            "deterministic_linear_head": deterministic,
            "fresh_natural_suites_pass": natural_good,
            "fresh_structured_suites_pass": structured_good,
            "eligible_for_private_v10": eligible,
        },
        "private_v10_evaluated": False,
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
    }
    return artifact, report, eligible


def main() -> int:
    artifact, report, eligible = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "decision-policy-v5b.artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (OUT / "decision-policy-v5b-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact_sha256": report["artifact_sha256"],
        "encoder_revision": v5.MODEL_REVISION,
        "eligible_for_private_v10": eligible,
        "gates": report["gates"],
        "suites": {name: {k: value[k] for k in ("accuracy", "critical_accuracy", "per_label_recall", "dangerous_false_allow", "abstain_rate", "failure_count", "failure_summary")} for name, value in report["suites"].items()},
        "automatic_activation": False,
        "automatic_promotion": False,
        "execution_authority": False,
        "allow_grants_permission": False,
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
