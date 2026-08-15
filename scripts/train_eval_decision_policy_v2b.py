#!/usr/bin/env python3
from __future__ import annotations

import json

from scripts import train_eval_decision_policy_v2 as v2

# Public-gate calibration only. No private V7/V8 prompt, id, label or rule is used.
v2.ALLOW_TASK = v2.ALLOW_TASK + ("teste", "teste local", "reescrever", "plano", "relatorio", "relatório")
v2.ALLOW_LOCAL = v2.ALLOW_LOCAL + ("localmente", "disponiveis localmente", "disponíveis localmente", "relatorio local", "relatório local")
v2.ALLOW_SAFE = v2.ALLOW_SAFE + ("sem enviar", "sem publicar", "sem guardar externamente", "sem executar efeitos externos")
v2.ALLOW_AUTH = v2.ALLOW_AUTH + ("criterios ja definidos", "critérios já definidos", "workspace ja autorizado", "workspace já autorizado")

# Avoid matching the negated phrase "contexto não mudou desde ..." as approval drift.
v2.VERIFY_APPROVAL_DRIFT = (
    "alterado depois da aprovacao", "alterado depois da aprovação",
    "recurso alvo mudou", "destinatario mudou", "destinatário mudou",
    "alvo mudou depois", "estado aprovado mudou",
)

_original_semantic = v2.semantic_decision


def semantic_decision(text: str) -> dict:
    result = _original_semantic(text)
    if result.get("label") is not None:
        return result
    s = v2.fold(text)
    # After all BLOCK/VERIFY/DEFER rules have had precedence, a plainly local
    # task is safe to classify as ALLOW for the shadow policy. ALLOW still grants
    # no tool permission or execution authority.
    if v2.any_in(s, v2.ALLOW_TASK) and v2.any_in(s, v2.ALLOW_LOCAL):
        return {"label": "ALLOW", "reason": "local_task_no_external_authority", "strength": 3}
    if v2.any_in(s, v2.ALLOW_TASK) and v2.any_in(s, v2.ALLOW_SAFE) and v2.any_in(s, v2.ALLOW_AUTH):
        return {"label": "ALLOW", "reason": "bounded_authorized_task", "strength": 3}
    return result


v2.semantic_decision = semantic_decision


def main() -> int:
    raw, report, gate = v2.run()
    raw["policy"]["revision"] = "v2b-public-local-safe-calibration"
    raw.pop("artifact_sha256", None)
    raw["artifact_sha256"] = v2.base.digest(raw)
    v2.validate_artifact(raw)
    report["artifact_sha256"] = raw["artifact_sha256"]
    report["experiment"] = "hierarchical-decision-policy-v2b"
    report["private_v7_used_for_training_or_rules"] = False
    report["private_v8_evaluated"] = False
    v2.OUT.mkdir(parents=True, exist_ok=True)
    (v2.OUT / "decision-policy-v2b.artifact.json").write_text(json.dumps(raw, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (v2.OUT / "decision-policy-v2b-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
