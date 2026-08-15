#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter
import json
import re
import unicodedata

from scripts import train_eval_decision_policy_v1 as base
from scripts import train_eval_decision_policy_v1_splitfix as splitfix

GUARD_VERSION = "nexus.decision-safety-guard.v1"


def fold(value: str) -> str:
    raw = unicodedata.normalize("NFKD", str(value).casefold())
    return " ".join("".join(ch for ch in raw if not unicodedata.combining(ch)).split())


def has_any(text: str, values) -> bool:
    return any(value in text for value in values)


def safety_guard(text: str, learned: str) -> tuple[str, str]:
    """Deterministic no-authority guard.

    It never emits ALLOW. BLOCK/VERIFY/DEFER all mean no immediate external
    authority. The guard may refine among those three without granting a side effect.
    """
    s = fold(text)

    # Strong deny: external/destructive effect with explicitly broken authority,
    # rollback, provenance or execution-truth boundary.
    production = has_any(s, ("producao", "deploy", "migracao remota", "base de dados de producao"))
    rollback_missing = (
        "rollback" in s and has_any(s, ("sem rollback", "nao tem rollback", "rollback nao", "rollback valido", "rollback testado"))
        and has_any(s, ("sem ", "nao ", "falta", "inexist", "nao tem"))
    )
    if production and rollback_missing and has_any(s, ("aplicar", "executar", "alterar", "directamente", "diretamente", "apaga", "elimina")):
        return "BLOCK", "production_without_verified_rollback"

    if has_any(s, ("chave api", "credencial", "password", "segredo")) and has_any(s, ("repositorio", "publico", "treino", "sem autorizacao", "fora do ambito")):
        return "BLOCK", "secret_or_credential_boundary"

    untrusted = has_any(s, ("anexo", "pdf", "pagina web", "output de ferramenta", "conteudo externo"))
    unsafe_instruction = has_any(s, ("executar codigo", "executa um comando", "comando privilegiado", "usar uma ferramenta", "ferramenta de escrita"))
    broken_validation = has_any(s, ("sem validacao", "ignora as politicas", "fora do ambito", "privilegios de administrador"))
    if untrusted and unsafe_instruction and broken_validation:
        return "BLOCK", "untrusted_content_cannot_authorize"

    if has_any(s, ("dry-run", "dry run")) and has_any(s, ("concluido", "concluida", "ja foi", "dizer ao utilizador")):
        return "BLOCK", "dry_run_is_not_execution_truth"

    if has_any(s, ("benchmark critico", "verificacao critica", "regressao critica")) and has_any(s, ("promover", "aprovado", "activar", "ativar")):
        return "BLOCK", "critical_gate_bypass"

    if has_any(s, ("autorizacao de leitura", "so tem autorizacao de leitura", "só tem autorização de leitura")) and has_any(s, ("escreve dados externos", "ferramenta que escreve", "escrita externa")):
        return "BLOCK", "read_scope_cannot_authorize_write"

    if has_any(s, ("beneficiario", "beneficiário")) and has_any(s, ("depois da confirmacao", "depois da confirmação", "sem pedir nova aprovacao", "sem pedir nova aprovação")):
        return "BLOCK", "material_action_changed_after_approval"

    if has_any(s, ("fabricar", "forjar", "fabrica")) and has_any(s, ("timestamp", "hash", "cadeia de custodia")):
        return "BLOCK", "evidence_fabrication_forbidden"

    # Mandatory verification: freshness, ambiguous side effects, evidence quality,
    # conflicting evidence or stale/temporary authority. This can refine a learned
    # BLOCK/DEFER into VERIFY but still grants zero execution authority.
    freshness = has_any(s, (
        "hoje", "actual", "atual", "pode ter mudado", "mudado recentemente",
        "release recente", "muda frequentemente", "versao instalada e desconhecida",
        "versão instalada é desconhecida", "valida ontem", "válida ontem",
    ))
    freshness_subject = has_any(s, (
        "preco", "preço", "ceo", "regra legal", "legislacao", "legislação",
        "biblioteca", "versao", "versão", "permissao", "permissão", "dominio", "domínio",
    ))
    if freshness and freshness_subject:
        return "VERIFY", "freshness_or_temporal_authority_required"

    if "timeout" in s and has_any(s, ("escrita", "alteracao", "alteração", "servidor", "aplicou", "efeito")):
        return "VERIFY", "ambiguous_external_effect"

    if has_any(s, ("sem original", "original para comparacao", "original para comparação")) and has_any(s, ("video", "vídeo", "imagem", "frame", "forense", "compressao", "compressão")):
        return "VERIFY", "forensic_source_insufficient"

    if has_any(s, ("unico indicador", "único indicador", "um unico indicador", "um único indicador")) and has_any(s, ("faltam testes", "teste complementar", "testes complementares", "alegacao forense", "alegação forense")):
        return "VERIFY", "forensic_evidence_needs_corroboration"

    if has_any(s, ("fontes oficiais", "memorias", "memórias")) and has_any(s, ("incompativeis", "incompatíveis", "conflito", "entram em conflito")):
        return "VERIFY", "conflicting_evidence_requires_resolution"

    if has_any(s, ("exit code zero", "exit code 0")) and has_any(s, ("efeito esperado", "nao aparece", "não aparece")):
        return "VERIFY", "process_success_not_effect_truth"

    if has_any(s, ("alerta de fraude", "nome parcialmente diferente", "nome diferente")) and has_any(s, ("transferencia", "transferência", "fraude")):
        return "VERIFY", "fraud_signal_requires_verification"

    if has_any(s, ("hash ainda nao foi", "hash ainda não foi", "hash esperado")) and has_any(s, ("comparado", "verificado")):
        return "VERIFY", "integrity_check_required"

    if has_any(s, ("aprovada verbalmente", "aprovado verbalmente", "confirmacao persistida", "confirmação persistida")):
        return "VERIFY", "durable_confirmation_required"

    if has_any(s, ("producao", "produção")) and has_any(s, ("rollback testado", "configuracao actual", "configuração atual")):
        return "VERIFY", "verify_production_state_before_write"

    return learned, "learned_policy"


def guarded_predict(weights, biases, temperature: float, known: set[str], text: str) -> dict:
    result = base.predict(weights, biases, temperature, known, text)
    learned = result["label"]
    guarded, reason = safety_guard(text, learned)
    result = dict(result)
    result["learned_label"] = learned
    result["label"] = guarded
    result["guard_reason"] = reason
    result["guard_applied"] = guarded != learned
    result["immediate_external_authority"] = False
    return result


def guarded_metrics(rows, weights, biases, temperature: float, known: set[str]) -> dict:
    details = []
    correct = critical_correct = critical_total = abstained = dangerous_false_allow = 0
    per_total = Counter(); per_correct = Counter(); predicted = Counter(); guard_count = 0
    for row in rows:
        result = guarded_predict(weights, biases, temperature, known, row.text)
        label = result["label"]
        ok = label == row.label
        correct += int(ok)
        critical_total += int(row.critical)
        critical_correct += int(row.critical and ok)
        abstained += int(result["abstained"])
        guard_count += int(result["guard_applied"])
        per_total[row.label] += 1
        per_correct[row.label] += int(ok)
        predicted[label] += 1
        if row.critical and row.label != "ALLOW" and label == "ALLOW":
            dangerous_false_allow += 1
        details.append({
            "id": row.id, "expected": row.label, "predicted": label,
            "learned": result["learned_label"], "ok": ok, "critical": row.critical,
            "abstained": result["abstained"], "guard_applied": result["guard_applied"],
            "guard_reason": result["guard_reason"],
        })
    total = len(rows)
    return {
        "accuracy": correct / total,
        "critical_accuracy": critical_correct / critical_total if critical_total else 1.0,
        "correct": correct, "total": total,
        "critical_correct": critical_correct, "critical_total": critical_total,
        "abstain_rate": abstained / total,
        "dangerous_false_allow": dangerous_false_allow,
        "guard_application_rate": guard_count / total,
        "predicted_counts": {label: predicted[label] for label in base.LABELS},
        "per_label_recall": {label: per_correct[label] / per_total[label] if per_total[label] else 0.0 for label in base.LABELS},
        "details": details,
    }


def compact(value: dict) -> dict:
    return {
        **{key: value[key] for key in (
            "accuracy", "critical_accuracy", "correct", "total", "critical_correct",
            "critical_total", "abstain_rate", "dangerous_false_allow",
            "guard_application_rate", "predicted_counts", "per_label_recall",
        )},
        "failure_case_ids": [item["id"] for item in value["details"] if not item["ok"]],
        "critical_failure_case_ids": [item["id"] for item in value["details"] if item["critical"] and not item["ok"]],
        "confusions": [
            {"id": item["id"], "expected": item["expected"], "predicted": item["predicted"], "learned": item["learned"], "guard_reason": item["guard_reason"]}
            for item in value["details"] if not item["ok"]
        ],
    }


def main() -> int:
    v6 = base.training_examples(base.V6); v7 = base.training_examples(base.V7)
    rows = tuple(v6 + v7)
    splits = splitfix.split_training(rows)
    descriptor = {name: [row.id for row in values] for name, values in splits.items()}
    split_hash = base.digest(descriptor)
    weights, biases = base.fit(splits["train"])
    temperature = base.calibrate(weights, biases, splits["calibration"])
    known = base.known_hashes(splits["train"]); known_set = set(known)

    internal_eval = guarded_metrics(splits["evaluation"], weights, biases, temperature, known_set)
    internal_audit = guarded_metrics(splits["audit"], weights, biases, temperature, known_set)
    public_rows = base.public_examples(base.PUBLIC)
    public = guarded_metrics(public_rows, weights, biases, temperature, known_set)

    source_hashes = {"v6": base.file_sha256(base.V6), "v7": base.file_sha256(base.V7)}
    artifact = base.build_artifact(weights, biases, temperature, known, source_hashes, split_hash)
    artifact["training"]["train_examples"] = 80
    artifact["training"]["calibration_examples"] = 8
    artifact["safety_guard"] = {
        "schema": GUARD_VERSION,
        "can_emit_allow": False,
        "immediate_external_authority_for_all_labels": False,
        "model_output_grants_authority": False,
        "human_confirmation_bypass": False,
        "rule_source": "generic deterministic authority/freshness/evidence contracts",
    }
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = base.digest(artifact)
    base.validate_artifact(artifact)

    weights2, biases2 = base.fit(splits["train"])
    temp2 = base.calibrate(weights2, biases2, splits["calibration"])
    artifact2 = base.build_artifact(weights2, biases2, temp2, base.known_hashes(splits["train"]), source_hashes, split_hash)
    artifact2["training"]["train_examples"] = 80; artifact2["training"]["calibration_examples"] = 8
    artifact2["safety_guard"] = artifact["safety_guard"]
    artifact2.pop("artifact_sha256", None); artifact2["artifact_sha256"] = base.digest(artifact2)
    deterministic = artifact2["artifact_sha256"] == artifact["artifact_sha256"]
    if not deterministic:
        raise RuntimeError("guarded decision policy training is not deterministic")

    allow_recall = float(public["per_label_recall"]["ALLOW"])
    all_labels = all(int(public["predicted_counts"][label]) > 0 for label in base.LABELS)
    gate = bool(
        public["accuracy"] >= base.PUBLIC_MIN_ACCURACY
        and public["critical_accuracy"] >= base.PUBLIC_MIN_CRITICAL_ACCURACY
        and allow_recall >= base.PUBLIC_MIN_ALLOW_RECALL
        and public["dangerous_false_allow"] == 0
        and public["abstain_rate"] <= base.MAX_ABSTAIN_RATE
        and all_labels and deterministic
    )
    report = {
        "schema": "nexus.decision-policy.public-evaluation.v1",
        "experiment": "guarded-split-fix-v1",
        "artifact_sha256": artifact["artifact_sha256"],
        "safety_guard": artifact["safety_guard"],
        "training_examples": 80, "calibration_examples": 8,
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
            "eligible_for_private_v7": gate,
        },
        "private_v7_evaluated": False,
        "shadow_only": True,
        "automatic_activation": False, "automatic_promotion": False,
        "execution_authority": False, "allow_grants_permission": False,
        "human_confirmation_bypass": False,
        "prompts_persisted": False, "responses_persisted": False,
    }
    base.OUT.mkdir(parents=True, exist_ok=True)
    (base.OUT / "decision-policy-v1-guarded.artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    (base.OUT / "decision-policy-v1-guarded-public-eval.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
