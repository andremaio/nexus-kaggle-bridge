#!/usr/bin/env python3
from __future__ import annotations

from scripts import train_eval_decision_policy_v3 as v3
from scripts import train_eval_decision_policy_v3_runner as runner

_original_override = v3.strong_safety_override
_original_build_artifact = v3.build_artifact


def strong_safety_override(text: str) -> tuple[str | None, str]:
    label, reason = _original_override(text)
    if label is not None:
        return label, reason
    s = v3.fold(text)

    # Data integrity/provenance: derived or imputed values must not be passed off
    # as observations while concealing their provenance.
    derived = any(x in s for x in ("imput", "estimated", "estimad", "synthetic", "sintet", "inferred", "inferid", "derived", "derivad"))
    observed_claim = any(x in s for x in ("observacoes reais", "observações reais", "real observations", "observed values", "valores observados", "as if observed", "como observado"))
    conceal = any(x in s for x in ("sem revelar", "without disclos", "without reveal", "hide", "ocultar", "nao indicar", "não indicar", "undisclosed"))
    if derived and observed_claim and conceal:
        return "BLOCK", "derived_data_must_not_be_misrepresented_as_observed"

    # Forensics/provenance: a technical anomaly without the original/reference is
    # insufficient for a definitive conclusion, but it is verifiable if the
    # missing provenance can be obtained.
    forensic_media = any(x in s for x in ("video", "vídeo", "image", "imagem", "audio", "áudio", "frame", "forensic", "forense", "evidence", "evidencia", "evidência"))
    anomaly = any(x in s for x in ("compression", "compressao", "compressão", "irregular", "artifact", "artefact", "anomaly", "anomalia", "metadata", "metadados"))
    missing_reference = any(x in s for x in ("sem original", "without original", "no original", "without reference", "sem referencia", "sem referência", "para comparacao", "para comparação", "for comparison"))
    if forensic_media and anomaly and missing_reference:
        return "VERIFY", "forensic_anomaly_requires_original_or_reference"

    # Temporal authority: a permission/approval that was valid previously does
    # not prove that it is valid now. Re-check current authority before action.
    authority = any(x in s for x in ("permission", "permissao", "permissão", "approval", "aprovacao", "aprovação", "authorization", "autorizacao", "autorização", "lease"))
    temporary = any(x in s for x in ("temporary", "temporaria", "temporária", "temporario", "temporário", "expires", "expira", "validade", "validity"))
    past = any(x in s for x in ("yesterday", "ontem", "previously", "anteriormente", "was valid", "foi valida", "foi válida", "era valida", "era válida"))
    current_unknown = any(x in s for x in ("today", "hoje", "now", "agora", "current", "actual", "atual", "nao tens informacao", "não tens informação", "unknown", "desconhecid"))
    if authority and (temporary or past) and current_unknown:
        return "VERIFY", "current_authority_must_be_revalidated"

    return None, "learned_policy"


def build_artifact(weights, biases, temperature: float, known, train_count: int) -> dict:
    artifact = _original_build_artifact(weights, biases, temperature, known, train_count)
    artifact["policy"]["version"] = "nexus.decision-policy.v3b"
    artifact["policy"]["guard_revision"] = "integrity-forensic-provenance-temporal-authority-v1"
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = v3.base.digest(artifact)
    return artifact


v3.strong_safety_override = strong_safety_override
v3.build_artifact = build_artifact
v3.fit_candidate = runner.fixed_fit_candidate

if __name__ == "__main__":
    raise SystemExit(v3.main())
