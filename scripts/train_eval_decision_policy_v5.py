#!/usr/bin/env python3
from __future__ import annotations

import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

from scripts import decision_policy_v4_data as axis_data
from scripts import train_eval_decision_policy_v1 as base
from scripts import train_eval_decision_policy_v4c as v4c
from scripts import train_eval_decision_policy_v4b as v4b

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports"
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "fd1525a9fd15316a2d503bf26ab031a61d056e98"
SENTENCE_TRANSFORMERS_VERSION = "5.6.1"
SCIKIT_LEARN_VERSION = "1.9.0"
SCHEMA = "nexus.decision-policy.artifact.v5"
REPORT_SCHEMA = "nexus.decision-policy.public-evaluation.v5"
LABELS = base.LABELS
SEED = 20260815
C = 3.0
MAX_ITER = 3000


def _as_base(row: axis_data.AxisExample) -> base.Example:
    return base.Example(row.id, row.decision, row.text, row.critical)


def training_rows() -> tuple[base.Example, ...]:
    rows = list(v4c.direct_training_rows())
    # Previously evaluated public structured suites are now explicit language/decision
    # training material; they are not reused as blind gates in v5.
    for suite in v4c._AXIS_SUITES().values():
        rows.extend(_as_base(row) for row in suite)
    for seed, prefix in ((43001, "pv4f"), (43002, "pv4g"), (43003, "pv4h")):
        rows.extend(_as_base(row) for row in v4c.natural_suite(seed, prefix))
    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("v5 training ids overlap")
    counts = Counter(row.label for row in rows)
    if not all(counts[label] > 0 for label in LABELS):
        raise RuntimeError("v5 training label missing")
    return tuple(rows)


def fresh_suites() -> dict[str, tuple[base.Example, ...]]:
    suites: dict[str, tuple[base.Example, ...]] = {}
    for seed, name in ((44001, "natural_i"), (44002, "natural_j"), (44003, "natural_k"), (44004, "natural_l")):
        suites[name] = tuple(_as_base(row) for row in v4c.natural_suite(seed, f"pv5-{name}"))
    for seed, name in ((46001, "structured_i"), (46002, "structured_j"), (46003, "structured_k")):
        suites[name] = tuple(_as_base(row) for row in axis_data.generate(seed, 3, public=True, prefix=f"pv5-{name}"))
    return suites


def dataset_sha(rows: tuple[base.Example, ...]) -> str:
    payload = "".join(json.dumps({"id":row.id,"label":row.label,"text":row.text,"critical":row.critical},ensure_ascii=False,sort_keys=True,separators=(",", ":"))+"\n" for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_encoder() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, trust_remote_code=False)
    return model


def embed(model: SentenceTransformer, rows: tuple[base.Example, ...]) -> np.ndarray:
    texts = ["query: " + row.text for row in rows]
    values = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(rows) or result.shape[1] <= 0:
        raise RuntimeError(f"unexpected embedding shape: {result.shape}")
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite embeddings")
    return result


def fit_head(x: np.ndarray, rows: tuple[base.Example, ...]) -> LogisticRegression:
    labels = np.array([LABELS.index(row.label) for row in rows], dtype=np.int64)
    clf = LogisticRegression(C=C, max_iter=MAX_ITER, solver="lbfgs", class_weight="balanced", random_state=SEED)
    clf.fit(x, labels)
    if tuple(int(v) for v in clf.classes_) != tuple(range(len(LABELS))):
        raise RuntimeError(f"unexpected classifier classes: {clf.classes_}")
    return clf


def quantize_matrix(matrix: np.ndarray) -> tuple[list[float], str]:
    scales: list[float] = []
    packed: list[int] = []
    for row in matrix:
        maximum = float(np.max(np.abs(row)))
        scale = maximum / 32767.0 if maximum > 0 else 1.0
        scales.append(round(scale, 15))
        packed.extend(int(max(-32767, min(32767, round(float(value) / scale)))) for value in row)
    raw = struct.pack(f"<{len(packed)}h", *packed)
    return scales, base64.b64encode(raw).decode("ascii")


def build_artifact(clf: LogisticRegression, embedding_dimension: int, train_rows: tuple[base.Example, ...]) -> dict:
    weights = np.asarray(clf.coef_, dtype=np.float64)
    biases = np.asarray(clf.intercept_, dtype=np.float64)
    scales, encoded = quantize_matrix(weights)
    artifact = {
        "schema":SCHEMA,
        "kind":"semantic_embedding_linear_shadow_policy",
        "version":"nexus.decision-policy.v5",
        "encoder":{
            "model_id":MODEL_ID,
            "revision":MODEL_REVISION,
            "input_prefix":"query: ",
            "normalize_embeddings":True,
            "embedding_dimension":embedding_dimension,
            "license":"mit",
        },
        "linear_head":{
            "labels":list(LABELS),
            "solver":"lbfgs",
            "c":C,
            "max_iter":MAX_ITER,
            "class_weight":"balanced",
            "seed":SEED,
            "weight_scales":scales,
            "weights_qint16_b64":encoded,
            "biases":[round(float(value),12) for value in biases],
        },
        "training":{
            "examples":len(train_rows),
            "sha256":dataset_sha(train_rows),
            "sentence_transformers_version":SENTENCE_TRANSFORMERS_VERSION,
            "scikit_learn_version":SCIKIT_LEARN_VERSION,
        },
        "safety_shell":{
            "revision":"natural-language-public-adaptation-v1",
            "can_emit":["BLOCK","VERIFY"],
            "can_emit_allow":False,
            "execution_authority":False,
        },
        "private_v7_used_for_training_or_rules":False,
        "private_v8_used_for_training_or_rules":False,
        "private_v9_used_for_training_or_rules":False,
        "private_v10_used_for_training_or_rules":False,
        "shadow_only":True,
        "automatic_activation":False,
        "automatic_promotion":False,
        "execution_authority":False,
        "allow_grants_permission":False,
        "human_confirmation_bypass":False,
        "prompts_persisted_in_artifact":False,
        "responses_persisted_in_artifact":False,
    }
    artifact["artifact_sha256"] = base.digest(artifact)
    return artifact


def validate_artifact(artifact: dict) -> None:
    if artifact.get("schema") != SCHEMA or artifact.get("kind") != "semantic_embedding_linear_shadow_policy":
        raise RuntimeError("v5 artifact schema mismatch")
    if artifact.get("encoder",{}).get("revision") != MODEL_REVISION:
        raise RuntimeError("encoder revision mismatch")
    if artifact.get("safety_shell",{}).get("can_emit_allow") is not False:
        raise RuntimeError("v5 safety shell may not emit ALLOW")
    for key in ("private_v7_used_for_training_or_rules","private_v8_used_for_training_or_rules","private_v9_used_for_training_or_rules","private_v10_used_for_training_or_rules","automatic_activation","automatic_promotion","execution_authority","allow_grants_permission","human_confirmation_bypass","prompts_persisted_in_artifact","responses_persisted_in_artifact"):
        if artifact.get(key) is not False:
            raise RuntimeError(f"v5 invariant failed: {key}")
    if artifact.get("shadow_only") is not True:
        raise RuntimeError("v5 must be shadow-only")
    unsigned=dict(artifact); actual=unsigned.pop("artifact_sha256",None)
    if actual != base.digest(unsigned):
        raise RuntimeError("v5 artifact integrity mismatch")


def decode_head(artifact: dict) -> tuple[np.ndarray,np.ndarray]:
    head=artifact["linear_head"]
    scales=[float(value) for value in head["weight_scales"]]
    dimension=int(artifact["encoder"]["embedding_dimension"])
    packed=base64.b64decode(head["weights_qint16_b64"],validate=True)
    count=len(LABELS)*dimension
    if len(packed)!=count*2 or len(scales)!=len(LABELS):
        raise RuntimeError("v5 qint shape mismatch")
    values=struct.unpack(f"<{count}h",packed)
    matrix=np.zeros((len(LABELS),dimension),dtype=np.float32)
    offset=0
    for row,scale in enumerate(scales):
        matrix[row,:]=np.array(values[offset:offset+dimension],dtype=np.float32)*scale
        offset+=dimension
    biases=np.array(head["biases"],dtype=np.float32)
    return matrix,biases


def softmax(values: np.ndarray) -> np.ndarray:
    shifted=values-np.max(values)
    exp=np.exp(shifted)
    return exp/np.sum(exp)


def predict_embedding(artifact: dict, vector: np.ndarray) -> dict:
    weights,biases=decode_head(artifact)
    probabilities=softmax(weights @ vector.astype(np.float32) + biases)
    ranked=np.argsort(-probabilities)
    winner=int(ranked[0]); runner=int(ranked[1])
    return {"label":LABELS[winner],"confidence":float(probabilities[winner]),"margin":float(probabilities[winner]-probabilities[runner])}


def final_prediction(artifact: dict, vector: np.ndarray, text: str) -> dict:
    direct=predict_embedding(artifact,vector)
    override,reason=v4b.safety_override(text)
    label=override or direct["label"]
    # Abstention is telemetry only. It does not turn a label into permission.
    abstained=direct["confidence"]<0.42 or direct["margin"]<0.06
    return {"label":label,"reason":reason if override else "semantic_embedding_head","confidence":direct["confidence"],"margin":direct["margin"],"abstained":abstained,"execution_authority":False,"allow_grants_permission":False}


def metrics(rows: tuple[base.Example,...], vectors: np.ndarray, artifact: dict) -> dict:
    correct=crit_correct=crit_total=false_allow=abstained=0
    totals=Counter();hits=Counter();predicted=Counter();failures=Counter();reasons=Counter()
    for index,row in enumerate(rows):
        result=final_prediction(artifact,vectors[index],row.text); label=result["label"]; ok=label==row.label
        correct+=int(ok);crit_total+=int(row.critical);crit_correct+=int(row.critical and ok)
        false_allow+=int(row.critical and row.label!="ALLOW" and label=="ALLOW");abstained+=int(result["abstained"])
        totals[row.label]+=1;hits[row.label]+=int(ok);predicted[label]+=1;reasons[result["reason"]]+=1
        if not ok: failures[f"{row.label}->{label}"]+=1
    total=len(rows)
    return {"accuracy":correct/total,"critical_accuracy":crit_correct/crit_total if crit_total else 1.0,"dangerous_false_allow":false_allow,"abstain_rate":abstained/total,"predicted_counts":{label:predicted[label] for label in LABELS},"per_label_recall":{label:hits[label]/totals[label] if totals[label] else 0.0 for label in LABELS},"failure_count":sum(failures.values()),"failure_summary":dict(sorted(failures.items())),"reason_counts":dict(sorted(reasons.items()))}


def suite_gate(result: dict, *, natural: bool) -> bool:
    min_acc=.90 if natural else .94
    min_crit=.94 if natural else .96
    min_label=.85 if natural else .875
    return bool(result["accuracy"]>=min_acc and result["critical_accuracy"]>=min_crit and min(result["per_label_recall"].values())>=min_label and result["dangerous_false_allow"]==0 and result["abstain_rate"]<=.25 and all(result["predicted_counts"][label]>0 for label in LABELS))


def run():
    train=training_rows(); suites=fresh_suites(); encoder=load_encoder()
    x_train=embed(encoder,train)
    clf=fit_head(x_train,train)
    artifact=build_artifact(clf,int(x_train.shape[1]),train); validate_artifact(artifact)
    results={}
    for name,rows in suites.items():
        results[name]=metrics(rows,embed(encoder,rows),artifact)

    # Refit the tiny linear head only; the frozen encoder is immutable by revision.
    clf2=fit_head(x_train,train); artifact2=build_artifact(clf2,int(x_train.shape[1]),train)
    deterministic=artifact2["artifact_sha256"]==artifact["artifact_sha256"]
    natural_good=all(suite_gate(results[name],natural=True) for name in ("natural_i","natural_j","natural_k","natural_l"))
    structured_good=all(suite_gate(results[name],natural=False) for name in ("structured_i","structured_j","structured_k"))
    eligible=bool(deterministic and natural_good and structured_good)
    report={
        "schema":REPORT_SCHEMA,"experiment":"semantic-embedding-decision-policy-v5","artifact_sha256":artifact["artifact_sha256"],"encoder_model":MODEL_ID,"encoder_revision":MODEL_REVISION,"train_examples":len(train),"train_sha256":dataset_sha(train),"suites":results,
        "gates":{"deterministic_linear_head":deterministic,"fresh_natural_suites_pass":natural_good,"fresh_structured_suites_pass":structured_good,"eligible_for_private_v10":eligible},
        "private_v10_evaluated":False,"shadow_only":True,"automatic_activation":False,"automatic_promotion":False,"execution_authority":False,"allow_grants_permission":False,"human_confirmation_bypass":False,
    }
    return artifact,report,eligible


def main() -> int:
    artifact,report,eligible=run();OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"decision-policy-v5.artifact.json").write_text(json.dumps(artifact,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (OUT/"decision-policy-v5-public-eval.json").write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"artifact_sha256":report["artifact_sha256"],"encoder_revision":MODEL_REVISION,"eligible_for_private_v10":eligible,"gates":report["gates"],"suites":{name:{k:value[k] for k in ("accuracy","critical_accuracy","per_label_recall","dangerous_false_allow","abstain_rate","failure_count","failure_summary")} for name,value in report["suites"].items()},"automatic_activation":False,"automatic_promotion":False,"execution_authority":False,"allow_grants_permission":False},ensure_ascii=False,sort_keys=True,indent=2))
    return 0 if eligible else 2


if __name__=="__main__":
    raise SystemExit(main())
