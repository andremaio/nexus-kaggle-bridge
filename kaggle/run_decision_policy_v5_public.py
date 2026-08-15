#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

SOURCE_COMMIT = "2e9bea96d47bc09f940ed679e97507ed23f5e42b"
FILES = (
    "scripts/train_eval_decision_policy_v1.py",
    "scripts/decision_policy_v4_data.py",
    "scripts/train_eval_decision_policy_v4.py",
    "scripts/train_eval_decision_policy_v4b.py",
    "scripts/train_eval_decision_policy_v4c.py",
    "scripts/train_eval_decision_policy_v5.py",
    "training/seed_sft_v6_decision_balance.jsonl",
    "training/seed_sft_v7_decision_boundary.jsonl",
    "training/decision_holdout_v1.jsonl",
)


def download(path: str, root: Path) -> None:
    url = f"https://raw.githubusercontent.com/andremaio/nexus-kaggle-bridge/{SOURCE_COMMIT}/{path}"
    request = urllib.request.Request(url, headers={"User-Agent":"nexus-decision-v5-kaggle"})
    with urllib.request.urlopen(request, timeout=90) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty source: {path}")
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def main() -> int:
    root = Path("/kaggle/working/nexus-v5-source")
    root.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        download(path, root)
    (root / "scripts/__init__.py").write_text("", encoding="utf-8")

    subprocess.check_call([
        sys.executable,"-m","pip","install","--disable-pip-version-check","-q",
        "sentence-transformers==5.6.1","scikit-learn==1.9.0",
    ])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root)
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    process = subprocess.run([sys.executable,"scripts/train_eval_decision_policy_v5.py"],cwd=root,env=env,check=False)
    for name in ("decision-policy-v5.artifact.json","decision-policy-v5-public-eval.json"):
        source=root/"reports"/name
        if not source.exists():
            raise RuntimeError(f"missing v5 output: {name}")
        shutil.copy2(source,Path("/kaggle/working")/name)
    Path("/kaggle/working/v5-exit-code.txt").write_text(str(process.returncode)+"\n",encoding="utf-8")
    print(f"NEXUS_DECISION_POLICY_V5_PUBLIC_COMPLETE source_commit={SOURCE_COMMIT} exit={process.returncode}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
