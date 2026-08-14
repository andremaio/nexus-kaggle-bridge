#!/usr/bin/env python3
from __future__ import annotations
import base64, importlib.metadata, json, os, runpy, subprocess, sys, urllib.request
from pathlib import Path
PROBE_COMMIT='32d2ea642f159688cc986d6ab639ddde02123e33'
PROBE_BLOB_SHA='5480ab7fc6f8d457a7e030a1bdcf09e01d22b1a3'
PROBE_API=f'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/kaggle/probe_qwen3_4b_instruct2507.py?ref={PROBE_COMMIT}'
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['HF_HOME']='/kaggle/temp/hf-cache-instruct2507'; os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1'); os.environ.setdefault('DO_NOT_TRACK','1'); os.environ.setdefault('WANDB_DISABLED','true')
    try: importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError: pass
    else: subprocess.run([sys.executable,'-m','pip','uninstall','-y','torchao'],check=False)
    req=urllib.request.Request(PROBE_API,headers={'Accept':'application/vnd.github+json','User-Agent':'nexus-instruct2507-bootstrap'})
    with urllib.request.urlopen(req,timeout=60) as r: p=json.loads(r.read().decode())
    if p.get('sha')!=PROBE_BLOB_SHA or p.get('encoding')!='base64': raise RuntimeError('probe integrity mismatch')
    b=base64.b64decode(p['content'],validate=False); target=Path('/kaggle/working/probe_qwen3_4b_instruct2507.py'); target.write_bytes(b); print(f'NEXUS_INSTRUCT2507_BOOTSTRAP commit={PROBE_COMMIT} blob={PROBE_BLOB_SHA} bytes={len(b)}',flush=True); runpy.run_path(str(target),run_name='__main__')
if __name__=='__main__': main()
