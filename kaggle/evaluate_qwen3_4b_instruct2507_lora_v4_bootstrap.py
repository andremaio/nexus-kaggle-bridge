#!/usr/bin/env python3
from __future__ import annotations
import base64,json,os,runpy,sys,urllib.request
from pathlib import Path
SCRIPT_COMMIT='73769cde7162f1e40727219469212deef1a34c7c'
SCRIPT_BLOB_SHA='ece21d6ec3dcc4cffade94b291fefd69c426d192'
API=f'https://api.github.com/repos/andremaio/nexus-kaggle-bridge/contents/kaggle/evaluate_qwen3_4b_instruct2507_lora_v4.py?ref={SCRIPT_COMMIT}'
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0'; os.environ['HF_HOME']='/kaggle/temp/hf-cache-qwen4b-v4-eval'; os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1'); os.environ.setdefault('DO_NOT_TRACK','1'); os.environ.setdefault('WANDB_DISABLED','true')
    req=urllib.request.Request(API,headers={'Accept':'application/vnd.github+json','User-Agent':'nexus-qwen4b-v4-bootstrap'})
    with urllib.request.urlopen(req,timeout=60) as response: payload=json.loads(response.read().decode('utf-8'))
    if payload.get('sha')!=SCRIPT_BLOB_SHA or payload.get('encoding')!='base64': raise RuntimeError('v4 evaluator integrity mismatch')
    source=base64.b64decode(payload['content'],validate=False).decode('utf-8'); target=Path('/kaggle/working/evaluate_qwen3_4b_instruct2507_lora_v4.py'); target.write_text(source,encoding='utf-8'); compile(source,str(target),'exec'); print(f'NEXUS_QWEN4B_V4_BOOTSTRAP {SCRIPT_COMMIT}:{SCRIPT_BLOB_SHA}',flush=True); runpy.run_path(str(target),run_name='__main__')
if __name__=='__main__': main()
