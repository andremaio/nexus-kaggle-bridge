#!/usr/bin/env python3
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import random
import time
import zipfile

from v6_common import *

ADAPTER = OUT / 'nexus-adapter-v6'
BUNDLE = OUT / 'nexus-candidate-v6.zip'


def compact(score: dict) -> dict:
    return {k: score[k] for k in ['score','safety_score','format_score','passed','total','critical_passed','critical_total','forbidden_hits'] if k in score}


def main() -> None:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU is required')
    if torch.cuda.device_count()!=1: raise RuntimeError(f'v6 requires exactly one visible GPU, got {torch.cuda.device_count()}')
    gpu=torch.cuda.get_device_name(0); print('NEXUS_V6_TRAINING_START',gpu,flush=True)

    repo_paths = OLD_SEEDS + V6_SEEDS + EVAL_FILES + SPEC_FILES
    files={p:download(p) for p in repo_paths}
    old_rows=[]
    for p in OLD_SEEDS: old_rows += jsonl(files[p])
    v6_rows=[]
    for p in V6_SEEDS: v6_rows += jsonl(files[p])
    train_rows=old_rows+v6_rows
    fixed=jsonl(files['training/benchmark_fixed_v1.jsonl']); adv=jsonl(files['training/benchmark_adversarial_v1.jsonl'])
    dev=jsonl(files['training/holdout_v2.jsonl']); primary=jsonl(files['training/holdout_v3.jsonl'])
    system_prompt=files['training/system_prompt_v2.txt'].read_text(encoding='utf-8').strip()
    spec=json.loads(files['training/evaluator_v2_spec.json'].read_text(encoding='utf-8'))
    if not spec.get('frozen_before_v6_training') or spec.get('primary_holdout')!='training/holdout_v3.jsonl':
        raise RuntimeError('evaluator spec freeze invariant failed')
    dataset_manifest=validate_dataset(train_rows, fixed+adv+dev+primary)
    dataset_manifest['source_sha256']={p:sha(path) for p,path in files.items()}
    dataset_manifest['old_examples']=len(old_rows); dataset_manifest['system_prompt_sha256']=sha(files['training/system_prompt_v2.txt'])
    dump(OUT/'dataset-manifest-v6.json',dataset_manifest)

    install_stack()
    from datasets import Dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    torch.manual_seed(SEED); random.seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False,token=False)
    if tokenizer.pad_token is None: tokenizer.pad_token=tokenizer.eos_token
    dataset=Dataset.from_list([prompt_completion(x,system_prompt) for x in train_rows]).shuffle(seed=SEED)
    base=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False,token=False,dtype=torch.float16)
    base.config.use_cache=False
    lora=LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    args=SFTConfig(output_dir=str(OUT/'trainer-output-v6'),num_train_epochs=2.0,per_device_train_batch_size=1,
                   gradient_accumulation_steps=8,gradient_checkpointing=True,learning_rate=6e-5,warmup_steps=6,
                   weight_decay=0.01,logging_steps=5,save_strategy='no',max_length=1536,completion_only_loss=True,
                   loss_type='chunked_nll',report_to='none',push_to_hub=False,seed=SEED,data_seed=SEED,fp16=True,bf16=False)
    trainer=SFTTrainer(model=base,args=args,train_dataset=dataset,processing_class=tokenizer,peft_config=lora)
    started=time.time(); result=trainer.train(); train_seconds=round(time.time()-started,3)
    ADAPTER.mkdir(parents=True,exist_ok=False); trainer.save_model(str(ADAPTER)); tokenizer.save_pretrained(str(ADAPTER))
    train_loss=float((getattr(result,'metrics',{}) or {}).get('train_loss',getattr(result,'training_loss',0.0)))
    cleanup(trainer,base,result)

    v4_adapter,v4_manifest_path=find_v4_adapter()
    if v4_manifest_path:
        m=json.loads(v4_manifest_path.read_text(encoding='utf-8'))
        if m.get('model_id')!=MODEL_ID or m.get('revision')!=MODEL_REV or m.get('automatic_promotion_authorized') is not False:
            raise RuntimeError('v4 provenance/base contract mismatch')
    print('V4_ADAPTER',v4_adapter,flush=True)

    base_model=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False,token=False,dtype=torch.float16).to('cuda')
    base_primary_out=generate_predictions(base_model,tokenizer,system_prompt,primary,'BASE_PRIMARY_V3'); cleanup(base_model)

    vb=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False,token=False,dtype=torch.float16)
    v4_model=PeftModel.from_pretrained(vb,str(v4_adapter),is_trainable=False).to('cuda')
    v4_primary_out=generate_predictions(v4_model,tokenizer,system_prompt,primary,'V4_PRIMARY_V3')
    v4_dev_out=generate_predictions(v4_model,tokenizer,system_prompt,dev,'V4_DEV_V2'); cleanup(v4_model,vb)

    cb=AutoModelForCausalLM.from_pretrained(MODEL_ID,revision=MODEL_REV,trust_remote_code=False,token=False,dtype=torch.float16)
    v6_model=PeftModel.from_pretrained(cb,str(ADAPTER),is_trainable=False).to('cuda')
    v6_primary_out=generate_predictions(v6_model,tokenizer,system_prompt,primary,'V6_PRIMARY_V3')
    v6_dev_out=generate_predictions(v6_model,tokenizer,system_prompt,dev,'V6_DEV_V2')
    legacy_cases=fixed+adv
    v6_legacy_out=generate_predictions(v6_model,tokenizer,system_prompt,legacy_cases,'V6_LEGACY')

    base_primary=score_holdout(primary,base_primary_out); v4_primary=score_holdout(primary,v4_primary_out); v6_primary=score_holdout(primary,v6_primary_out)
    v4_dev=score_holdout(dev,v4_dev_out); v6_dev=score_holdout(dev,v6_dev_out)
    v6_fixed=score_legacy(fixed,v6_legacy_out); v6_adv=score_legacy(adv,v6_legacy_out); legacy=aggregate_legacy(v6_fixed,v6_adv)
    primary_reg=regressions(primary,v4_primary,v6_primary); dev_reg=regressions(dev,v4_dev,v6_dev); all_reg=sorted(set(primary_reg+dev_reg))
    primary_gain=v6_primary['score']-v4_primary['score']; dev_gain=v6_dev['score']-v4_dev['score']
    primary_safety_gain=v6_primary['safety_score']-v4_primary['safety_score']; dev_safety_gain=v6_dev['safety_score']-v4_dev['safety_score']
    t=spec['promotion_thresholds']
    gates={
      'primary_score':v6_primary['score']>=t['primary_holdout_score_min'],
      'primary_safety':v6_primary['safety_score']>=t['primary_holdout_safety_min'],
      'primary_gain':primary_gain>=t['primary_holdout_gain_vs_v4_min'],
      'development_gain':dev_gain>=t['development_holdout_gain_vs_v4_min'],
      'primary_safety_no_regression':primary_safety_gain>=-1e-12,
      'development_safety_no_regression':dev_safety_gain>=-1e-12,
      'critical_regressions':len(all_reg)<=t['critical_regressions_vs_v4_max'],
      'legacy_fixed_floor':legacy['fixed_score']>=t['legacy_fixed_no_regression_below'],
      'legacy_adversarial_floor':legacy['adversarial_score']>=t['legacy_adversarial_no_regression_below'],
      'legacy_safety_floor':legacy['safety_score']>=t['legacy_safety_no_regression_below'],
    }
    thresholds_passed=all(gates.values())
    report={
      'schema':'nexus.training.candidate-eval.kaggle.v6','ok':True,
      'evaluator_schema':spec['schema'],'system_prompt':'training/system_prompt_v2.txt',
      'blind_primary_holdout':'training/holdout_v3.jsonl','development_holdout':'training/holdout_v2.jsonl',
      'base_primary_v3':compact(base_primary),'v4_primary_v3':compact(v4_primary),'v6_primary_v3':compact(v6_primary),
      'v4_development_v2':compact(v4_dev),'v6_development_v2':compact(v6_dev),
      'v6_legacy':{k:legacy[k] for k in ['fixed_score','adversarial_score','overall_score','safety_score','format_score']},
      'primary_gain_vs_v4':primary_gain,'development_gain_vs_v4':dev_gain,
      'primary_safety_gain_vs_v4':primary_safety_gain,'development_safety_gain_vs_v4':dev_safety_gain,
      'critical_regressions_primary':primary_reg,'critical_regressions_development':dev_reg,'critical_regressions_all':all_reg,
      'gates':gates,'thresholds':t,'thresholds_passed':thresholds_passed,
      'eligible_for_human_review':thresholds_passed,'manual_semantic_review_required':True,
      'automatic_promotion_authorized':False,
    }
    dump(OUT/'candidate-eval-v6.json',report)
    for name,obj in [('primary-details-base-v6.json',base_primary),('primary-details-v4-v6.json',v4_primary),('primary-details-v6.json',v6_primary),('development-details-v4-v6.json',v4_dev),('development-details-v6.json',v6_dev)]: dump(OUT/name,obj)
    write_predictions(OUT/'base-primary-predictions-v6.jsonl','holdout_v3',primary,base_primary_out)
    write_predictions(OUT/'v4-primary-predictions-v6.jsonl','holdout_v3',primary,v4_primary_out)
    write_predictions(OUT/'v6-primary-predictions-v6.jsonl','holdout_v3',primary,v6_primary_out)
    write_predictions(OUT/'v4-development-predictions-v6.jsonl','holdout_v2',dev,v4_dev_out)
    write_predictions(OUT/'v6-development-predictions-v6.jsonl','holdout_v2',dev,v6_dev_out)
    write_predictions(OUT/'v6-fixed-predictions.jsonl','fixed_v1',fixed,v6_legacy_out)
    write_predictions(OUT/'v6-adversarial-predictions.jsonl','adversarial_v1',adv,v6_legacy_out)

    run_manifest={'schema':'nexus.training.run.kaggle.v6','model_id':MODEL_ID,'revision':MODEL_REV,'dataset_commit':DATA_COMMIT,
                  'method':'sft_lora_from_exact_base_with_system_policy_v2','seed':SEED,'examples':len(train_rows),'old_examples':len(old_rows),'v6_examples':len(v6_rows),
                  'gpu':gpu,'visible_gpu_count':torch.cuda.device_count(),'train_seconds':train_seconds,'train_loss':train_loss,
                  'learning_rate':6e-5,'epochs':2.0,'paid_service_used':False,'telemetry':False,'push_to_hub':False,
                  'human_review_required':True,'automatic_promotion_authorized':False,
                  'packages':{name:importlib.metadata.version(name) for name in ['transformers','datasets','peft','trl','accelerate']}}
    dump(OUT/'run-manifest-v6.json',run_manifest)

    bundle_files=['candidate-eval-v6.json','dataset-manifest-v6.json','run-manifest-v6.json','primary-details-base-v6.json','primary-details-v4-v6.json','primary-details-v6.json','development-details-v4-v6.json','development-details-v6.json','base-primary-predictions-v6.jsonl','v4-primary-predictions-v6.jsonl','v6-primary-predictions-v6.jsonl','v4-development-predictions-v6.jsonl','v6-development-predictions-v6.jsonl','v6-fixed-predictions.jsonl','v6-adversarial-predictions.jsonl']
    with zipfile.ZipFile(BUNDLE,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in ADAPTER.rglob('*'):
            if p.is_file() and not p.is_symlink(): z.write(p,p.relative_to(OUT))
        for name in bundle_files: z.write(OUT/name,name)
    print('NEXUS_V6_TRAINING_COMPLETE',flush=True); print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True); print('BUNDLE_SHA256='+sha(BUNDLE),flush=True)
    cleanup(v6_model,cb)

if __name__=='__main__':
    main()
