"""LEARN-191 frozen 3M model/data/evaluation contract."""
from __future__ import annotations
import json, math
from dataclasses import asdict
from pathlib import Path
from typing import Any
import torch
from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import CheckpointIdentity, hash_json, save_trainer_checkpoint, sha256_file, verify_checkpoint
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

WORKER_ID="LEARN-191-REAL-3M"
EXPECTED_CORPUS_ID="422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
EXPECTED_EVAL_ID="7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
PARAMETERS=3_213_120
MODEL_SPEC_SHA256="462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc"
INIT_SPEC_SHA256="86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5"
TARGETS=(16_632,65_772,131_292)
MIDPOINT_TARGET=65_772
FINAL_TARGET=131_292
MAX_STEPS=256
SEED=1337
CLIP_NORM=1.0
MIXTURE=m100.MIXTURE
STRATA=("uk","en","code")
SOURCE_FAMILY={"uk":"project-authored:uk:corpus-v01","en":"project-authored:en:corpus-v01","code":"project-authored:code:corpus-v01"}
SELECTION_EXAMPLES={"uk":256,"en":192,"code":128}
TRAIN_PROBE_EXAMPLES={"uk":32,"en":24,"code":16}
LAUNCH_BINDING={"workflow":"learn191-real-3m","scale":"3m"}

class Learn191Error(RuntimeError): pass
def readj(p:Path)->dict[str,Any]:
    v=json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(v,dict): raise Learn191Error(f"{p} must be JSON object")
    return v
def writej(p:Path,v:object)->None:
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
def appendj(p:Path,v:object)->None:
    with p.open("a",encoding="utf-8") as h: h.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+"\n")

def model_spec()->ModelSpec:
    s=ModelSpec(schema_version=1,vocab_size=256,max_seq_len=256,d_model=192,n_layers=7,n_heads=12,n_kv_heads=12,head_dim=16,d_ff=528,activation="swiglu",norm_kind="rmsnorm",norm_placement="pre",norm_eps=1e-5,position_embedding="rope",rope_theta=10_000.0,rope_rotary_dim=16,attention_bias=False,mlp_bias=False,attention_dropout=0.0,final_norm=True,tie_word_embeddings=True,lm_head_bias=False)
    if s.parameter_count()!=PARAMETERS or s.identity_sha256()!=MODEL_SPEC_SHA256: raise Learn191Error("3M ModelSpec drift")
    return s
def init_spec()->InitSpec:
    s=InitSpec()
    if s.identity_sha256()!=INIT_SPEC_SHA256: raise Learn191Error("InitSpec drift")
    return s
def trainer_config()->TrainerConfig:
    return TrainerConfig(learning_rate=3e-4,weight_decay=0.0,betas=(0.9,0.95),eps=1e-8,max_steps=MAX_STEPS,warmup_steps=0,scheduler="constant",gradient_accumulation_steps=1,gradient_clip_norm=CLIP_NORM,precision="fp32",seed=SEED,deterministic_algorithms=True,deterministic_warn_only=False)
def locks(repo:Path)->dict[str,Any]:
    ps=[Path("requirements/locks/linux-x86_64")/x for x in ("toolchain.lock.txt","runtime.lock.txt","dev.lock.txt")]
    files={p.as_posix():sha256_file(repo/p) for p in ps}; return {"files":files,"combined_sha256":hash_json(files)}

def common(repo:Path,sha:str,out:Path,build:bool):
    m100._require_head(repo,sha); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    manifest=m100._build_corpus(repo,out) if build else readj(out/"corpus-manifest.json")
    if manifest["corpus_identity_sha256"]!=EXPECTED_CORPUS_ID or manifest["train_validation_content_overlap"]!=0: raise Learn191Error("DATA-25 identity/leakage failure")
    train=int(manifest["by_split"]["train"]["byte_tokens"])
    if FINAL_TARGET/train>=0.01: raise Learn191Error("source exposure ceiling exceeded")
    tok=ByteTokenizer()
    if tok.identity.vocab_size!=256 or tok.identity.special_tokens: raise Learn191Error("byte tokenizer drift")
    return manifest,tok,model_spec(),init_spec(),trainer_config(),locks(repo)

def _identity(manifest,tok,split,limits,schema,purpose):
    v={"schema":schema,"corpus_identity_sha256":manifest["corpus_identity_sha256"],"split":split,"packing_version":m100.PACKING_VERSION,"sequence_length":128,"cross_document":False,"ordered_strata":list(STRATA),"packed_example_limits":dict(limits),"selection_rule":"first-N deterministic document-isolated packed examples per stratum","tokenizer_config_sha256":tok.identity.config_sha256,"tokenizer_vocab_sha256":tok.identity.vocab_sha256,"purpose":purpose}
    v["identity_sha256"]=hash_json(v); return v
def run_manifest(sha,manifest,tok,spec,init,cfg,lock)->dict[str,Any]:
    v={"schema":"12-6.learn191-run-manifest.v1","worker_id":WORKER_ID,"source_sha":sha,"model_spec":spec.to_dict(),"model_spec_sha256":spec.identity_sha256(),"parameter_count":spec.parameter_count(),"geometry_authority":{"source":"RESEARCH-138","target_parameters":3_221_432,"family":"RESEARCH-41 MHA/context-256 fixed-control continuation"},"init_spec":init.to_dict(),"init_spec_sha256":init.identity_sha256(),"tokenizer":{"version":tok.identity.version,"config_sha256":tok.identity.config_sha256,"vocab_sha256":tok.identity.vocab_sha256,"vocab_size":256},"corpus_identity_sha256":manifest["corpus_identity_sha256"],"retained_m150_evaluation_identity_sha256":EXPECTED_EVAL_ID,"selection_validation":_identity(manifest,tok,"validation",SELECTION_EXAMPLES,"12-6.learn191-selection-validation.v1","checkpoint selection only; not final test"),"train_probe":_identity(manifest,tok,"train",TRAIN_PROBE_EXAMPLES,"12-6.learn191-train-probe.v1","memorization/generalization diagnostic only"),"trainer_config":asdict(cfg),"batch_size":8,"packing":{"version":m100.PACKING_VERSION,"sequence_length":128,"cross_document":False},"mixture_pattern":list(MIXTURE),"optimized_token_targets":list(TARGETS),"fresh_process_resume_target":MIDPOINT_TARGET,"max_optimizer_steps_safety_ceiling":MAX_STEPS,"budget_basis":{"research138_overfit_evidence":"468K near-best 65K-131K, worse by 262K","train_corpus_byte_tokens":int(manifest["by_split"]["train"]["byte_tokens"]),"source_exposure_fraction":FINAL_TARGET/int(manifest["by_split"]["train"]["byte_tokens"]),"recycling_study":False},"compute_proxy_final_6NT":6*spec.parameter_count()*FINAL_TARGET,"environment_lock_sha256":lock["combined_sha256"],"random_initialization":True,"foreign_pretrained_weights":False,"sft":False,"paid_compute":False}
    v=json.loads(json.dumps(v,sort_keys=True,separators=(",",":"))); v["identity_sha256"]=hash_json(v); return v

def _subset(corpus,manifest,tok,split,stratum,limit):
    xs=[]
    for x in m100._packed(corpus,manifest,tok,split,stratum):
        xs.append(x)
        if len(xs)==limit: break
    if len(xs)!=limit: raise Learn191Error(f"{split}/{stratum} subset exhausted")
    return xs
@torch.no_grad()
def evaluate(model:TwelveSixDecoder,corpus:Path,manifest,tok,split:str,limits:dict[str,int])->dict[str,Any]:
    before=m100._state_hash(model); mode=model.training; model.eval(); total_nll=0.0; total_tok=0; by={}
    try:
        for s in STRATA:
            xs=_subset(corpus,manifest,tok,split,s,limits[s]); nll=0.0; nt=0
            for i in range(0,len(xs),32):
                a,b=m100._eval_examples(model,xs[i:i+32]); nll+=a; nt+=b
            by[s]={"bits_per_byte":nll/math.log(2)/nt,"loss_nats_per_byte":nll/nt,"predicted_byte_tokens":nt}; total_nll+=nll; total_tok+=nt
    finally: model.train(mode)
    after=m100._state_hash(model)
    if before!=after: raise Learn191Error("evaluation mutated model")
    return {"split":split,"bits_per_byte":total_nll/math.log(2)/total_tok,"loss_nats_per_byte":total_nll/total_tok,"predicted_byte_tokens":total_tok,"by_stratum":by,"by_source_family":{SOURCE_FAMILY[s]:{**by[s],"stratum":s} for s in STRATA},"model_state_sha256_before":before,"model_state_sha256_after":after,"non_mutation_passed":True}

def checkpoint_identity(sha,spec,tok,manifest,run,cfg,trainer,lock):
    return CheckpointIdentity(git_sha=sha,model_spec=spec.to_dict(),parameter_count=spec.parameter_count(),tokenizer_hash=tok.identity.config_sha256,tokenizer_vocab_hash=tok.identity.vocab_sha256,dataset_manifest_hash=manifest["corpus_identity_sha256"],run_manifest_hash=run["identity_sha256"],training_config={"trainer":asdict(cfg),"selection_validation_identity_sha256":run["selection_validation"]["identity_sha256"]},seed=cfg.seed,precision=cfg.precision,step=trainer.optimizer_step,tokens_seen=trainer.tokens_seen,optimizer={"name":"AdamW","learning_rate":cfg.learning_rate,"betas":list(cfg.betas),"eps":cfg.eps,"weight_decay":cfg.weight_decay},scheduler=None,environment_lock_hash=lock["combined_sha256"])
def checkpoint_dir(out:Path,target:int)->Path: return out/f"checkpoint-t{target:06d}"
def save_checkpoint(out,target,sha,spec,tok,manifest,run,cfg,trainer,lock):
    p=checkpoint_dir(out,target); save_trainer_checkpoint(p,model=trainer.model,trainer=trainer,identity=checkpoint_identity(sha,spec,tok,manifest,run,cfg,trainer,lock),overwrite=True); c=verify_checkpoint(p)
    return {"target_optimized_tokens":target,"actual_optimized_tokens":trainer.tokens_seen,"optimizer_step":trainer.optimizer_step,"checkpoint_id":c["checkpoint_id"],"path":p.name}
def snapshot(model): return [p.detach().cpu().clone() for p in model.parameters()]
def update_ratio(model,before):
    ds=bs=0.0
    for old,p in zip(before,model.parameters(),strict=True):
        new=p.detach().cpu().float(); old=old.float(); d=new-old; ds+=float((d*d).sum()); bs+=float((old*old).sum())
    return math.sqrt(ds)/math.sqrt(bs) if bs>0 else 0.0
