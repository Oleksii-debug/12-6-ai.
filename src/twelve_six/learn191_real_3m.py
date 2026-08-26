"""LEARN-191 execution: prepare, scratch train, midpoint resume, finalize."""
from __future__ import annotations
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import torch
from twelve_six import milestone100_first_learned as m100
from twelve_six.checkpoint import hash_json, load_trainer_checkpoint, verify_checkpoint
from twelve_six.launch_gate import require_launch_envelope_from_env
from twelve_six.model import TwelveSixDecoder
from twelve_six.training import Trainer
from twelve_six.training.observability import TrainingObserver
from twelve_six.learn191_contract import *

def machine(sha,lock): return {"source_sha":sha,"python":platform.python_version(),"torch":torch.__version__,"platform":platform.platform(),"machine":platform.machine(),"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"cuda_available":torch.cuda.is_available(),"device":"cpu","pid":os.getpid(),"paid_compute":False,"environment_locks":lock}
def newseg(): return {"nll_times_tokens":0.0,"tokens":0,"steps":0,"clip_activations":0,"grad_norm_min":None,"grad_norm_max":None,"update_ratios":[]}
def recseg(seg,m,ratio):
    loss=float(m.update_loss if m.update_loss is not None else m.loss); t=int(m.tokens); seg["nll_times_tokens"]+=loss*t; seg["tokens"]+=t; seg["steps"]+=1
    if m.grad_norm is not None:
        g=float(m.grad_norm); seg["grad_norm_min"]=g if seg["grad_norm_min"] is None else min(seg["grad_norm_min"],g); seg["grad_norm_max"]=g if seg["grad_norm_max"] is None else max(seg["grad_norm_max"],g); seg["clip_activations"]+=int(g>CLIP_NORM)
    if ratio is not None: seg["update_ratios"].append(float(ratio))
def finish(seg):
    t=int(seg["tokens"]); rs=seg["update_ratios"]
    return {"train_bits_per_byte":seg["nll_times_tokens"]/math.log(2)/t if t else None,"optimized_tokens":t,"optimizer_steps":int(seg["steps"]),"clip_activations":int(seg["clip_activations"]),"clip_fraction":seg["clip_activations"]/seg["steps"] if seg["steps"] else None,"gradient_norm_min":seg["grad_norm_min"],"gradient_norm_max":seg["grad_norm_max"],"update_ratio_mean":sum(rs)/len(rs) if rs else None,"update_ratio_max":max(rs) if rs else None,"update_ratio_samples":len(rs)}
def evalpoint(model,corpus,manifest,tok,target,trainer,seg):
    val=evaluate(model,corpus,manifest,tok,"validation",SELECTION_EXAMPLES); probe=evaluate(model,corpus,manifest,tok,"train",TRAIN_PROBE_EXAMPLES)
    return {"target_optimized_tokens":target,"actual_optimized_tokens":trainer.tokens_seen,"optimizer_step":trainer.optimizer_step,"selection_validation":val,"train_probe":probe,"memorization":{"method":"fixed-train-probe-vs-immutable-selection-validation-gap","train_minus_validation_bpb":probe["bits_per_byte"]-val["bits_per_byte"],"privacy_leakage_claim":False},"train_segment":finish(seg)}
def loaded_common(repo,sha,out):
    manifest,tok,spec,init,cfg,lock=common(repo,sha,out,False); run=run_manifest(sha,manifest,tok,spec,init,cfg,lock)
    if readj(out/"run-manifest.json")!=run: raise Learn191Error("run manifest changed after prepare")
    return manifest,tok,spec,init,cfg,lock,run

def prepare(repo:Path,sha:str,out:Path):
    out.mkdir(parents=True,exist_ok=True); manifest,tok,spec,init,cfg,lock=common(repo,sha,out,True); run=run_manifest(sha,manifest,tok,spec,init,cfg,lock); writej(out/"run-manifest.json",run)
    truth={"schema":"12-6.learn191-truth.v1","worker_id":WORKER_ID,"source_sha":sha,"parameter_count":spec.parameter_count(),"model_spec_sha256":spec.identity_sha256(),"corpus_identity_sha256":manifest["corpus_identity_sha256"],"selection_validation_identity_sha256":run["selection_validation"]["identity_sha256"],"optimized_token_targets":list(TARGETS),"midpoint_resume_target":MIDPOINT_TARGET,"source_exposure_fraction_at_final_target":FINAL_TARGET/int(manifest["by_split"]["train"]["byte_tokens"]),"compute_proxy_final_6NT":run["compute_proxy_final_6NT"],"random_initialization":True,"foreign_pretrained_weights":False,"sft":False,"paid_compute":False}; truth["identity_sha256"]=hash_json(truth); writej(out/"truth.json",truth); writej(out/"machine-prepare.json",machine(sha,lock)); return truth

def train_step(observer,trainer,model,batch,wait,seg,curve,phase,stratum):
    take=(trainer.optimizer_step+1)%8==0; before=snapshot(model) if take else None; m=observer.train_microbatch(trainer,batch,data_wait_seconds=wait); ratio=update_ratio(model,before) if before is not None else None; recseg(seg,m,ratio)
    appendj(curve,{"phase":phase,"optimizer_step":trainer.optimizer_step,"optimized_tokens":trainer.tokens_seen,"stratum":stratum,"train_bits_per_byte":float(m.update_loss if m.update_loss is not None else m.loss)/math.log(2),"grad_norm_pre_clip":m.grad_norm,"clip_active":bool(m.grad_norm is not None and m.grad_norm>CLIP_NORM),"update_ratio_sample":ratio,"learning_rate":m.learning_rate})

def phase1(repo:Path,sha:str,out:Path):
    require_launch_envelope_from_env(repo,expected_binding=LAUNCH_BINDING); started=time.perf_counter(); manifest,tok,spec,init,cfg,lock,run=loaded_common(repo,sha,out); so=out/"3m"; so.mkdir(parents=True,exist_ok=True); corpus=out/"corpus-a"
    torch.manual_seed(SEED); model=TwelveSixDecoder(spec,init); random_hash=m100._state_hash(model); trainer=Trainer(model,cfg,device="cpu"); obs=TrainingObserver(run,device="cpu",max_step_samples=512); curve=so/"train-curve.jsonl"; curve.unlink(missing_ok=True)
    evals=[evalpoint(model,corpus,manifest,tok,0,trainer,newseg())]; cps=[]; its=m100._train_iters(corpus,manifest,tok,0); batches={s:m100._batches(it) for s,it in its.items()}; pending=[TARGETS[0],TARGETS[1]]; seg=newseg()
    for i in range(MAX_STEPS):
        if trainer.tokens_seen>=MIDPOINT_TARGET: break
        s=MIXTURE[i%len(MIXTURE)]; batch,wait=obs.measure_next(batches[s]); train_step(obs,trainer,model,batch,wait,seg,curve,"phase1",s)
        while pending and trainer.tokens_seen>=pending[0]:
            target=pending.pop(0); captured=seg; cp=obs.measure_region("checkpoint",f"save-t{target}",lambda t=target:save_checkpoint(so,t,sha,spec,tok,manifest,run,cfg,trainer,lock),optimizer_step=trainer.optimizer_step,tokens_seen=trainer.tokens_seen); cps.append(cp); evals.append(obs.measure_region("evaluation",f"eval-t{target}",lambda t=target,x=captured:evalpoint(model,corpus,manifest,tok,t,trainer,x),optimizer_step=trainer.optimizer_step,tokens_seen=trainer.tokens_seen)); seg=newseg()
    if trainer.tokens_seen<MIDPOINT_TARGET or pending or cps[-1]["target_optimized_tokens"]!=MIDPOINT_TARGET: raise Learn191Error("midpoint target not reached")
    r={"schema":"12-6.learn191-phase1.v1","source_sha":sha,"process_pid":os.getpid(),"python_executable":sys.executable,"random_init_state_sha256":random_hash,"optimizer_step":trainer.optimizer_step,"optimized_tokens":trainer.tokens_seen,"checkpoints":cps,"evaluations":evals,"observability":obs.summary(),"wall_seconds":time.perf_counter()-started,"machine":machine(sha,lock)}; r["identity_sha256"]=hash_json(r); writej(so/"phase1.json",r); return r

def resume(repo:Path,sha:str,out:Path):
    require_launch_envelope_from_env(repo,expected_binding=LAUNCH_BINDING); started=time.perf_counter(); manifest,tok,spec,init,cfg,lock,run=loaded_common(repo,sha,out); so=out/"3m"; p1=readj(so/"phase1.json"); torch.manual_seed(SEED); model=TwelveSixDecoder(spec,init); trainer=Trainer(model,cfg,device="cpu")
    load_trainer_checkpoint(checkpoint_dir(so,MIDPOINT_TARGET),model=model,trainer=trainer,strict_model=True,restore_rng=True,expected_git_sha=sha,expected_model_spec_hash=spec.identity_sha256(),expected_tokenizer_hash=tok.identity.config_sha256,expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],expected_run_manifest_hash=run["identity_sha256"])
    if trainer.tokens_seen!=p1["optimized_tokens"] or trainer.optimizer_step!=p1["optimizer_step"] or os.getpid()==int(p1["process_pid"]): raise Learn191Error("fresh midpoint resume proof failed")
    obs=TrainingObserver(run,device="cpu",max_step_samples=512); its=m100._train_iters(out/"corpus-a",manifest,tok,trainer.optimizer_step); batches={s:m100._batches(it) for s,it in its.items()}; curve=so/"train-curve.jsonl"; seg=newseg(); i=trainer.optimizer_step
    while trainer.tokens_seen<FINAL_TARGET and trainer.optimizer_step<MAX_STEPS:
        s=MIXTURE[i%len(MIXTURE)]; i+=1; batch,wait=obs.measure_next(batches[s]); train_step(obs,trainer,model,batch,wait,seg,curve,"resume",s)
    if trainer.tokens_seen<FINAL_TARGET: raise Learn191Error("final token target not reached")
    cp=obs.measure_region("checkpoint",f"save-t{FINAL_TARGET}",lambda:save_checkpoint(so,FINAL_TARGET,sha,spec,tok,manifest,run,cfg,trainer,lock),optimizer_step=trainer.optimizer_step,tokens_seen=trainer.tokens_seen); ev=obs.measure_region("evaluation",f"eval-t{FINAL_TARGET}",lambda:evalpoint(model,out/"corpus-a",manifest,tok,FINAL_TARGET,trainer,seg),optimizer_step=trainer.optimizer_step,tokens_seen=trainer.tokens_seen); generation=m100._generation(checkpoint_dir(so,FINAL_TARGET))
    trained=[x for x in [*p1["evaluations"],ev] if x["target_optimized_tokens"]>0]; best=min(trained,key=lambda x:x["selection_validation"]["bits_per_byte"]); bt=int(best["target_optimized_tokens"]); bestrec={"selection_rule":"minimum preregistered selection-validation BPB","target_optimized_tokens":bt,"actual_optimized_tokens":best["actual_optimized_tokens"],"optimizer_step":best["optimizer_step"],"selection_validation_bpb":best["selection_validation"]["bits_per_byte"],"checkpoint_path":checkpoint_dir(so,bt).name}; finalrec={"target_optimized_tokens":FINAL_TARGET,"actual_optimized_tokens":trainer.tokens_seen,"optimizer_step":trainer.optimizer_step,"selection_validation_bpb":ev["selection_validation"]["bits_per_byte"],"checkpoint_path":checkpoint_dir(so,FINAL_TARGET).name}; writej(so/"best-checkpoint.json",bestrec); writej(so/"final-checkpoint.json",finalrec)
    r={"schema":"12-6.learn191-resume.v1","source_sha":sha,"process_pid":os.getpid(),"phase1_process_pid":p1["process_pid"],"fresh_process_resume_passed":True,"optimizer_step":trainer.optimizer_step,"optimized_tokens":trainer.tokens_seen,"source_exposure_fraction":trainer.tokens_seen/int(manifest["by_split"]["train"]["byte_tokens"]),"final_checkpoint":cp,"final_evaluation":ev,"best_checkpoint":bestrec,"generation":generation,"observability":obs.summary(),"wall_seconds":time.perf_counter()-started,"machine":machine(sha,lock)}; r["identity_sha256"]=hash_json(r); writej(so/"resume.json",r); return r

def finalize(repo:Path,sha:str,out:Path):
    m100._require_head(repo,sha); run=readj(out/"run-manifest.json"); p1=readj(out/"3m/phase1.json"); r=readj(out/"3m/resume.json"); best=readj(out/"3m/best-checkpoint.json"); final=readj(out/"3m/final-checkpoint.json"); evs=[*p1["evaluations"],r["final_evaluation"]]
    if [x["target_optimized_tokens"] for x in evs]!=[0,*TARGETS] or not r["fresh_process_resume_passed"] or r["source_exposure_fraction"]>=0.01 or not all(x["selection_validation"]["non_mutation_passed"] and x["train_probe"]["non_mutation_passed"] for x in evs): raise Learn191Error("final scientific gate failed")
    report={"schema":"12-6.learn191-real-3m.v1","worker_id":WORKER_ID,"authority":"LOCAL_FREE_LEARNED_BASE_RESEARCH_ARTIFACT_NOT_STAGE_PROMOTION","source_sha":sha,"model":{"spec":run["model_spec"],"spec_sha256":run["model_spec_sha256"],"parameter_count":run["parameter_count"],"init_spec":run["init_spec"],"init_spec_sha256":run["init_spec_sha256"],"random_initialization":True},"tokenizer":run["tokenizer"],"corpus_identity_sha256":run["corpus_identity_sha256"],"retained_m150_evaluation_identity_sha256":run["retained_m150_evaluation_identity_sha256"],"selection_validation":run["selection_validation"],"train_probe":run["train_probe"],"optimizer":run["trainer_config"],"packing":run["packing"],"batch_size":run["batch_size"],"seed":SEED,"budget_basis":run["budget_basis"],"compute_proxy_final_6NT":run["compute_proxy_final_6NT"],"evaluations":evs,"checkpoints":[*p1["checkpoints"],r["final_checkpoint"]],"best_checkpoint":best,"final_checkpoint":final,"fresh_process_resume":{"required_target_optimized_tokens":MIDPOINT_TARGET,"phase1_pid":p1["process_pid"],"resume_pid":r["process_pid"],"passed":True},"observability":{"phase1":p1["observability"],"resume":r["observability"]},"source_exposure_fraction":r["source_exposure_fraction"],"generation":r["generation"],"comparison_contract":{"direct_1m":"same M150 family/corpus/tokenizer/init/optimizer; exact numeric comparison requires this selection identity","direct_10m":"no comparable learned 10M authority yet; future 10M must use same corpus/tokenizer/evaluation identity","prediction_not_retuned_after_observation":True},"truth_boundary":{"external_real_world_training_data_present":False,"representative_external_corpus_claim":False,"foreign_pretrained_weights":False,"sft":False,"paid_compute":False,"privacy_leakage_claim":False,"instruction_following_claim":False,"production_readiness_claim":False}}; report["identity_sha256"]=hash_json(report); writej(out/"learn191-real-3m-report.json",report); return report

def verify_final(repo:Path,sha:str,out:Path):
    m100._require_head(repo,sha); manifest,tok,spec,init,cfg,_lock,run=loaded_common(repo,sha,out); final=readj(out/"3m/final-checkpoint.json"); p=out/"3m"/final["checkpoint_path"]; checked=verify_checkpoint(p); torch.manual_seed(SEED); model=TwelveSixDecoder(spec,init); trainer=Trainer(model,cfg,device="cpu"); load_trainer_checkpoint(p,model=model,trainer=trainer,strict_model=True,restore_rng=False,expected_git_sha=sha,expected_model_spec_hash=spec.identity_sha256(),expected_tokenizer_hash=tok.identity.config_sha256,expected_tokenizer_vocab_hash=tok.identity.vocab_sha256,expected_dataset_manifest_hash=manifest["corpus_identity_sha256"],expected_run_manifest_hash=run["identity_sha256"]); proof={"schema":"12-6.learn191-final-fresh-load-proof.v1","source_sha":sha,"process_pid":os.getpid(),"checkpoint_id":checked["checkpoint_id"],"optimizer_step":trainer.optimizer_step,"optimized_tokens":trainer.tokens_seen,"model_state_sha256":m100._state_hash(model),"generation":m100._generation(p)}; proof["identity_sha256"]=hash_json(proof); writej(out/"3m/final-fresh-load-proof.json",proof); return proof

def main(argv=None):
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    for n in ("prepare","phase1","resume","finalize","verify-final"):
        p=sub.add_parser(n); p.add_argument("--repo-root",type=Path,default=Path(".")); p.add_argument("--source-sha",required=True); p.add_argument("--output-dir",type=Path,default=Path("learn191-evidence"))
    a=ap.parse_args(argv); f={"prepare":prepare,"phase1":phase1,"resume":resume,"finalize":finalize,"verify-final":verify_final}[a.cmd]; print(json.dumps(f(a.repo_root,a.source_sha,a.output_dir),ensure_ascii=False,sort_keys=True,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
