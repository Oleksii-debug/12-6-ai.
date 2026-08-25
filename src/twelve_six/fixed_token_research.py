"""Strict fixed-valid-token research experiments for 12-6 AI.

Additive successor to RESEARCH41. It reuses the canonical decoder, Trainer, byte
tokenizer, S0 controlled corpus, and RESEARCH41 batch trace while making token
checkpoints exact through aligned causal targets and loss masks.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from .model import InitSpec, ModelSpec, TwelveSixDecoder
from .scaling_experiment import (
    PACKING_ID as RESEARCH41_PACKING_ID,
    _byte_stream,
    _make_batch,
    _read_jsonl,
    _trainer_config,
    _validation_loss,
    controlled_specs,
)
from .tokenization import (
    BYTE_TOKENIZER_HASH,
    BYTE_TOKENIZER_VERSION,
    BYTE_VOCAB_HASH,
    ByteTokenizer,
)
from .training import Trainer

SCHEMA = "12-6.fixed-token-research.v1"
COLLECTION_SCHEMA = "12-6.fixed-token-research-collection.v1"
CHECKPOINT_SCHEMA = "12-6.fixed-token-research-checkpoint.v1"
AUTHORITY = "LOCAL_FREE_FIXED_VALID_CAUSAL_TOKEN_RESEARCH_NOT_PROMOTION"
TRACE_PROTOCOL = "research06-research41-batch-trace-exact-loss-mask-v1"
COMPUTE_PROXY = "6 * trainable_parameters * exact_optimized_valid_causal_tokens"
DEFAULT_BUDGETS = (16_384, 65_536)
DEFAULT_BATCH_SIZE = 4
DEFAULT_SEQUENCE_LENGTH = 64
DEFAULT_SEED = 1337
DEFAULT_THREADS = 2

_DEPTH_WIDTH_GEOMETRIES: tuple[tuple[str, dict[str, int]], ...] = (
    ("wide_shallow_d64_l2", {"d_model":64,"n_layers":2,"n_heads":4,"n_kv_heads":4,"head_dim":16,"d_ff":132}),
    ("balanced_d48_l3", {"d_model":48,"n_layers":3,"n_heads":4,"n_kv_heads":4,"head_dim":12,"d_ff":138}),
    ("deeper_d40_l4", {"d_model":40,"n_layers":4,"n_heads":4,"n_kv_heads":4,"head_dim":10,"d_ff":132}),
    ("deep_d32_l6", {"d_model":32,"n_layers":6,"n_heads":4,"n_kv_heads":4,"head_dim":8,"d_ff":116}),
    ("very_deep_narrow_d28_l8", {"d_model":28,"n_layers":8,"n_heads":2,"n_kv_heads":2,"head_dim":14,"d_ff":100}),
)
_EXPECTED_DEPTH_WIDTH: Mapping[str, tuple[int, str]] = {
    "wide_shallow_d64_l2": (100_160, "8ff79ab5a9bb7b4db063b1cf66a757ef12214d3a9b4bd71742e823606095a38b"),
    "balanced_d48_l3": (99_888, "132af19baedd8c2d1ef1c5ac84703a9994994a383d7a506dfa74b0a4015d2761"),
    "deeper_d40_l4": (99_560, "466a081277ad0007e57bcd4e2e28757e51916bbd9aef8c016d3d593751d3d117"),
    "deep_d32_l6": (100_000, "3c973ef75f5212ee4c15c43c13a0a16d1ceaed8d2bc1cf3c7c0e1bb94e4471cc"),
    "very_deep_narrow_d28_l8": (99_932, "a93b00cf3218f356009c4460f541bb239afeb1864199b032a33a3ee6e48202db"),
}


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(repo_root: Path) -> str:
    return subprocess.run(["git","rev-parse","HEAD"], cwd=repo_root, check=True, capture_output=True, text=True).stdout.strip()


def _model_spec(g: Mapping[str, int]) -> ModelSpec:
    return ModelSpec(
        schema_version=1, vocab_size=256, max_seq_len=256,
        d_model=int(g["d_model"]), n_layers=int(g["n_layers"]),
        n_heads=int(g["n_heads"]), n_kv_heads=int(g["n_kv_heads"]),
        head_dim=int(g["head_dim"]), d_ff=int(g["d_ff"]),
        activation="swiglu", norm_kind="rmsnorm", norm_placement="pre", norm_eps=1e-5,
        position_embedding="rope", rope_theta=10_000.0, rope_rotary_dim=int(g["head_dim"]),
        attention_bias=False, mlp_bias=False, attention_dropout=0.0,
        final_norm=True, tie_word_embeddings=True, lm_head_bias=False,
    )


def depth_width_specs() -> dict[str, ModelSpec]:
    specs = {name: _model_spec(geometry) for name, geometry in _DEPTH_WIDTH_GEOMETRIES}
    for name, spec in specs.items():
        count, identity = _EXPECTED_DEPTH_WIDTH[name]
        if spec.parameter_count() != count or spec.identity_sha256() != identity:
            raise RuntimeError(f"depth/width ModelSpec drift for {name}")
    return specs


def scaling_specs() -> dict[str, ModelSpec]:
    return {f"scale_{spec.parameter_count():07d}": spec for spec in controlled_specs()}


def candidate_specs(family: str) -> dict[str, ModelSpec]:
    if family == "scaling":
        return scaling_specs()
    if family == "depth_width_100k":
        return depth_width_specs()
    raise ValueError(f"unsupported family: {family}")


def config_payload() -> dict[str, Any]:
    init = InitSpec()
    return {
        "schema":"12-6.model08-depth-width-config.v1",
        "authority":"EXPERIMENTAL_RESEARCH_CONFIG_NOT_CANONICAL_S0",
        "fixed_controls": {
            "tokenizer":BYTE_TOKENIZER_VERSION,"vocab_size":256,"max_seq_len":256,
            "init_spec":init.to_dict(),"init_identity_sha256":init.identity_sha256(),
            "optimizer":{"name":"AdamW","learning_rate":3e-4,"betas":[0.9,0.95],"eps":1e-8,"weight_decay":0.0,"gradient_clip_norm":1.0,"scheduler":"constant","warmup_steps":0,"precision":"fp32","seed":DEFAULT_SEED},
            "batch_size":DEFAULT_BATCH_SIZE,"sequence_length":DEFAULT_SEQUENCE_LENGTH,
            "trace_protocol":TRACE_PROTOCOL,"research41_parent_packing":RESEARCH41_PACKING_ID,
        },
        "candidates":[{"candidate_id":name,"parameters":spec.parameter_count(),"model_identity_sha256":spec.identity_sha256(),"model_spec":spec.to_dict()} for name,spec in depth_width_specs().items()],
    }


def _validate_budgets(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(int(v) for v in values)
    if not result or any(v <= 0 for v in result) or tuple(sorted(set(result))) != result:
        raise ValueError("token budgets must be positive, strictly increasing, and unique")
    return result


def _steps_for_budgets(budgets: tuple[int, ...], capacity: int) -> int:
    previous = total = 0
    for budget in budgets:
        total += math.ceil((budget - previous) / capacity)
        previous = budget
    return total


def _aligned_batch(input_ids: torch.Tensor, valid_tokens: int) -> dict[str, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[1] < 2:
        raise ValueError("input_ids must have shape [batch,time>=2]")
    capacity = int(input_ids.shape[0] * (input_ids.shape[1] - 1))
    if not 0 < valid_tokens <= capacity:
        raise ValueError(f"valid_tokens must be in [1,{capacity}]")
    targets = torch.full_like(input_ids, -100)
    targets[:, :-1] = input_ids[:, 1:]
    mask = torch.zeros_like(input_ids)
    valid = torch.zeros((input_ids.shape[0], input_ids.shape[1]-1), dtype=mask.dtype)
    valid.reshape(-1)[:valid_tokens] = 1
    mask[:, :-1] = valid
    return {"input_ids":input_ids,"target_ids":targets,"loss_mask":mask}


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _state_hash(state: Any) -> str:
    digest = hashlib.sha256()
    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            cpu=value.detach().cpu().contiguous(); digest.update(b"T"); digest.update(str(cpu.dtype).encode()); digest.update(json.dumps(list(cpu.shape)).encode()); digest.update(cpu.numpy().tobytes())
        elif isinstance(value, Mapping):
            digest.update(b"{")
            for key in sorted(value, key=lambda item:str(item)):
                digest.update(str(key).encode()); visit(value[key])
            digest.update(b"}")
        elif isinstance(value,(list,tuple)):
            digest.update(b"[")
            for item in value: visit(item)
            digest.update(b"]")
        elif value is None or isinstance(value,(bool,int,float,str)):
            digest.update(json.dumps(value,sort_keys=True).encode())
        else:
            digest.update(repr(value).encode())
    visit(state); return digest.hexdigest()


def _model_state_hash(model: TwelveSixDecoder) -> str:
    return _state_hash(model.state_dict())


def _trainer_state_hash(trainer: Trainer) -> str:
    return _state_hash(asdict(trainer.state_dict()))


def _optimizer_tensor_bytes(trainer: Trainer) -> int:
    return sum(v.numel()*v.element_size() for state in trainer.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor))


def _parameter_tensor_bytes(model: TwelveSixDecoder) -> int:
    return sum(p.numel()*p.element_size() for p in model.parameters())


def _rss_hwm_mib() -> float:
    value=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value/(1024.0*1024.0) if sys.platform=="darwin" else value/1024.0


def _parameter_snapshot(model: TwelveSixDecoder) -> tuple[list[torch.Tensor], float]:
    snapshots=[]; norm_sq=0.0
    for parameter in model.parameters():
        value=parameter.detach().cpu().clone(); snapshots.append(value)
        norm_sq += float(torch.sum(value.float()*value.float()).item())
    return snapshots, math.sqrt(norm_sq)


def _update_stats(model: TwelveSixDecoder, before: list[torch.Tensor], weight_norm_before: float) -> tuple[float,float,float,int]:
    delta_sq=0.0; max_abs=0.0; changed=0
    for parameter,old in zip(model.parameters(),before,strict=True):
        delta=parameter.detach().cpu()-old; delta_sq += float(torch.sum(delta.float()*delta.float()).item())
        if delta.numel():
            max_abs=max(max_abs,float(delta.abs().max().item())); changed += int(torch.count_nonzero(delta).item())
    delta_l2=math.sqrt(delta_sq)
    return delta_l2, delta_l2/weight_norm_before if weight_norm_before>0 else math.inf, max_abs, changed


def _activation_probe(model: TwelveSixDecoder, input_ids: torch.Tensor) -> dict[str,Any]:
    observations=[]; handles=[]
    def make_hook(name: str):
        def capture(_module:torch.nn.Module,_inputs:tuple[Any,...],output:Any)->None:
            if isinstance(output,torch.Tensor):
                value=output.detach().float(); observations.append({"name":name,"rms":float(torch.sqrt(torch.mean(value*value)).item()),"std":float(value.std(unbiased=False).item()),"max_abs":float(value.abs().max().item())})
        return capture
    handles.append(model.token_embedding.register_forward_hook(make_hook("token_embedding")))
    for i,block in enumerate(model.blocks): handles.append(block.register_forward_hook(make_hook(f"block_{i}")))
    was_training=model.training; model.eval()
    with torch.no_grad(): logits=model(input_ids).logits.detach().float()
    model.train(was_training)
    for handle in handles: handle.remove()
    finite=all(math.isfinite(float(item[k])) for item in observations for k in ("rms","std","max_abs")) and bool(torch.isfinite(logits).all().item())
    embedding_rms=float(observations[0]["rms"]) if observations else math.nan
    block_rms=[float(item["rms"]) for item in observations if str(item["name"]).startswith("block_")]
    amplification=max(block_rms)/embedding_rms if block_rms and embedding_rms>0 else math.inf
    reasons=[]
    if not finite: reasons.append("non_finite_initial_activation_or_logits")
    if not math.isfinite(amplification) or amplification>8.0: reasons.append("residual_activation_rms_amplification_gt_8x")
    return {"status":"PASS" if not reasons else "REJECT_UNSTABLE_INITIALIZATION","reasons":reasons,"layers":observations,"logits_rms":float(torch.sqrt(torch.mean(logits*logits)).item()),"max_block_to_embedding_rms":amplification}


def _summary(values: list[float]) -> dict[str,float|None]:
    if not values: return {"min":None,"mean":None,"median":None,"max":None}
    ordered=sorted(values); n=len(ordered); median=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/2
    return {"min":min(values),"mean":sum(values)/n,"median":median,"max":max(values)}


def _save_checkpoint(path:Path, *, controls_sha256:str, source_sha:str, family:str, candidate_id:str, model:TwelveSixDecoder, trainer:Trainer, completed_budget_index:int, token_budgets:tuple[int,...], trace_events:list[dict[str,Any]]) -> dict[str,Any]:
    trainer.assert_checkpoint_safe(); expected=token_budgets[completed_budget_index]
    if trainer.tokens_seen!=expected: raise RuntimeError("refusing checkpoint with token-accounting drift")
    payload={"schema":CHECKPOINT_SCHEMA,"controls_sha256":controls_sha256,"source_sha":source_sha,"family":family,"candidate_id":candidate_id,"model_identity_sha256":model.spec.identity_sha256(),"init_identity_sha256":model.init_spec.identity_sha256(),"completed_budget_index":completed_budget_index,"exact_optimized_tokens":trainer.tokens_seen,"optimizer_step":trainer.optimizer_step,"trace_sha256":_canonical_hash(trace_events),"trace_events":trace_events,"model_state":copy.deepcopy(model.state_dict()),"trainer_state":asdict(trainer.state_dict()),"torch_rng_state":torch.get_rng_state(),"python_rng_state":random.getstate()}
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)
    return {"path":str(path),"sha256":_file_sha256(path),"bytes":path.stat().st_size,"model_state_sha256":_state_hash(payload["model_state"]),"trainer_state_sha256":_state_hash(payload["trainer_state"]),"trace_sha256":payload["trace_sha256"]}


def _fresh_resume(checkpoint_path:Path, *, expected_controls_sha256:str, expected_source_sha:str, expected_family:str, expected_candidate_id:str, spec:ModelSpec, init_spec:InitSpec, max_steps:int, seed:int, expected_token_budgets:tuple[int,...]) -> tuple[TwelveSixDecoder,Trainer,list[dict[str,Any]],dict[str,Any]]:
    payload=torch.load(checkpoint_path,map_location="cpu",weights_only=False)
    if payload.get("schema")!=CHECKPOINT_SCHEMA: raise RuntimeError("research checkpoint schema mismatch")
    expected={"controls_sha256":expected_controls_sha256,"source_sha":expected_source_sha,"family":expected_family,"candidate_id":expected_candidate_id,"model_identity_sha256":spec.identity_sha256(),"init_identity_sha256":init_spec.identity_sha256()}
    for key,value in expected.items():
        if payload.get(key)!=value: raise RuntimeError(f"research checkpoint identity drift: {key}")
    index=int(payload["completed_budget_index"])
    if not 0<=index<len(expected_token_budgets) or int(payload["exact_optimized_tokens"])!=expected_token_budgets[index]: raise RuntimeError("checkpoint optimized-token count drift")
    trace_events=list(payload["trace_events"])
    if _canonical_hash(trace_events)!=payload["trace_sha256"]: raise RuntimeError("checkpoint trace hash drift")
    random.seed(seed); torch.manual_seed(seed); model=TwelveSixDecoder(spec,init_spec); trainer=Trainer(model,_trainer_config(max_steps=max_steps,seed=seed),device="cpu")
    model.load_state_dict(payload["model_state"],strict=True); trainer.load_state_dict(payload["trainer_state"])
    if trainer.tokens_seen!=expected_token_budgets[index] or trainer.optimizer_step!=int(payload["optimizer_step"]): raise RuntimeError("restored trainer counter drift")
    if _model_state_hash(model)!=_state_hash(payload["model_state"]) or _trainer_state_hash(trainer)!=_state_hash(payload["trainer_state"]): raise RuntimeError("restored state mismatch")
    torch.set_rng_state(payload["torch_rng_state"]); random.setstate(payload["python_rng_state"])
    return model,trainer,trace_events,{"completed_budget_index":index,"exact_optimized_tokens":trainer.tokens_seen,"optimizer_step":trainer.optimizer_step,"model_state_sha256":_model_state_hash(model),"trainer_state_sha256":_trainer_state_hash(trainer),"trace_sha256":payload["trace_sha256"],"fresh_objects":True}


def _controls(repo_root:Path, *, source_sha:str, token_budgets:tuple[int,...], batch_size:int, sequence_length:int, seed:int, threads:int, max_steps:int) -> dict[str,Any]:
    init=InitSpec(); cfg=_trainer_config(max_steps=max_steps,seed=seed)
    return {"source_sha":source_sha,"token_budgets":list(token_budgets),"batch_size":batch_size,"sequence_length":sequence_length,"valid_causal_token_capacity_per_full_step":batch_size*(sequence_length-1),"tokenizer_id":BYTE_TOKENIZER_VERSION,"tokenizer_config_sha256":BYTE_TOKENIZER_HASH,"vocab_sha256":BYTE_VOCAB_HASH,"vocab_size":256,"model_max_seq_len":256,"research41_parent_packing_id":RESEARCH41_PACKING_ID,"trace_protocol":TRACE_PROTOCOL,"train_sha256":_file_sha256(repo_root/"data/s0/packaged/train.jsonl"),"validation_sha256":_file_sha256(repo_root/"data/s0/packaged/validation.jsonl"),"manifest_sha256":_file_sha256(repo_root/"data/s0/packaged/manifest.json"),"init_spec":init.to_dict(),"init_identity_sha256":init.identity_sha256(),"trainer_config":asdict(cfg),"precision":"fp32","seed":seed,"torch_threads":threads,"evaluation_optimized_tokens":0}


def run_candidate(*, repo_root:Path, source_sha:str, family:str, candidate_id:str, output_path:Path, checkpoint_dir:Path, token_budgets:tuple[int,...]=DEFAULT_BUDGETS, batch_size:int=DEFAULT_BATCH_SIZE, sequence_length:int=DEFAULT_SEQUENCE_LENGTH, seed:int=DEFAULT_SEED, torch_threads:int=DEFAULT_THREADS, exercise_resume:bool=True) -> dict[str,Any]:
    token_budgets=_validate_budgets(token_budgets)
    observed=_git_head(repo_root)
    if observed!=source_sha: raise RuntimeError(f"exact-checkout mismatch: expected {source_sha}, observed {observed}")
    specs=candidate_specs(family)
    if candidate_id not in specs: raise ValueError(f"unknown candidate_id {candidate_id!r} for {family}")
    if batch_size<=0 or sequence_length<2 or sequence_length>256 or torch_threads<=0: raise ValueError("invalid execution geometry")
    torch.set_num_threads(torch_threads); torch.use_deterministic_algorithms(True)
    capacity=batch_size*(sequence_length-1); max_steps=_steps_for_budgets(token_budgets,capacity)
    controls=_controls(repo_root,source_sha=source_sha,token_budgets=token_budgets,batch_size=batch_size,sequence_length=sequence_length,seed=seed,threads=torch_threads,max_steps=max_steps); controls_sha256=_canonical_hash(controls)
    tokenizer=ByteTokenizer(); train_records=_read_jsonl(repo_root/"data/s0/packaged/train.jsonl"); validation_records=_read_jsonl(repo_root/"data/s0/packaged/validation.jsonl")
    train_ids={str(r["id"]) for r in train_records}; validation_ids={str(r["id"]) for r in validation_records}; overlap=sorted(train_ids&validation_ids)
    if overlap: raise RuntimeError(f"train/validation record overlap: {overlap!r}")
    train_stream=_byte_stream(train_records,tokenizer); spec=specs[candidate_id]; init_spec=InitSpec()
    random.seed(seed); torch.manual_seed(seed); model=TwelveSixDecoder(spec,init_spec); trainer=Trainer(model,_trainer_config(max_steps=max_steps,seed=seed),device="cpu")
    probe=_make_batch(train_stream,step=0,batch_size=batch_size,sequence_length=sequence_length); activation=_activation_probe(model,probe)
    before_model=_model_state_hash(model); before_counters=(trainer.tokens_seen,trainer.optimizer_step); initial_loss,validation_tokens=_validation_loss(model,validation_records,tokenizer)
    if (trainer.tokens_seen,trainer.optimizer_step)!=before_counters or _model_state_hash(model)!=before_model: raise RuntimeError("initial evaluation mutated training state")
    trace_events=[]; grad_norms=[]; update_ratios=[]; update_l2s=[]; update_max_abs=[]; changed_elements=[]; clip_events=0; optimization_wall=0.0; checkpoints=[]; resume_events=[]; experiment_started=time.perf_counter(); previous_budget=0
    for budget_index,budget in enumerate(token_budgets):
        expected_segment_steps=math.ceil((budget-previous_budget)/capacity); segment_steps=0
        while trainer.tokens_seen<budget:
            remaining=budget-trainer.tokens_seen; valid_tokens=min(capacity,remaining); step_before=trainer.optimizer_step; tokens_before=trainer.tokens_seen
            raw=_make_batch(train_stream,step=step_before,batch_size=batch_size,sequence_length=sequence_length); batch=_aligned_batch(raw,valid_tokens); before,weight_norm=_parameter_snapshot(model)
            started=time.perf_counter(); metrics=trainer.train_microbatch(batch); optimization_wall += time.perf_counter()-started
            delta_l2,update_ratio,max_abs,changed=_update_stats(model,before,weight_norm); del before
            if not metrics.optimizer_stepped or metrics.tokens!=valid_tokens or trainer.tokens_seen!=tokens_before+valid_tokens or trainer.tokens_seen>budget or trainer.optimizer_step!=step_before+1: raise RuntimeError("strict token/update accounting drift")
            if metrics.grad_norm is None or not math.isfinite(metrics.grad_norm) or not math.isfinite(update_ratio): raise RuntimeError("non-finite gradient/update statistic")
            grad_norms.append(float(metrics.grad_norm)); update_ratios.append(update_ratio); update_l2s.append(delta_l2); update_max_abs.append(max_abs); changed_elements.append(changed); clip_events += int(metrics.grad_norm>1.0)
            trace_events.append({"optimizer_step":trainer.optimizer_step,"budget_index":budget_index,"budget":budget,"valid_causal_tokens":valid_tokens,"cumulative_optimized_tokens":trainer.tokens_seen,"input_sha256":_tensor_sha256(raw),"target_sha256":_tensor_sha256(batch["target_ids"]),"loss_mask_sha256":_tensor_sha256(batch["loss_mask"])})
            segment_steps += 1
        if trainer.tokens_seen!=budget or segment_steps!=expected_segment_steps: raise RuntimeError("failed exact token-budget landing")
        model_hash=_model_state_hash(model); trainer_hash=_trainer_state_hash(trainer); counters=(trainer.tokens_seen,trainer.optimizer_step); eval_started=time.perf_counter(); validation_loss,checked_tokens=_validation_loss(model,validation_records,tokenizer); evaluation_wall=time.perf_counter()-eval_started
        if checked_tokens!=validation_tokens or (trainer.tokens_seen,trainer.optimizer_step)!=counters or _model_state_hash(model)!=model_hash or _trainer_state_hash(trainer)!=trainer_hash: raise RuntimeError("evaluation leaked into optimized state")
        checkpoint_path=checkpoint_dir/f"{family}-{candidate_id}-tokens-{budget}.pt"; checkpoint_evidence=_save_checkpoint(checkpoint_path,controls_sha256=controls_sha256,source_sha=source_sha,family=family,candidate_id=candidate_id,model=model,trainer=trainer,completed_budget_index=budget_index,token_budgets=token_budgets,trace_events=trace_events)
        checkpoints.append({"requested_token_budget":budget,"optimized_tokens":trainer.tokens_seen,"optimizer_steps":trainer.optimizer_step,"validation_loss":validation_loss,"validation_bpb":validation_loss/math.log(2.0),"validation_tokens":checked_tokens,"evaluation_optimized_tokens":0,"compute_proxy":6*spec.parameter_count()*trainer.tokens_seen,"optimization_wall_seconds":optimization_wall,"evaluation_wall_seconds":evaluation_wall,"checkpoint":checkpoint_evidence})
        if exercise_resume and budget_index==0 and len(token_budgets)>1:
            expected_model_hash=_model_state_hash(model); expected_trainer_hash=_trainer_state_hash(trainer); del trainer,model; gc.collect()
            model,trainer,trace_events,resume=_fresh_resume(checkpoint_path,expected_controls_sha256=controls_sha256,expected_source_sha=source_sha,expected_family=family,expected_candidate_id=candidate_id,spec=spec,init_spec=init_spec,max_steps=max_steps,seed=seed,expected_token_budgets=token_budgets)
            if resume["model_state_sha256"]!=expected_model_hash or resume["trainer_state_sha256"]!=expected_trainer_hash: raise RuntimeError("fresh-resume hash mismatch")
            resume_events.append(resume)
        previous_budget=budget
    experiment_wall=time.perf_counter()-experiment_started
    if trainer.tokens_seen!=token_budgets[-1] or trainer.optimizer_step!=max_steps: raise RuntimeError("final accounting drift")
    final=checkpoints[-1]
    report={"schema":SCHEMA,"authority":AUTHORITY,"source_sha":source_sha,"family":family,"candidate_id":candidate_id,"controls":controls,"controls_sha256":controls_sha256,"data":{"train_records":len(train_records),"validation_records":len(validation_records),"train_validation_record_overlap":overlap,"train_stream_bytes":len(train_stream)},"model_spec":spec.to_dict(),"model_identity_sha256":spec.identity_sha256(),"parameters":spec.parameter_count(),"parameter_allocation":spec.parameter_breakdown(),"init_spec":init_spec.to_dict(),"init_identity_sha256":init_spec.identity_sha256(),"initial_activation_scale":activation,"initial_validation_loss":initial_loss,"initial_validation_bpb":initial_loss/math.log(2.0),"validation_tokens":validation_tokens,"checkpoints":checkpoints,"trace_sha256":_canonical_hash(trace_events),"trace_steps":len(trace_events),"resume_exercised":bool(resume_events),"resume_events":resume_events,"gradient_norm":_summary(grad_norms),"clip_frequency":clip_events/len(grad_norms) if grad_norms else 0.0,"update_ratio":_summary(update_ratios),"update_l2":_summary(update_l2s),"update_max_abs":_summary(update_max_abs),"changed_parameter_elements_per_step":_summary([float(v) for v in changed_elements]),"optimization_wall_seconds":optimization_wall,"experiment_wall_seconds":experiment_wall,"optimized_tokens_per_optimization_second":trainer.tokens_seen/optimization_wall if optimization_wall>0 else None,"memory":{"process_rss_hwm_mib":_rss_hwm_mib(),"model_parameter_tensor_bytes":_parameter_tensor_bytes(model),"optimizer_tensor_bytes":_optimizer_tensor_bytes(trainer)},"final_model_state_sha256":_model_state_hash(model),"final_trainer_state_sha256":_trainer_state_hash(trainer),"final_validation_improvement":initial_loss-float(final["validation_loss"]),"truth_boundary":["Project-authored tiny S0 EN/UK fixture is recycled; this is controlled local evidence, not representative broad-corpus quality.","Only valid causal loss targets increment optimized tokens; held-out evaluation never mutates Trainer counters or optimizer state.","No paid compute, foreign pretrained weights, stage promotion, or capability claim is authorized by this report."],"runtime":{"python":platform.python_version(),"torch":torch.__version__,"platform":platform.platform(),"device":"cpu","paid_compute":False}}
    report["report_sha256"]=_canonical_hash(report); output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return report


def _validate_candidate(report:dict[str,Any], *, expected_source_sha:str|None=None)->None:
    if report.get("schema")!=SCHEMA: raise ValueError("candidate report schema mismatch")
    claimed=report.get("report_sha256"); material=dict(report); material.pop("report_sha256",None)
    if claimed!=_canonical_hash(material): raise ValueError("candidate report self-hash mismatch")
    if expected_source_sha is not None and report.get("source_sha")!=expected_source_sha: raise ValueError("candidate report source SHA mismatch")
    budgets=tuple(int(v) for v in report["controls"]["token_budgets"]); checkpoints=report["checkpoints"]
    if [int(p["optimized_tokens"]) for p in checkpoints]!=list(budgets): raise ValueError("token-budget overshoot/drift")
    if any(int(p["evaluation_optimized_tokens"])!=0 for p in checkpoints): raise ValueError("evaluation tokens counted as optimized")
    if len(budgets)>1 and not report.get("resume_exercised"): raise ValueError("multi-budget run did not exercise fresh resume")


def _dominance_status(reports:list[dict[str,Any]])->dict[str,dict[str,Any]]:
    result={}
    for report in reports:
        candidate=str(report["candidate_id"])
        if report["initial_activation_scale"]["status"]!="PASS": result[candidate]={"status":"REJECT_UNSTABLE_INITIALIZATION","dominated_by":[]}; continue
        own_loss=float(report["checkpoints"][-1]["validation_loss"]); own_speed=float(report["optimized_tokens_per_optimization_second"]); own_params=int(report["parameters"]); dominators=[]
        for other in reports:
            if other is report or other["initial_activation_scale"]["status"]!="PASS": continue
            if abs(int(other["parameters"])-own_params)/own_params>0.01: continue
            loss=float(other["checkpoints"][-1]["validation_loss"]); speed=float(other["optimized_tokens_per_optimization_second"])
            if loss<=own_loss and speed>=own_speed and (loss<own_loss or speed>own_speed): dominators.append(str(other["candidate_id"]))
        result[candidate]={"status":"REJECT_DOMINATED_ISO_PARAMETER_BOTTLENECK" if dominators else "PASS","dominated_by":dominators}
    return result


def collect_reports(*, family:str, input_paths:list[Path], output_path:Path, expected_source_sha:str)->dict[str,Any]:
    reports=[json.loads(path.read_text(encoding="utf-8")) for path in input_paths]
    for report in reports:
        _validate_candidate(report,expected_source_sha=expected_source_sha)
        if report["family"]!=family: raise ValueError("mixed experiment families")
    expected=set(candidate_specs(family)); observed={str(r["candidate_id"]) for r in reports}
    if observed!=expected: raise ValueError(f"candidate set mismatch: {observed!r} != {expected!r}")
    controls={str(r["controls_sha256"]) for r in reports}; traces={str(r["trace_sha256"]) for r in reports}
    if len(controls)!=1: raise ValueError("fixed-control identity drift")
    if len(traces)!=1: raise ValueError("exact token-trace drift")
    rows=[]
    for report in reports:
        point=report["checkpoints"][-1]; improvement=float(report["initial_validation_loss"])-float(point["validation_loss"]); parameters=int(report["parameters"]); compute=int(point["compute_proxy"]); wall=float(report["optimization_wall_seconds"])
        rows.append({"candidate_id":report["candidate_id"],"parameters":parameters,"validation_loss":float(point["validation_loss"]),"validation_bpb":float(point["validation_bpb"]),"validation_improvement":improvement,"improvement_per_parameter":improvement/parameters,"improvement_per_compute_proxy":improvement/compute,"improvement_per_optimization_wall_second":improvement/wall,"compute_proxy":compute,"optimization_wall_seconds":wall,"experiment_wall_seconds":float(report["experiment_wall_seconds"]),"tokens_per_second":float(report["optimized_tokens_per_optimization_second"]),"rss_hwm_mib":float(report["memory"]["process_rss_hwm_mib"]),"gradient_norm":report["gradient_norm"],"clip_frequency":float(report["clip_frequency"]),"update_ratio":report["update_ratio"],"initial_activation_scale":report["initial_activation_scale"],"model_identity_sha256":report["model_identity_sha256"],"parameter_allocation":report["parameter_allocation"]})
    ranks={"best_validation":[r["candidate_id"] for r in sorted(rows,key=lambda r:r["validation_loss"])],"validation_improvement_per_parameter":[r["candidate_id"] for r in sorted(rows,key=lambda r:r["improvement_per_parameter"],reverse=True)],"validation_improvement_per_compute":[r["candidate_id"] for r in sorted(rows,key=lambda r:r["improvement_per_compute_proxy"],reverse=True)],"validation_improvement_per_wall_second":[r["candidate_id"] for r in sorted(rows,key=lambda r:r["improvement_per_optimization_wall_second"],reverse=True)]}
    dominance=_dominance_status(reports) if family=="depth_width_100k" else {}; non_rejected=[r for r in rows if not dominance or dominance[str(r["candidate_id"])]["status"]=="PASS"]; recommended=min(non_rejected,key=lambda r:r["validation_loss"])["candidate_id"]
    collection={"schema":COLLECTION_SCHEMA,"authority":AUTHORITY,"source_sha":expected_source_sha,"family":family,"fixed_token_budget":int(reports[0]["controls"]["token_budgets"][-1]),"controls_sha256":next(iter(controls)),"exact_trace_sha256":next(iter(traces)),"rows":sorted(rows,key=lambda r:int(r["parameters"])),"rankings":ranks,"depth_width_rejection":dominance,"recommended_best_validation_non_rejected":recommended,"recommendation_rule":"Lowest held-out validation loss at largest exact token budget among candidates passing initialization and, for iso-parameter geometry, not dominated in both validation and optimized tokens/sec by another <=1%-parameter candidate.","truth_boundary":["Rankings use held-out validation, never train loss, as generalization metric.","The controlled S0 fixture is tiny and repeatedly cycled; recommendation is local to this experiment box.","No paid compute or promotion authority is implied."]}
    collection["report_sha256"]=_canonical_hash(collection); output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_text(json.dumps(collection,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return collection


def _parse_args(argv:list[str]|None=None)->argparse.Namespace:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run-candidate"); run.add_argument("--repo-root",type=Path,default=Path(".")); run.add_argument("--source-sha",required=True); run.add_argument("--family",choices=("scaling","depth_width_100k"),required=True); run.add_argument("--candidate-id",required=True); run.add_argument("--output",type=Path,required=True); run.add_argument("--checkpoint-dir",type=Path,required=True); run.add_argument("--token-budgets",type=int,nargs="+",default=list(DEFAULT_BUDGETS)); run.add_argument("--batch-size",type=int,default=DEFAULT_BATCH_SIZE); run.add_argument("--sequence-length",type=int,default=DEFAULT_SEQUENCE_LENGTH); run.add_argument("--seed",type=int,default=DEFAULT_SEED); run.add_argument("--torch-threads",type=int,default=DEFAULT_THREADS); run.add_argument("--no-exercise-resume",action="store_true")
    collect=sub.add_parser("collect"); collect.add_argument("--family",choices=("scaling","depth_width_100k"),required=True); collect.add_argument("--expected-source-sha",required=True); collect.add_argument("--output",type=Path,required=True); collect.add_argument("inputs",type=Path,nargs="+")
    validate=sub.add_parser("validate"); validate.add_argument("path",type=Path); validate.add_argument("--expected-source-sha")
    config=sub.add_parser("write-depth-width-config"); config.add_argument("--output",type=Path,required=True)
    return parser.parse_args(argv)


def main(argv:list[str]|None=None)->int:
    args=_parse_args(argv)
    if args.command=="run-candidate":
        report=run_candidate(repo_root=args.repo_root.resolve(),source_sha=args.source_sha,family=args.family,candidate_id=args.candidate_id,output_path=args.output,checkpoint_dir=args.checkpoint_dir,token_budgets=tuple(args.token_budgets),batch_size=args.batch_size,sequence_length=args.sequence_length,seed=args.seed,torch_threads=args.torch_threads,exercise_resume=not args.no_exercise_resume); print(json.dumps({"candidate_id":report["candidate_id"],"report_sha256":report["report_sha256"]},sort_keys=True)); return 0
    if args.command=="collect":
        report=collect_reports(family=args.family,input_paths=list(args.inputs),output_path=args.output,expected_source_sha=args.expected_source_sha); print(json.dumps({"family":report["family"],"rankings":report["rankings"],"recommended":report["recommended_best_validation_non_rejected"]},indent=2,sort_keys=True)); return 0
    if args.command=="validate":
        report=json.loads(args.path.read_text(encoding="utf-8"))
        if report.get("schema")==SCHEMA: _validate_candidate(report,expected_source_sha=args.expected_source_sha)
        elif report.get("schema")==COLLECTION_SCHEMA:
            claimed=report.get("report_sha256"); material=dict(report); material.pop("report_sha256",None)
            if claimed!=_canonical_hash(material): raise ValueError("collection report self-hash mismatch")
            if args.expected_source_sha and report.get("source_sha")!=args.expected_source_sha: raise ValueError("collection source SHA mismatch")
        else: raise ValueError("unknown report schema")
        print(f"{report['schema']}: PASS"); return 0
    if args.command=="write-depth-width-config":
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(config_payload(),indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(args.output); return 0
    raise AssertionError("unreachable")

if __name__=="__main__": sys.exit(main())
