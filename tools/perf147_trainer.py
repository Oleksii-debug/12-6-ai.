#!/usr/bin/env python3
"""End-to-end Trainer timing for objective-identical fixed vs right-trimmed D04 batches."""
from __future__ import annotations

import argparse, hashlib, json, statistics, time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder, load_stage_config
from twelve_six.packing import TextRecord, collate_right_trimmed_rows, collate_rows, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer, TrainerConfig

EXPECTED_CORPUS_ID="422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
BATCH=8
MIXTURE=("uk","en","uk","code","en","uk","en","uk","code","uk","en","uk","en","code","uk","en","uk","code","en","uk")


def load_rows(corpus: Path, manifest: dict[str,Any]) -> list[dict[str,Any]]:
    out=[]
    for shard in manifest["shards"]:
        with (corpus/shard["path"]).open("r",encoding="utf-8") as h:
            for raw in h:
                if raw.strip():
                    row=json.loads(raw)
                    if row["split"]=="train": out.append(row)
    return out


def spec500() -> ModelSpec:
    spec=ModelSpec(schema_version=1,vocab_size=256,max_seq_len=256,d_model=96,n_layers=4,n_heads=6,n_kv_heads=6,head_dim=16,d_ff=256,rope_rotary_dim=16)
    if spec.parameter_count()!=467_808: raise RuntimeError("500K fixed-control drift")
    return spec


def config(steps:int,seed:int=1337) -> TrainerConfig:
    return TrainerConfig(learning_rate=3e-4,weight_decay=0.0,betas=(0.9,0.95),eps=1e-8,max_steps=steps,warmup_steps=0,
        scheduler="constant",gradient_accumulation_steps=1,gradient_clip_norm=1.0,precision="fp32",seed=seed,
        deterministic_algorithms=True,deterministic_warn_only=False)


def digest_examples(examples) -> bytes:
    h=hashlib.sha256()
    for ex in examples:
        actual=sum(ex.attention_mask); h.update(ex.record_ids[0].encode()+b"\0"); h.update(bytes(ex.input_ids[:actual])); h.update(actual.to_bytes(4,"big"))
    return h.digest()


def run(data:list[dict[str,Any]],spec:ModelSpec,context:int,steps:int,policy:str,seed:int=1337) -> dict[str,Any]:
    grouped={s:[r for r in data if r["stratum"]==s] for s in ("uk","en","code")}; tok=ByteTokenizer()
    def new_iter(s:str):
        return iter_packed_examples((TextRecord(str(r["record_id"]),str(r["text"]),"train") for r in grouped[s]),tok,
            expected_split="train",sequence_length=context,cross_document=False)
    iters={s:new_iter(s) for s in grouped}; torch.manual_seed(seed); model=TwelveSixDecoder(spec,InitSpec()); trainer=Trainer(model,config(steps,seed),device="cpu")
    data_s=train_s=0.0; optimized=0; widths=[]; losses=[]; trace=hashlib.sha256(); consumed=Counter()
    for step in range(steps):
        s=MIXTURE[step%len(MIXTURE)]; t0=time.perf_counter(); examples=[next(iters[s]) for _ in range(BATCH)]; consumed[s]+=BATCH; trace.update(digest_examples(examples))
        rows=collate_rows(examples,target_mode="labels") if policy=="fixed" else collate_right_trimmed_rows(examples,target_mode="labels")
        batch={"input_ids":torch.tensor(rows["input_ids"],dtype=torch.long),"labels":torch.tensor(rows["labels"],dtype=torch.long)}; widths.append(batch["input_ids"].shape[1]); t1=time.perf_counter()
        metrics=trainer.train_microbatch(batch); t2=time.perf_counter(); data_s+=t1-t0; train_s+=t2-t1; optimized+=metrics.tokens; losses.append(metrics.loss)
    restart=True
    for s in grouped:
        fresh=new_iter(s)
        for _ in range(consumed[s]): next(fresh)
        expected=[next(iters[s]) for _ in range(BATCH)]; rebuilt=[next(fresh) for _ in range(BATCH)]
        restart = restart and digest_examples(expected)==digest_examples(rebuilt)
    total=data_s+train_s
    return {"policy":policy,"parameters":spec.parameter_count(),"model_identity_sha256":spec.identity_sha256(),"context":context,"steps":steps,"batch_size":BATCH,
        "optimized_tokens":optimized,"batch_trace_sha256":trace.hexdigest(),"data_restart_exact":restart,"data_wait_seconds":data_s,"trainer_seconds":train_s,
        "end_to_end_seconds":total,"data_wait_pct":100*data_s/total,"optimized_tokens_per_second":optimized/total,"trainer_only_optimized_tokens_per_second":optimized/train_s,
        "mean_tensor_width":statistics.mean(widths),"min_tensor_width":min(widths),"max_tensor_width":max(widths),"losses":losses}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--output",type=Path,default=Path("artifacts/perf147-trainer.json")); p.add_argument("--steps",type=int,default=8); a=p.parse_args()
    torch.set_num_threads(1); repo=a.repo.resolve(); corpus=(repo/"artifacts/perf147-trainer-corpus"); manifest=build_corpus(repo/"configs/data/corpus_v01.json",corpus)
    retained=json.loads((repo/"data/corpus/v0.1/manifest.json").read_text(encoding="utf-8"))
    if manifest!=retained or manifest["corpus_identity_sha256"]!=EXPECTED_CORPUS_ID: raise RuntimeError("DATA-25 identity drift")
    data=load_rows(corpus,manifest); cases=(("500k",spec500(),256),("1m",load_stage_config(repo/"configs/stages/s2_1m.json").model,512)); results=[]
    for label,spec,context in cases:
        fixed=run(data,spec,context,a.steps,"fixed"); trim=run(data,spec,context,a.steps,"right_trim")
        if fixed["batch_trace_sha256"]!=trim["batch_trace_sha256"]: raise RuntimeError("candidate changed training trace")
        deltas=[abs(x-y) for x,y in zip(fixed["losses"],trim["losses"])]
        results.append({"scale":label,"fixed":fixed,"right_trim":trim,"speedup_pct":100*(trim["optimized_tokens_per_second"]/fixed["optimized_tokens_per_second"]-1),
            "loss_trace_exact":fixed["losses"]==trim["losses"],"max_loss_abs_delta":max(deltas)})
    report={"schema_version":"12-6.perf147-trainer.v1","corpus_identity_sha256":EXPECTED_CORPUS_ID,"torch":torch.__version__,"torch_num_threads":torch.get_num_threads(),
        "paid_compute":False,"runs":results}; out=repo/a.output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8"); print(json.dumps(report,sort_keys=True)); return 0

if __name__=="__main__": raise SystemExit(main())
