#!/usr/bin/env python3
"""Full-corpus PERF-147 D04 utilization and provenance benchmark."""
from __future__ import annotations

import argparse, hashlib, json, math, statistics, time
from pathlib import Path
from typing import Any

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.packing import TextRecord, document_window_spans, iter_packed_examples
from twelve_six.tokenization import ByteTokenizer

EXPECTED_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
CONTEXTS = (128, 256, 512, 1024)
BATCH = 8


def line(v: object) -> bytes:
    return (json.dumps(v, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def rows(corpus: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for shard in manifest["shards"]:
        with (corpus / shard["path"]).open("r", encoding="utf-8") as handle:
            for raw in handle:
                if raw.strip():
                    row = json.loads(raw)
                    if row["split"] == "train":
                        out.append(row)
    return out


def percentile(values: list[int], q: float) -> float:
    x = sorted(values); p = (len(x) - 1) * q; lo = math.floor(p); hi = math.ceil(p)
    return float(x[lo]) if lo == hi else x[lo] * (hi - p) + x[hi] * (p - lo)


def distribution(data: list[dict[str, Any]]) -> dict[str, Any]:
    x = [int(r["byte_tokens"]) for r in data]
    out: dict[str, Any] = {"documents": len(x), "byte_tokens": sum(x), "min": min(x),
        "p01": percentile(x,.01), "p10": percentile(x,.10), "p25": percentile(x,.25),
        "p50": percentile(x,.50), "mean": statistics.mean(x), "p75": percentile(x,.75),
        "p90": percentile(x,.90), "p99": percentile(x,.99), "max": max(x), "by_stratum": {}}
    for s in ("uk","en","code"):
        y = [int(r["byte_tokens"]) for r in data if r["stratum"] == s]
        out["by_stratum"][s] = {"documents":len(y),"byte_tokens":sum(y),"min":min(y),
            "p50":percentile(y,.5),"mean":statistics.mean(y),"p90":percentile(y,.9),"max":max(y)}
    return out


def span_iter(data: list[dict[str, Any]], context: int):
    for ri, row in enumerate(data):
        for span in document_window_spans(str(row["record_id"]), int(row["byte_tokens"]), sequence_length=context):
            yield ri, span


def mapping(out: Path, data: list[dict[str, Any]], context: int) -> dict[str, Any]:
    path = out / f"mapping-{context}.jsonl"; h = hashlib.sha256(); suffix = hashlib.sha256()
    total = sum(len(document_window_spans(str(r["record_id"]), int(r["byte_tokens"]), sequence_length=context)) for r in data)
    cursor = total // 2; started = time.perf_counter()
    with path.open("wb") as handle:
        for bi, (ri, span) in enumerate(span_iter(data, context)):
            payload = line([bi,ri,span.source_start,span.source_end,span.actual_length]); handle.write(payload); h.update(payload)
            if bi >= cursor: suffix.update(payload)
    expected = suffix.hexdigest(); rebuilt = hashlib.sha256()
    for bi, (ri, span) in enumerate(span_iter(data, context)):
        if bi >= cursor: rebuilt.update(line([bi,ri,span.source_start,span.source_end,span.actual_length]))
    if rebuilt.hexdigest() != expected: raise RuntimeError("restart suffix mismatch")
    return {"path":path.name,"bytes":path.stat().st_size,"sha256":h.hexdigest(),"blocks":total,
        "restart_cursor":cursor,"restart_suffix_sha256":expected,"restart_exact":True,
        "seconds":time.perf_counter()-started,
        "rule":"p<actual_length => record_table[record_index], source_start+p; otherwise padding/null"}


def measure(data: list[dict[str, Any]], context: int) -> dict[str, Any]:
    started = time.perf_counter(); cpu = time.process_time(); actual=[]; optimized=0; blocks=0; docs_per_block=set()
    records = (TextRecord(str(r["record_id"]), str(r["text"]), "train") for r in data)
    for ex in iter_packed_examples(records, ByteTokenizer(), expected_split="train", sequence_length=context, cross_document=False):
        actual.append(sum(ex.attention_mask)); optimized += ex.num_loss_tokens; blocks += 1; docs_per_block.add(len(ex.record_ids))
    wall = time.perf_counter()-started; cpu = time.process_time()-cpu
    if docs_per_block != {1}: raise RuntimeError("document boundary leak")
    source = sum(int(r["byte_tokens"]) for r in data); expected = source-len(data)
    if optimized != expected: raise RuntimeError("optimized causal-pair count drift")
    tensor=blocks*context; semantic=sum(actual); padding=tensor-semantic; overlap=semantic-source
    trimmed=sum(max(actual[i:i+BATCH])*len(actual[i:i+BATCH]) for i in range(0,len(actual),BATCH))
    drop_tail=0
    for r in data:
        spans=document_window_spans(str(r["record_id"]),int(r["byte_tokens"]),sequence_length=context)
        if spans and spans[-1].padding_length: drop_tail += spans[-1].optimized_pairs
    return {"context":context,"documents":len(data),"blocks":blocks,"blocks_per_document":blocks/len(data),
        "distinct_documents_per_block":1.0,"source_tokens":source,"optimized_loss_tokens":optimized,
        "terminal_tail_tokens":len(data),"terminal_tail_pct_source":100*len(data)/source,
        "tensor_input_tokens":tensor,"actual_input_positions":semantic,"padding_tokens":padding,
        "padding_waste_pct_tensor":100*padding/tensor,"window_overlap_tokens":overlap,
        "window_overlap_pct_source":100*overlap/source,"input_tokens_per_optimized_loss_token":tensor/optimized,
        "semantic_input_positions_per_optimized_loss_token":semantic/optimized,"packing_wall_seconds":wall,
        "packing_cpu_seconds":cpu,"packing_optimized_tokens_per_cpu_second":optimized/cpu,
        "right_trim_same_order":{"eligible":True,"tensor_input_tokens":trimmed,
            "tensor_token_reduction_pct":100*(tensor-trimmed)/tensor,"input_tokens_per_optimized_loss_token":trimmed/optimized,
            "pair_trace_change":False,"block_order_change":False},
        "rejected":{"no_overlap":{"missing_pairs":sum((int(r["byte_tokens"])-1)//context for r in data)},
            "drop_tail":{"missing_pairs":drop_tail},"naive_cross_document":{"invalid_pairs":len(data)-1}}}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--repo",type=Path,default=Path.cwd()); p.add_argument("--output",type=Path,default=Path("artifacts/perf147-static")); a=p.parse_args()
    repo=a.repo.resolve(); out=(repo/a.output).resolve(); out.mkdir(parents=True,exist_ok=True); corpus=out/"corpus"
    manifest=build_corpus(repo/"configs/data/corpus_v01.json", corpus)
    retained=json.loads((repo/"data/corpus/v0.1/manifest.json").read_text(encoding="utf-8"))
    if manifest != retained or manifest["corpus_identity_sha256"] != EXPECTED_CORPUS_ID: raise RuntimeError("DATA-25 identity drift")
    data=rows(corpus,manifest); table=out/"record-table.jsonl"
    with table.open("wb") as h:
        for i,r in enumerate(data): h.write(line([i,r["record_id"],r["stratum"],r["byte_tokens"]]))
    report={"schema_version":"12-6.perf147-static.v1","corpus_identity_sha256":EXPECTED_CORPUS_ID,
        "truth_boundary":manifest["truth_boundary"],"packing_identity_unchanged":True,"document_boundary_policy":"isolate",
        "length_distribution":distribution(data),"contexts":{},"mappings":{},"record_table":table.name}
    for c in CONTEXTS: report["contexts"][str(c)]=measure(data,c); report["mappings"][str(c)]=mapping(out,data,c)
    (out/"report.json").write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
