"""DATA-184 real-source diversity experiment.

Fail-closed, bounded experiment only. It reconstructs the accepted real-source
control set from immutable M100 pins, adds independent Ukrainian/English/code
families only after DATA-24 purpose-specific rights resolution, cross-family
near-deduplicates, and runs a matched 467,808-parameter scratch-Base A/B.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import resource
import statistics
import subprocess
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch

from .data.document_quality import assess_document
from .data.external_sources import (
    EligibilityResolver,
    build_external_source_registry,
    external_source_from_mapping,
)
from .data.privacy_filter import scan_record
from .data.source_intake import DownloadedBytes, extract_text
from .model import InitSpec, TwelveSixDecoder
from .scaling_500k_evidence import _model_state_sha256, _target_spec
from .scaling_experiment import _byte_stream, _make_batch, _trainer_config, _validation_loss
from .tokenization import BYTE_TOKENIZER_HASH, BYTE_TOKENIZER_VERSION, BYTE_VOCAB_HASH, ByteTokenizer
from .training import Trainer

SCHEMA = "12-6.data184-real-source-diversity-report.v1"
CONFIG_SCHEMA = "12-6.data184-real-source-diversity.v1"
PROJECT_SCHEMA = "12-6.data184-parent-project-code.v1"
POLICY_REF = "policy://12-6/data/explicit-model-training-evidence-v1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
TOKENS = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def sha256b(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256f(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_hash(value: Any) -> str:
    return sha256b(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def http_get(url: str, limit: int = 64 << 20) -> bytes:
    req = Request(url, headers={"User-Agent": "12-6-data184/1.0", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=120) as r:
        size = r.headers.get("Content-Length")
        if size and int(size) > limit:
            raise RuntimeError(f"{url}: declared size exceeds limit")
        value = r.read(limit + 1)
    if len(value) > limit:
        raise RuntimeError(f"{url}: payload exceeds limit")
    return value


def git_blob_sha1(value: bytes) -> str:
    return hashlib.sha1(f"blob {len(value)}\0".encode() + value).hexdigest()


def evidence_sha(root: Path, source: dict[str, Any]) -> str:
    got = sha256f(root / source["rights_evidence_path"])
    if got != source["rights_evidence_sha256"]:
        raise RuntimeError(f"{source['source_id']}: rights evidence drift")
    return got


def probe_rights(source: dict[str, Any]) -> dict[str, Any]:
    url = source["terms_url"]
    sid = source["source_id"]
    if sid == "hf:lang-uk/perestoroha-ocr":
        raw = url.replace("/blob/", "/raw/")
        payload = http_get(raw, 256 << 10)
        text = payload.decode("utf-8")
        for needle in ("license: cc-by-4.0", "public domain", "released under"):
            if needle not in text.casefold():
                raise RuntimeError(f"{sid}: immutable dataset-card rights facts drifted")
        return {"url": raw, "sha256": sha256b(payload), "checks": ["cc-by-4.0", "public domain"]}
    payload = http_get(url, 512 << 10)
    if source.get("license_sha256") and sha256b(payload) != source["license_sha256"]:
        raise RuntimeError(f"{sid}: license SHA-256 drift")
    if source.get("license_blob_sha1") and git_blob_sha1(payload) != source["license_blob_sha1"]:
        raise RuntimeError(f"{sid}: license Git-blob drift")
    return {"url": url, "sha256": sha256b(payload), "git_blob_sha1": git_blob_sha1(payload)}


def make_spec(
    *,
    source: dict[str, Any],
    source_id: str,
    version: str,
    source_url: str,
    snapshot_uri: str,
    raw_sha: str,
    raw_bytes: int,
    rights_sha: str,
    evidence_kind: str = "explicit_permission",
) -> Any:
    refs = [
        {
            "evidence_id": f"{source_id}:rights",
            "evidence_kind": evidence_kind,
            "uri": f"file:{source['rights_evidence_path']}",
            "sha256": rights_sha,
            "captured_at": "2026-08-26T00:00:00+03:00",
            "source_id": source_id,
            "source_version": version,
        },
        {
            "evidence_id": f"{source_id}:policy",
            "evidence_kind": "policy_decision",
            "uri": f"file:{source['rights_evidence_path']}",
            "sha256": rights_sha,
            "captured_at": "2026-08-26T00:00:00+03:00",
            "source_id": source_id,
            "source_version": version,
        },
    ]
    return external_source_from_mapping(
        {
            "source_id": source_id,
            "source_version": version,
            "provider": source["provider"],
            "source_url": source_url,
            "source_kind": "source_code" if source["modality"] == "code" else "rights_reviewed_real_text",
            "purpose": "pretraining",
            "synthetic": False,
            "benchmark_material": False,
            "held_out": False,
            "snapshot": {
                "uri": snapshot_uri,
                "sha256": raw_sha,
                "size_bytes": raw_bytes,
                "retrieved_at": "2026-08-26",
                "upstream_version": source["source_version"],
                "retrieval_method": "DATA184_EXACT_OBJECT",
            },
            "rights": {
                "status": "APPROVED_FOR_TRAINING",
                "license_id": source["license_id"],
                "terms_url": source["terms_url"],
                "allows_model_training": True,
                "allows_derivatives": True,
                "allows_redistribution": True,
                "policy_ref": POLICY_REF,
                "reviewed_at": "2026-08-26",
                "reviewer_ref": "SWARM_WORKER_ID:DATA-184-REAL-SOURCE-DIVERSITY",
                "uses": {
                    "acquisition": "ALLOWED",
                    "storage": "ALLOWED",
                    "analysis": "ALLOWED",
                    "model_training": "ALLOWED",
                    "redistribution": "ALLOWED",
                },
                "evidence_refs": refs,
            },
        }
    )


def acquire_external(root: Path, config: dict[str, Any], out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    q = out / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    acquired, specs, probes = [], [], {}
    for source in config["sources"]:
        rsha = evidence_sha(root, source)
        probes[source["source_id"]] = probe_rights(source)
        for i, obj in enumerate(source["objects"]):
            url = obj.get("url") or source["raw_url_template"].format(path=obj["path"])
            payload = http_get(url)
            raw, blob = sha256b(payload), git_blob_sha1(payload)
            if obj.get("raw_sha256") and raw != obj["raw_sha256"]:
                raise RuntimeError(f"{source['source_id']}:{i}: raw SHA drift")
            if obj.get("raw_bytes") and len(payload) != int(obj["raw_bytes"]):
                raise RuntimeError(f"{source['source_id']}:{i}: raw size drift")
            if obj.get("git_blob_sha1") and blob != obj["git_blob_sha1"]:
                raise RuntimeError(f"{source['source_id']}:{i}: Git blob drift")
            path = q / f"external-{len(acquired):03d}-{raw[:16]}.bin"
            path.write_bytes(payload)
            oid = f"{source['source_id']}.object-{raw[:16]}"
            ver = f"{source['source_version']}/object:{raw[:16]}"
            spec = make_spec(
                source=source,
                source_id=oid,
                version=ver,
                source_url=url,
                snapshot_uri=f"file:{path.relative_to(root).as_posix()}",
                raw_sha=raw,
                raw_bytes=len(payload),
                rights_sha=rsha,
            )
            specs.append(spec)
            acquired.append(
                {
                    "source": source,
                    "object": obj,
                    "url": url,
                    "payload_path": path,
                    "raw_sha256": raw,
                    "raw_bytes": len(payload),
                    "git_blob_sha1": blob,
                    "d03_source_id": oid,
                    "d03_source_version": ver,
                    "source_manifest_sha256": spec.source_manifest_sha256,
                }
            )
    registry = build_external_source_registry(specs)
    resolver = EligibilityResolver(registry)
    for spec in specs:
        resolver.assert_model_training_eligible(spec.source_id, spec.source_version, spec.source_manifest_sha256)
    return acquired, {
        "registry_identity_sha256": registry["registry_identity_sha256"],
        "eligibility_inventory": resolver.inventory(),
        "rights_probes": probes,
    }


def extract_external(acquired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs = []
    for item in acquired:
        source, obj = item["source"], item["object"]
        payload = Path(item["payload_path"]).read_bytes()
        if obj.get("adapter") == "parquet_transcription":
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("pyarrow required for pinned Perestoroha parquet") from exc
            values = pq.read_table(pa.BufferReader(payload), columns=[obj["field"]])[obj["field"]].to_pylist()
            for j, text in enumerate(values):
                if isinstance(text, str) and text.strip():
                    docs.append(
                        {
                            "id": f"{source['source_id']}:{j}",
                            "text": text,
                            "family": source["family"],
                            "language": source["language"],
                            "modality": source["modality"],
                            "source_id": item["d03_source_id"],
                            "source_version": item["d03_source_version"],
                        }
                    )
            continue
        if source["modality"] == "code":
            text = payload.decode("utf-8")
        else:
            adapter = obj.get("adapter") if obj.get("adapter") in {"html_text", "plain_text"} else "plain_text"
            text, _ = extract_text(DownloadedBytes(payload), adapter)
        docs.append(
            {
                "id": f"{source['source_id']}:{len(docs)}",
                "text": text,
                "family": source["family"],
                "language": source["language"],
                "modality": source["modality"],
                "source_id": item["d03_source_id"],
                "source_version": item["d03_source_version"],
            }
        )
    return docs


def parent_project_source(root: Path, out: Path, cfg_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = load_json(cfg_path)
    if cfg.get("schema_version") != PROJECT_SCHEMA:
        raise RuntimeError("project-code config schema drift")
    parent = cfg["parent_git_sha"]
    if not SHA40.fullmatch(parent):
        raise RuntimeError("project-code parent SHA invalid")
    rsha = sha256f(root / cfg["rights_evidence_path"])
    if rsha != cfg["rights_evidence_sha256"]:
        raise RuntimeError("project-code rights evidence drift")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", parent, "--", "src/twelve_six"], cwd=root, text=True
    ).splitlines()
    paths = sorted(p for p in paths if p.endswith(".py"))
    docs, snapshot = [], []
    for path in paths:
        payload = subprocess.check_output(["git", "show", f"{parent}:{path}"], cwd=root)
        text = payload.decode("utf-8")
        raw = sha256b(payload)
        docs.append(
            {
                "id": "parent-" + hashlib.sha256(path.encode()).hexdigest()[:24],
                "text": text,
                "family": cfg["family"],
                "language": "code",
                "modality": "code",
                "source_id": cfg["source_id"],
                "source_version": cfg["source_version"],
            }
        )
        snapshot.append({"path": path, "sha256": raw, "git_blob_sha1": git_blob_sha1(payload), "text": text})
    q = out / "quarantine"
    q.mkdir(parents=True, exist_ok=True)
    snap = q / f"parent-project-code-{parent}.jsonl"
    with snap.open("w", encoding="utf-8", newline="\n") as f:
        for row in snapshot:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    source = {
        "provider": "Oleksii-debug/12-6-ai.",
        "modality": "code",
        "license_id": cfg["license_id"],
        "terms_url": "https://github.com/Oleksii-debug/12-6-ai.",
        "source_version": cfg["source_version"],
        "rights_evidence_path": cfg["rights_evidence_path"],
    }
    spec = make_spec(
        source=source,
        source_id=cfg["source_id"],
        version=cfg["source_version"],
        source_url="https://github.com/Oleksii-debug/12-6-ai.",
        snapshot_uri=f"file:{snap.relative_to(root).as_posix()}",
        raw_sha=sha256f(snap),
        raw_bytes=snap.stat().st_size,
        rights_sha=rsha,
        evidence_kind="project_authorship",
    )
    registry = build_external_source_registry([spec])
    resolver = EligibilityResolver(registry)
    resolver.assert_model_training_eligible(spec.source_id, spec.source_version, spec.source_manifest_sha256)
    return docs, {
        "registry_identity_sha256": registry["registry_identity_sha256"],
        "eligibility_inventory": resolver.inventory(),
        "source_files": len(paths),
        "snapshot_sha256": sha256f(snap),
        "source_manifest_sha256": spec.source_manifest_sha256,
    }


def utf8_chunks(text: str, target: int) -> list[str]:
    out, cur, size = [], [], 0
    for line in text.splitlines(keepends=True):
        for piece in ([line] if len(line.encode()) <= target else list(line)):
            b = piece.encode()
            if cur and size + len(b) > target:
                out.append("".join(cur))
                cur, size = [], 0
            cur.append(piece)
            size += len(b)
    if cur:
        out.append("".join(cur))
    return [x for x in out if x.strip()]


def chunk_and_filter(docs: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    for d in docs:
        totals[d["family"]] += len(d["text"].encode())
    rows, rejected = [], []
    indexes = Counter()
    for d in docs:
        local = min(target, max(512, totals[d["family"]] // 2))
        for text in utf8_chunks(d["text"], local):
            idx = indexes[d["family"]]
            indexes[d["family"]] += 1
            rid = "d184-" + hashlib.sha256(f"{d['id']}:{idx}".encode()).hexdigest()[:24]
            privacy = scan_record(
                record_id=rid,
                source_id=d["source_id"],
                source_version=d["source_version"],
                modality=d["modality"],
                text=text,
            )
            if not privacy.train_eligible_after_privacy or privacy.sanitized_text is None:
                rejected.append({"id": rid, "family": d["family"], "reason": f"privacy:{privacy.action}"})
                continue
            text = privacy.sanitized_text
            quality = assess_document(rid, text, "code" if d["modality"] == "code" else d["language"])
            if not quality.accepted:
                rejected.append({"id": rid, "family": d["family"], "reason": "quality:" + ",".join(quality.reasons)})
                continue
            rows.append(
                {
                    "id": rid,
                    "text": text,
                    "family": d["family"],
                    "language": d["language"],
                    "modality": d["modality"],
                    "source_id": d["source_id"],
                    "source_version": d["source_version"],
                    "utf8_bytes": len(text.encode()),
                    "text_sha256": sha256b(text.encode()),
                }
            )
    return rows, rejected


def bound_external(rows: list[dict[str, Any]], cap: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    kept, dropped = [], []
    for family in sorted(grouped):
        used = 0
        for i, row in enumerate(sorted(grouped[family], key=lambda x: x["id"])):
            n = row["utf8_bytes"]
            if i >= 2 and used + n > cap:
                dropped.append({"id": row["id"], "family": family, "utf8_bytes": n, "reason": "family_mass_cap"})
            else:
                kept.append(row)
                used += n
    return kept, dropped


def shingles(text: str) -> set[tuple[str, ...]]:
    toks = TOKENS.findall(text.casefold())
    return {tuple(toks[i:i+5]) for i in range(max(1, len(toks)-4))} if toks else set()


def near_dedup(base: list[dict[str, Any]], cand: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept = list(base)
    exact = {r["text_sha256"]: r for r in base}
    sh = {r["id"]: shingles(r["text"]) for r in base}
    accepted, removed = [], []
    for row in sorted(cand, key=lambda x: (x["family"], x["id"])):
        if row["text_sha256"] in exact:
            other = exact[row["text_sha256"]]
            removed.append({"candidate_id": row["id"], "matched_id": other["id"], "reason": "exact_duplicate", "similarity": 1.0})
            continue
        sr, best = shingles(row["text"]), None
        for other in kept:
            if other["family"] == row["family"]:
                continue
            so = sh[other["id"]]
            score = len(sr & so) / len(sr | so) if sr and so else 0.0
            if score >= threshold and (best is None or score > best[0]):
                best = (score, other)
        if best:
            removed.append({"candidate_id": row["id"], "matched_id": best[1]["id"], "reason": "cross_family_near_duplicate", "similarity": best[0]})
            continue
        accepted.append(row)
        kept.append(row)
        exact[row["text_sha256"]] = row
        sh[row["id"]] = sr
    return accepted, removed


def percentile(values: list[int], q: float) -> float:
    x = sorted(values)
    pos = q * (len(x) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return float(x[lo]) if lo == hi else x[lo] * (hi-pos) + x[hi] * (pos-lo)


def diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fam, lang, mod, lengths = Counter(), Counter(), Counter(), []
    for r in rows:
        n = len(r["text"].encode())
        fam[r["family"]] += n
        lang[r["language"]] += n
        mod[r["modality"]] += n
        lengths.append(n)
    total = sum(fam.values())
    shares = {k: v / total for k, v in fam.items()}
    h = -sum(p * math.log(p) for p in shares.values())
    return {
        "families": len(fam),
        "documents": len(rows),
        "token_mass_byte_tokens": total,
        "bytes_by_family": dict(sorted(fam.items())),
        "family_shares": dict(sorted(shares.items())),
        "top_family_share": max(shares.values()),
        "entropy_nats": h,
        "entropy_bits": h / math.log(2),
        "effective_source_count": math.exp(h),
        "language_mass_bytes": dict(sorted(lang.items())),
        "modality_mass_bytes": dict(sorted(mod.items())),
        "document_length_utf8_bytes": {
            "min": min(lengths), "p25": percentile(lengths, .25), "p50": percentile(lengths, .5),
            "mean": statistics.fmean(lengths), "p75": percentile(lengths, .75),
            "p95": percentile(lengths, .95), "max": max(lengths),
        },
    }


def split_family(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    train, held = [], []
    for family in sorted(grouped):
        ranked = sorted(grouped[family], key=lambda r: hashlib.sha256(f"{family}\0{r['text_sha256']}".encode()).hexdigest())
        if len(ranked) < 2:
            raise RuntimeError(f"{family}: requires >=2 chunks")
        n = min(max(1, round(len(ranked) * .2)), len(ranked)-1)
        held.extend(ranked[:n])
        train.extend(ranked[n:])
    return train, held


def round_robin(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    qs = {}
    for f in sorted({r["family"] for r in rows}):
        qs[f] = deque(sorted((r for r in rows if r["family"] == f), key=lambda r: r["id"]))
    out = []
    while any(qs.values()):
        for f in sorted(qs):
            if qs[f]:
                out.append(qs[f].popleft())
    return out


def evaluate(model: TwelveSixDecoder, held: list[dict[str, Any]], tok: ByteTokenizer) -> dict[str, Any]:
    before = _model_state_sha256(model)
    loss, targets = _validation_loss(model, held, tok)
    by_family, per_chunk = {}, {}
    for f in sorted({r["family"] for r in held}):
        subset = [r for r in held if r["family"] == f]
        fl, ft = _validation_loss(model, subset, tok)
        by_family[f] = {"loss_nats": fl, "bpb": fl/math.log(2), "targets": ft, "chunks": len(subset)}
    for r in held:
        cl, ct = _validation_loss(model, [r], tok)
        per_chunk[r["id"]] = {"family": r["family"], "bpb": cl/math.log(2), "targets": ct}
    if before != _model_state_sha256(model):
        raise RuntimeError("evaluation mutated model")
    return {"aggregate_loss_nats": loss, "aggregate_bpb": loss/math.log(2), "aggregate_targets": targets,
            "by_family": by_family, "per_chunk": per_chunk, "evaluation_non_mutation": True}


def run_arm(name: str, rows: list[dict[str, Any]], held: list[dict[str, Any]], requested: int, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    tok, spec, init = ByteTokenizer(), _target_spec(), InitSpec()
    if spec.parameter_count() != 467_808:
        raise RuntimeError("500K control ModelSpec drift")
    bs, seq = 4, 64
    max_steps = math.ceil(requested / (bs * (seq-1)))
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, _trainer_config(max_steps=max_steps, seed=seed), device="cpu")
    initial_state = _model_state_sha256(model)
    initial_eval = evaluate(model, held, tok)
    stream = _byte_stream(round_robin(rows), tok)
    trace = hashlib.sha256()
    curve, t0 = [], time.perf_counter()
    for step in range(max_steps):
        batch = _make_batch(stream, step=step, batch_size=bs, sequence_length=seq)
        trace.update(batch.numpy().tobytes())
        m = trainer.train_microbatch({"input_ids": batch})
        if step == 0 or (step+1) % 64 == 0 or step+1 == max_steps:
            curve.append({"optimizer_step": m.optimizer_step, "optimized_tokens": trainer.tokens_seen,
                          "train_loss": m.update_loss, "grad_norm": m.grad_norm, "learning_rate": m.learning_rate})
    wall = time.perf_counter() - t0
    final_eval = evaluate(model, held, tok)
    return {
        "name": name,
        "model": {"parameters": spec.parameter_count(), "model_spec": spec.to_dict(),
                  "model_spec_sha256": spec.identity_sha256(), "init_spec_sha256": init.identity_sha256(),
                  "tokenizer": BYTE_TOKENIZER_VERSION, "tokenizer_config_sha256": BYTE_TOKENIZER_HASH,
                  "tokenizer_vocab_sha256": BYTE_VOCAB_HASH},
        "run": {"seed": seed, "requested_optimized_tokens": requested, "optimized_tokens": trainer.tokens_seen,
                "optimizer_steps": trainer.optimizer_step, "batch_size": bs, "sequence_length": seq,
                "training_records": len(rows), "training_stream_bytes": len(stream), "wall_seconds": wall,
                "tokens_per_second": trainer.tokens_seen/wall,
                "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                "batch_trace_sha256": trace.hexdigest(), "initial_model_state_sha256": initial_state,
                "final_model_state_sha256": _model_state_sha256(model), "train_curve": curve},
        "initial_evaluation": initial_eval,
        "final_evaluation": final_eval,
    }


def universal_bootstrap(control: dict[str, Any], expanded: dict[str, Any], reps: int, seed: int) -> dict[str, Any]:
    c, e = control["per_chunk"], expanded["per_chunk"]
    ids = sorted(c)
    if ids != sorted(e):
        raise RuntimeError("paired bootstrap heldout identity mismatch")
    diffs = [e[i]["bpb"] - c[i]["bpb"] for i in ids]
    rng = random.Random(seed)
    means = sorted(statistics.fmean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(reps))
    def q(x: float) -> float:
        p = x*(len(means)-1); lo, hi = math.floor(p), math.ceil(p)
        return means[lo] if lo == hi else means[lo]*(hi-p)+means[hi]*(p-lo)
    return {"method": "universal_nonparametric_paired_bootstrap_v1", "unit": "common_heldout_chunk",
            "metric": "expanded_minus_control_mean_chunk_bpb", "replicates": reps, "seed": seed,
            "n_chunks": len(ids), "observed_mean_delta_bpb": statistics.fmean(diffs),
            "ci95": [q(.025), q(.975)], "probability_expanded_better": sum(x < 0 for x in means)/len(means),
            "negative_delta_is_better": True}


def run(root: Path, out: Path, source_sha: str, config_path: Path) -> dict[str, Any]:
    if not SHA40.fullmatch(source_sha) or git_head(root) != source_sha:
        raise RuntimeError("DATA-184 requires exact source SHA")
    cfg = load_json(config_path)
    if cfg.get("schema_version") != CONFIG_SCHEMA:
        raise RuntimeError("DATA-184 config schema drift")
    out.mkdir(parents=True, exist_ok=True)

    latest_manifest = load_json(root / "data/s0/packaged/manifest.json")
    pins_path = root / cfg["base_truth"]["previous_pins"]
    previous_pins = load_json(pins_path)

    acquired, ext_d03 = acquire_external(root, cfg, out)
    ext_docs = extract_external(acquired)
    ext_chunks, rejections = chunk_and_filter(ext_docs, int(cfg["split"]["target_chunk_utf8_bytes"]))
    cap = int(cfg.get("max_admitted_utf8_bytes_per_external_family", 262_144))
    ext_chunks, bound_drops = bound_external(ext_chunks, cap)

    incumbent_families = {s["family"] for s in cfg["sources"] if s["role"] == "incumbent"}
    new_families = {s["family"] for s in cfg["sources"] if s["role"] == "new"}
    if (incumbent_families | new_families) - {r["family"] for r in ext_chunks}:
        raise RuntimeError("an external family vanished before dedup")

    incumbent_raw = [r for r in ext_chunks if r["family"] in incumbent_families]
    new_raw = [r for r in ext_chunks if r["family"] in new_families]
    incumbent, d0 = near_dedup([], incumbent_raw, float(cfg["dedup"]["threshold"]))

    project_docs, project_d03 = parent_project_source(root, out, root/"configs/data/data184_parent_project_code_v1.json")
    project_chunks, project_rej = chunk_and_filter(project_docs, int(cfg["split"]["target_chunk_utf8_bytes"]))
    project, d1 = near_dedup(incumbent, project_chunks, float(cfg["dedup"]["threshold"]))
    new, d2 = near_dedup(incumbent + project, new_raw, float(cfg["dedup"]["threshold"]))
    if incumbent_families - {r["family"] for r in incumbent}:
        raise RuntimeError("incumbent external family vanished after dedup")
    if not project:
        raise RuntimeError("project code vanished after dedup")
    if new_families - {r["family"] for r in new}:
        raise RuntimeError("new external family vanished after dedup")

    incumbent_all = incumbent + project
    control_train, control_held = split_family(incumbent_all)
    new_train, new_held = split_family(new)
    common_held = control_held + new_held
    expanded_train = control_train + new_train

    requested, seed = int(cfg["base_truth"]["ab_equal_requested_optimized_tokens"]), int(cfg["base_truth"]["seed"])
    control = run_arm("previous_source_family_control", control_train, common_held, requested, seed)
    expanded = run_arm("diversity_expanded", expanded_train, common_held, requested, seed)
    if control["run"]["optimized_tokens"] != expanded["run"]["optimized_tokens"]:
        raise RuntimeError("A/B optimized token mismatch")
    if control["run"]["initial_model_state_sha256"] != expanded["run"]["initial_model_state_sha256"]:
        raise RuntimeError("A/B initialization mismatch")
    if control["initial_evaluation"]["aggregate_bpb"] != expanded["initial_evaluation"]["aggregate_bpb"]:
        raise RuntimeError("A/B initial evaluation mismatch")

    raw_all, raw_new = Counter(), Counter()
    objects = []
    for x in acquired:
        sid = x["source"]["source_id"]
        raw_all[sid] += x["raw_bytes"]
        if x["source"]["role"] == "new":
            raw_new[sid] += x["raw_bytes"]
        objects.append({"source_id": sid, "role": x["source"]["role"], "family": x["source"]["family"],
                        "url": x["url"], "raw_sha256": x["raw_sha256"], "raw_bytes": x["raw_bytes"],
                        "git_blob_sha1": x["git_blob_sha1"], "d03_source_id": x["d03_source_id"],
                        "d03_source_version": x["d03_source_version"],
                        "source_manifest_sha256": x["source_manifest_sha256"]})
    admitted_new = Counter()
    for r in new:
        admitted_new[r["family"]] += r["utf8_bytes"]

    control_div, expanded_div = diversity(control_train), diversity(expanded_train)
    report_core = {
        "schema_version": SCHEMA,
        "status": "PASS",
        "source": {"git_sha": source_sha, "config_path": config_path.relative_to(root).as_posix(),
                   "config_sha256": sha256f(config_path)},
        "truth_boundary": {"authority": cfg["authority"], "foreign_pretrained_weights": False, "sft": False,
                           "rlhf": False, "dpo": False, "paid_compute": False, "intelligence_claim": False,
                           "production_readiness_claim": False, "alignment_claim": False,
                           "instruction_following_claim": False, "broad_representativeness_claim": False},
        "previous_corpus": {
            "latest_committed_manifest": latest_manifest,
            "latest_committed_manifest_is_real_source_control": False,
            "reason": "Exact parent still commits the tiny S0 fixture; failed M100 runtime output is not promoted. Control is freshly reconstructed from exact accepted M100 pins plus immutable parent project code.",
            "pins_path": pins_path.relative_to(root).as_posix(), "pins_sha256": sha256f(pins_path),
            "pins": previous_pins, "reconstructed_source_set_metrics": diversity(incumbent_all),
        },
        "data24": {"external": ext_d03, "project_code": project_d03,
                   "all_sources_passed_purpose_specific_model_training_resolution": True},
        "admission": {
            "configured_new_families": sorted(new_families), "actual_new_families": sorted({r["family"] for r in new}),
            "all_external_raw_acquired_bytes_by_source": dict(sorted(raw_all.items())),
            "new_raw_acquired_bytes_by_source": dict(sorted(raw_new.items())),
            "new_admitted_training_plus_heldout_bytes_by_family": dict(sorted(admitted_new.items())),
            "max_admitted_utf8_bytes_per_external_family": cap, "objects": objects,
            "quality_privacy_rejections": rejections + project_rej, "bounded_mass_drops": bound_drops,
            "cross_family_dedup_removed": d0+d1+d2, "near_dedup_threshold": float(cfg["dedup"]["threshold"]),
        },
        "diversity": {
            "ab_control": control_div, "ab_expanded": expanded_div,
            "delta": {"families": expanded_div["families"]-control_div["families"],
                      "top_family_share": expanded_div["top_family_share"]-control_div["top_family_share"],
                      "effective_source_count": expanded_div["effective_source_count"]-control_div["effective_source_count"],
                      "entropy_nats": expanded_div["entropy_nats"]-control_div["entropy_nats"]},
        },
        "common_heldout": {"chunks": len(common_held), "families": sorted({r["family"] for r in common_held}),
                           "identity_sha256": canon_hash([{"id": r["id"], "family": r["family"], "sha256": r["text_sha256"]}
                                                        for r in sorted(common_held, key=lambda x: x["id"])])},
        "matched_500k_ab": {
            "control": control, "expanded": expanded,
            "aggregate_final_delta_bpb": expanded["final_evaluation"]["aggregate_bpb"]-control["final_evaluation"]["aggregate_bpb"],
            "bootstrap": universal_bootstrap(control["final_evaluation"], expanded["final_evaluation"],
                                             int(cfg["bootstrap"]["replicates"]), int(cfg["bootstrap"]["seed"])),
        },
        "environment": {"torch_version": torch.__version__, "python": __import__("sys").version,
                        "pyarrow_version": __import__("pyarrow").__version__},
    }
    report = {**report_core, "report_sha256": canon_hash(report_core)}
    (out/"data184-report.json").write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return report


def validate(path: Path, expected_sha: str | None = None) -> dict[str, Any]:
    r = load_json(path)
    if r.get("schema_version") != SCHEMA or r.get("status") != "PASS":
        raise RuntimeError("bad DATA-184 report schema/status")
    if expected_sha and r["source"]["git_sha"] != expected_sha:
        raise RuntimeError("source SHA mismatch")
    core = dict(r); claimed = core.pop("report_sha256", None)
    if claimed != canon_hash(core):
        raise RuntimeError("report self-hash mismatch")
    a = r["matched_500k_ab"]
    if a["control"]["run"]["optimized_tokens"] != a["expanded"]["run"]["optimized_tokens"]:
        raise RuntimeError("A/B token mismatch")
    if not a["control"]["final_evaluation"]["evaluation_non_mutation"] or not a["expanded"]["final_evaluation"]["evaluation_non_mutation"]:
        raise RuntimeError("evaluation non-mutation missing")
    if set(r["admission"]["configured_new_families"]) != set(r["admission"]["actual_new_families"]):
        raise RuntimeError("not all new families admitted")
    return r


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    rr = sub.add_parser("run")
    rr.add_argument("--repo-root", type=Path, default=Path("."))
    rr.add_argument("--output-dir", type=Path, required=True)
    rr.add_argument("--source-sha", required=True)
    rr.add_argument("--config", type=Path, default=Path("configs/data/data184_real_source_diversity_v1.json"))
    vv = sub.add_parser("validate")
    vv.add_argument("report", type=Path)
    vv.add_argument("--expected-source-sha")
    args = p.parse_args(argv)
    if args.cmd == "run":
        root = args.repo_root.resolve()
        cfg = args.config if args.config.is_absolute() else root/args.config
        out = args.output_dir if args.output_dir.is_absolute() else root/args.output_dir
        r = run(root, out, args.source_sha, cfg)
        print(json.dumps({"status": r["status"], "report_sha256": r["report_sha256"],
                          "new_families": r["admission"]["actual_new_families"],
                          "delta_bpb": r["matched_500k_ab"]["aggregate_final_delta_bpb"]}, indent=2))
    else:
        r = validate(args.report, args.expected_source_sha)
        print(json.dumps({"status": r["status"], "report_sha256": r["report_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
