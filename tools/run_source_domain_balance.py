#!/usr/bin/env python3
"""Execute DATA-105 source/domain concentration analysis and balancing experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from twelve_six.data.corpus_v01 import build_corpus
from twelve_six.data.source_balance import (
    STRATA,
    SourcePolicy,
    analyze_records,
    build_source_balance_plan,
    policy_is_effective,
    sha256_json,
)
from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.packing.scale_contracts import MixturePlan, MixtureSource
from twelve_six.training.config import TrainerConfig
from twelve_six.training.trainer import Trainer

REPORT_SCHEMA = "12-6.data105-source-domain-balance-report.v1"
DEFAULT_POLICIES = (
    SourcePolicy("raw_proportional"),
    SourcePolicy("bounded_source_cap", cap_basis_points=3500),
    SourcePolicy("tempered_source_sqrt", temper_exponent="1/2"),
)
MODEL_SPEC = ModelSpec(
    schema_version=1,
    vocab_size=256,
    max_seq_len=256,
    d_model=72,
    n_layers=4,
    n_heads=6,
    n_kv_heads=6,
    head_dim=12,
    d_ff=192,
    rope_rotary_dim=12,
)
INIT_SPEC = InitSpec()
MODEL_SEED = 1337
BALANCE_SEED = 105
BATCH_SIZE = 4
SEQUENCE_TARGETS = 64
OPTIMIZER_STEPS = 96
MINORITY_SHARE_THRESHOLD = 0.25
MATERIAL_DOMAIN_REGRESSION_BPB = 0.03


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def _load_rows(build_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for shard in manifest["shards"]:
        rows.extend(_load_jsonl(build_dir / shard["path"]))
    return rows


def _digest_index(key: str, upper_bound: int) -> int:
    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    value = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest(), "big")
    return value % upper_bound


def _stratum_manifest(corpus_identity: str, stratum: str) -> str:
    return sha256_json({"corpus_identity_sha256": corpus_identity, "stratum": stratum})


def _mixture_plan(corpus_identity: str) -> MixturePlan:
    weights = {"uk": 45, "en": 35, "code": 20}
    return MixturePlan(
        plan_id="data105-incumbent-uk-en-code-45-35-20",
        tokenizer_config_sha256=hashlib.sha256(b"canonical-byte-tokenizer-config").hexdigest(),
        tokenizer_vocab_sha256=hashlib.sha256(bytes(range(256))).hexdigest(),
        packing_config_sha256=hashlib.sha256(b"data105-fixed-seq64-batch4").hexdigest(),
        sources=tuple(
            MixtureSource(
                name=stratum,
                manifest_sha256=_stratum_manifest(corpus_identity, stratum),
                weight_units=weights[stratum],
            )
            for stratum in STRATA
        ),
        seed=BALANCE_SEED,
        num_shards=1,
    )


def _records_by_source(
    rows: list[dict[str, Any]], split: str
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            grouped[row["source_id"]].append(row)
    for source in grouped:
        grouped[source].sort(key=lambda row: row["record_id"])
    return dict(sorted(grouped.items()))


def _training_batches(rows: list[dict[str, Any]], mixture: MixturePlan, balance_plan):
    sources = _records_by_source(rows, "train")
    stratum_draws = Counter()
    sample_index = 0
    trace = hashlib.sha256()
    for _step in range(OPTIMIZER_STEPS):
        sequences = []
        for _row in range(BATCH_SIZE):
            stratum = mixture.source_for_sample(sample_index)
            draw = stratum_draws[stratum]
            source_id = balance_plan.source_for_draw(stratum, draw)
            choices = sources[source_id]
            record_index = _digest_index(
                (
                    f"{balance_plan.corpus_identity_sha256}:"
                    f"{balance_plan.top_level_mixture_sha256}:"
                    f"{BALANCE_SEED}:{stratum}:{source_id}:{draw}:record"
                ),
                len(choices),
            )
            record = choices[record_index]
            payload = record["text"].encode("utf-8")
            needed = SEQUENCE_TARGETS + 1
            if len(payload) < needed:
                raise RuntimeError("admitted record is unexpectedly shorter than training window")
            max_offset = len(payload) - needed + 1
            offset = _digest_index(
                (
                    f"{balance_plan.corpus_identity_sha256}:"
                    f"{balance_plan.top_level_mixture_sha256}:"
                    f"{BALANCE_SEED}:{record['record_id']}:{draw}:offset"
                ),
                max_offset,
            )
            sequence = payload[offset : offset + needed]
            trace.update(
                (
                    f"{sample_index}\0{stratum}\0{source_id}\0"
                    f"{record['record_id']}\0{offset}\n"
                ).encode("utf-8")
            )
            sequences.append(torch.tensor(list(sequence), dtype=torch.long))
            sample_index += 1
            stratum_draws[stratum] += 1
        yield {"input_ids": torch.stack(sequences)}, trace


def _eval_domain_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    analysis = analyze_records(row for row in rows if row["split"] == "validation")
    family_for_record = {
        row["record_id"]: row["source_family"]
        for row in analysis["taxonomy"]["records"]
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "validation":
            continue
        grouped[family_for_record[row["record_id"]]].append(row)
    for family in grouped:
        grouped[family].sort(key=lambda row: row["record_id"])
    return dict(sorted(grouped.items()))


@torch.no_grad()
def _evaluate(model: TwelveSixDecoder, rows: list[dict[str, Any]]) -> dict[str, Any]:
    model.eval()
    domains = _eval_domain_rows(rows)
    by_domain: dict[str, Any] = {}
    total_nats = 0.0
    total_targets = 0
    for domain, domain_rows in domains.items():
        domain_nats = 0.0
        domain_targets = 0
        for row in domain_rows[:64]:
            payload = row["text"].encode("utf-8")[:129]
            if len(payload) < 2:
                continue
            ids = torch.tensor([list(payload)], dtype=torch.long)
            logits = model(ids).logits
            targets = ids[:, 1:]
            nats = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            target_count = targets.numel()
            domain_nats += float(nats.item())
            domain_targets += target_count
        if domain_targets <= 0:
            raise RuntimeError(f"held-out domain {domain!r} has no evaluable byte targets")
        by_domain[domain] = {
            "bpb": domain_nats / (math.log(2.0) * domain_targets),
            "byte_targets": domain_targets,
        }
        total_nats += domain_nats
        total_targets += domain_targets
    return {
        "aggregate_bpb": total_nats / (math.log(2.0) * total_targets),
        "byte_targets": total_targets,
        "by_domain": by_domain,
    }


def _train_policy(rows: list[dict[str, Any]], mixture: MixturePlan, balance_plan):
    torch.manual_seed(MODEL_SEED)
    model = TwelveSixDecoder(MODEL_SPEC, INIT_SPEC)
    initial = _evaluate(model, rows)
    config = TrainerConfig(
        learning_rate=3e-4,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
        max_steps=OPTIMIZER_STEPS,
        warmup_steps=0,
        scheduler="constant",
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        precision="fp32",
        seed=MODEL_SEED,
        deterministic_algorithms=True,
        deterministic_warn_only=False,
    )
    trainer = Trainer(model, config, device="cpu")
    losses = []
    trace_digest = None
    for batch, trace in _training_batches(rows, mixture, balance_plan):
        metrics = trainer.train_microbatch(batch)
        losses.append(metrics.update_loss)
        trace_digest = trace.hexdigest()
    if trainer.optimizer_step != OPTIMIZER_STEPS:
        raise RuntimeError("training did not reach the fixed optimizer-step budget")
    expected_tokens = OPTIMIZER_STEPS * BATCH_SIZE * SEQUENCE_TARGETS
    if trainer.tokens_seen != expected_tokens:
        raise RuntimeError("optimized-token budget mismatch")
    final = _evaluate(model, rows)
    return {
        "plan_sha256": balance_plan.sha256,
        "policy": balance_plan.policy.to_dict(),
        "sampling_with_replacement": True,
        "materialized_duplicate_documents": False,
        "trace_sha256": trace_digest,
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "initial_eval": initial,
        "final_eval": final,
        "first_update_loss": losses[0],
        "final_update_loss": losses[-1],
    }


def _domain_train_shares(analysis: dict[str, Any]) -> dict[str, float]:
    mass = analysis["mass"]["source_family_domain"]
    total = sum(item["byte_tokens"] for item in mass.values())
    return {key: item["byte_tokens"] / total for key, item in mass.items()}


def _select_recommendation(
    analysis: dict[str, Any], runs: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    origins = analysis["mass"]["origin"]
    real_tokens = origins.get("real_external", {}).get("byte_tokens", 0)
    per_stratum = analysis["concentration"]["per_stratum"]
    multi_source = all(per_stratum[name]["sources"]["members"] >= 2 for name in STRATA)
    raw = runs["raw_proportional"]
    shares = _domain_train_shares(analysis)
    candidates = []
    for name, run in runs.items():
        regressions = {}
        for domain, baseline in raw["final_eval"]["by_domain"].items():
            delta = run["final_eval"]["by_domain"][domain]["bpb"] - baseline["bpb"]
            regressions[domain] = delta
        minority_regression = max(
            (
                delta
                for domain, delta in regressions.items()
                if shares.get(domain, 0.0) < MINORITY_SHARE_THRESHOLD
            ),
            default=0.0,
        )
        candidates.append(
            {
                "policy": name,
                "aggregate_delta_bpb_vs_raw": (
                    run["final_eval"]["aggregate_bpb"]
                    - raw["final_eval"]["aggregate_bpb"]
                ),
                "per_domain_delta_bpb_vs_raw": regressions,
                "max_minority_domain_regression_bpb": minority_regression,
                "passes_minority_guard": (
                    minority_regression <= MATERIAL_DOMAIN_REGRESSION_BPB
                ),
            }
        )
    if real_tokens <= 0 or not multi_source:
        selected = "raw_proportional"
        verdict = "RETAIN_RAW_BLOCK_PROMOTION_NO_REAL_MULTI_SOURCE_CORPUS"
        authority = "MECHANICS_ONLY"
    else:
        safe = [item for item in candidates if item["passes_minority_guard"]]
        best = min(safe, key=lambda item: item["aggregate_delta_bpb_vs_raw"])
        selected = best["policy"]
        verdict = "EMPIRICAL_POLICY_SELECTED"
        authority = "REAL_CORPUS_LOCAL_FREE_EXPERIMENT"
    return {
        "verdict": verdict,
        "selected_policy": selected,
        "authority": authority,
        "real_external_train_byte_tokens": real_tokens,
        "all_strata_have_at_least_two_sources": multi_source,
        "minority_domain_share_threshold": MINORITY_SHARE_THRESHOLD,
        "material_regression_bpb": MATERIAL_DOMAIN_REGRESSION_BPB,
        "candidates": candidates,
    }


def run(config_path: Path, output_path: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="data105-corpus-") as temp:
        build_dir = Path(temp) / "corpus"
        manifest = build_corpus(config_path, build_dir)
        rows = _load_rows(build_dir, manifest)
        train_rows = [row for row in rows if row["split"] == "train"]
        analysis = analyze_records(train_rows)
        mixture = _mixture_plan(manifest["corpus_identity_sha256"])
        raw_policy = DEFAULT_POLICIES[0]
        raw_plan = build_source_balance_plan(
            corpus_identity_sha256=manifest["corpus_identity_sha256"],
            top_level_mixture_sha256=mixture.sha256,
            analysis=analysis,
            policy=raw_policy,
            seed=BALANCE_SEED,
        )
        plans = {}
        for policy in DEFAULT_POLICIES:
            plan = build_source_balance_plan(
                corpus_identity_sha256=manifest["corpus_identity_sha256"],
                top_level_mixture_sha256=mixture.sha256,
                analysis=analysis,
                policy=policy,
                seed=BALANCE_SEED,
            )
            plans[policy.name] = plan

        runs = {name: _train_policy(rows, mixture, plan) for name, plan in plans.items()}
        report_core = {
            "schema_version": REPORT_SCHEMA,
            "worker_id": "DATA-105-DOMAIN-BALANCE",
            "execution_class": "LOCAL_FREE_CPU",
            "corpus": {
                "identity_sha256": manifest["corpus_identity_sha256"],
                "truth_boundary": manifest["truth_boundary"],
                "train_documents": len(train_rows),
                "train_byte_tokens": sum(row["byte_tokens"] for row in train_rows),
            },
            "top_level_mixture": {
                "incumbent_mixture_plan_sha256": mixture.sha256,
                "weight_units": {"uk": 45, "en": 35, "code": 20},
                "replaced_by_data105": False,
            },
            "analysis": analysis,
            "policy_plans": {
                name: {
                    **plan.to_dict(),
                    "plan_sha256": plan.sha256,
                    "changes_raw_source_weights": policy_is_effective(plan, raw_plan),
                }
                for name, plan in plans.items()
            },
            "experiment": {
                "model_spec": MODEL_SPEC.to_dict(),
                "model_spec_sha256": MODEL_SPEC.identity_sha256(),
                "trainable_parameters": MODEL_SPEC.parameter_count(),
                "tokenizer": "canonical-byte-vocab-256",
                "optimizer": {
                    "name": "AdamW",
                    "learning_rate": 3e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.0,
                    "gradient_clip_norm": 1.0,
                },
                "optimized_tokens_per_policy": (
                    OPTIMIZER_STEPS * BATCH_SIZE * SEQUENCE_TARGETS
                ),
                "runs": runs,
            },
            "recommendation": _select_recommendation(analysis, runs),
            "truth_boundary": {
                "no_foreign_pretrained_weights": True,
                "no_paid_compute": True,
                "sampling_with_replacement_explicit": True,
                "duplicate_documents_materialized": False,
                "real_corpus_policy_claim_allowed": (
                    manifest["truth_boundary"]["contains_external_training_data"]
                    and all(
                        analysis["concentration"]["per_stratum"][name]["sources"]["members"]
                        >= 2
                        for name in STRATA
                    )
                ),
            },
        }
        report = {**report_core, "report_sha256": sha256_json(report_core)}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/corpus_v01.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("reports/data105/source_domain_balance_local_free.json"),
    )
    args = parser.parse_args()
    report = run(args.config, args.out)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
