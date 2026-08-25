#!/usr/bin/env python3
"""DATA-106 convergence wrapper over the live TOK-115 experiment.

This file deliberately does not own model, tokenizer, packing, Trainer,
observability, checkpoint, evaluation, or generation implementations. It imports
TOK-115's incumbent orchestration and narrows the matched experiment to the
DATA-106 control contract:

* fixed source-document slice;
* exactly 60,000 optimized valid targets in every A/B/C matched run;
* deterministic target thinning spread across the complete packed stream, so
  every packed batch/source slice is still executed;
* explicit unmarked/EOS cross-document transition loss;
* an independent replay of the incumbent strict-isolation control;
* fail-closed split-boundary and migration evidence.

The substantial selected ~1M run remains TOK-115's normal larger run and is not
artificially capped at the matched-comparison budget.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import torch

import run_tok115_eos_real_corpus as tok

MATCHED_OPTIMIZED_VALID_TOKENS = 60_000
SCHEMA = "12-6.data106-tok115-convergence.v1"
BRANCH = "data106/tok115-fixed-budget-convergence-20260826"
TOK115_PARENT = "a9e596a3cd0e49e227737b99aadd203fb870d58f"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _valid_mask(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return batch["loss_mask"].bool() & batch["target_ids"].ne(-100)


def _fair_plan(plan: tok.PreparedPlan) -> tok.PreparedPlan:
    raw_valid = int(plan.stats["valid_targets"])
    raw_content = int(plan.stats["content_targets"])
    raw_eos = int(plan.stats["eos_targets"])
    target = MATCHED_OPTIMIZED_VALID_TOKENS
    if raw_valid < target:
        raise RuntimeError(
            f"DATA-106 fixed budget exceeds raw valid targets: {raw_valid} < {target}"
        )

    # Even deterministic thinning keeps the optimization targets distributed
    # across the complete packed stream rather than simply truncating its tail.
    keep_ordinals = {(index * raw_valid) // target for index in range(target)}
    if len(keep_ordinals) != target:
        raise RuntimeError("deterministic thinning produced duplicate keep ordinals")

    new_batches: list[dict[str, torch.Tensor]] = []
    ordinal = 0
    kept_valid = 0
    kept_content = 0
    kept_eos = 0
    for batch in plan.batches:
        copied = {key: value.clone() for key, value in batch.items()}
        mask = _valid_mask(copied)
        positions = torch.nonzero(mask, as_tuple=False)
        for row, column in positions.tolist():
            token = int(copied["target_ids"][row, column].item())
            if ordinal in keep_ordinals:
                kept_valid += 1
                if 0 <= token <= 255:
                    kept_content += 1
                elif token == tok.EXPERIMENTAL_EOS_ID:
                    kept_eos += 1
                else:
                    raise RuntimeError(f"unexpected optimized target ID {token}")
            else:
                copied["target_ids"][row, column] = -100
                copied["loss_mask"][row, column] = False
            ordinal += 1
        if int(_valid_mask(copied).sum().item()) <= 0:
            raise RuntimeError(
                "fairness thinning would create an empty-loss batch; fail closed"
            )
        new_batches.append(copied)

    if ordinal != raw_valid or kept_valid != target:
        raise RuntimeError(
            f"fairness accounting drift: raw={ordinal}/{raw_valid}, kept={kept_valid}/{target}"
        )
    if kept_content + kept_eos != kept_valid:
        raise RuntimeError("optimized target composition does not sum to fixed budget")

    stats = dict(plan.stats)
    source_bytes = int(stats["source_slice"]["source_bytes"])
    allocated = int(stats["allocated_positions"])
    stats.update(
        {
            "raw_valid_targets_before_fairness_mask": raw_valid,
            "raw_content_targets_before_fairness_mask": raw_content,
            "raw_eos_targets_before_fairness_mask": raw_eos,
            "raw_packing_utilization_valid_targets": raw_valid / allocated,
            "raw_padding_waste_fraction": (allocated - raw_valid) / allocated,
            "optimized_valid_target_budget": target,
            "optimized_valid_target_budget_exact": True,
            "optimized_content_targets": kept_content,
            "optimized_eos_targets": kept_eos,
            "optimized_target_density": target / allocated,
            "optimization_masked_raw_valid_targets": raw_valid - target,
            "optimization_mask_policy": (
                "deterministic_even_thinning_over_complete_packed_stream_v1"
            ),
            "all_packed_batches_executed": True,
            "all_source_documents_fixed_across_candidates": True,
            "optimized_valid_tokens_per_source_byte": target / source_bytes,
            "raw_valid_targets_per_source_byte": raw_valid / source_bytes,
            "raw_content_targets_per_source_byte": raw_content / source_bytes,
        }
    )
    # These fields describe targets that actually enter Trainer after masking.
    stats["valid_targets"] = kept_valid
    stats["content_targets"] = kept_content
    stats["eos_targets"] = kept_eos
    stats["eos_overhead_vs_content"] = kept_eos / kept_content if kept_content else None
    return tok.PreparedPlan(batches=new_batches, stats=stats, records=list(plan.records))


def _cross_document_transition_bits(
    model: tok.TwelveSixDecoder,
    candidate: str,
    records: Sequence[tok.CorpusRow],
) -> dict[str, Any]:
    tokenizer = tok.tokenizer_for(candidate)
    docs = [row for row in records if row.text.encode("utf-8")][: tok.BOUNDARY_DOCS]
    if len(docs) < 2:
        raise RuntimeError("DATA-106 transition panel needs at least two documents")
    values: list[float] = []
    for previous, following in zip(docs[:-1], docs[1:]):
        previous_ids = tokenizer.encode(previous.text)
        following_ids = tokenizer.encode(following.text)
        if candidate == "A":
            context = previous_ids[-64:]
            context_kind = "unmarked_previous_document_tail"
        else:
            if tokenizer.eos_id != tok.EXPERIMENTAL_EOS_ID:
                raise RuntimeError("EOS candidate lost TOK-39 EOS identity")
            context = previous_ids[-63:] + [tok.EXPERIMENTAL_EOS_ID]
            context_kind = "previous_document_tail_plus_eos"
        log_probs = tok._next_log_probs(model, context)
        target = following_ids[0]
        values.append(float(-log_probs[target].item() / math.log(2.0)))
    return {
        "mean_bits": mean(values),
        "pairs": len(values),
        "context": context_kind,
        "target": "next_document_first_byte",
        "attention_reset_at_boundary": False,
    }


def _install_matched_patches() -> dict[str, Any]:
    original_prepare = tok.prepare_plan
    original_boundary = tok.boundary_metrics
    original_packing_identity = tok.packing_identity
    original_run = tok._run_one_matched

    def fair_prepare(candidate: str, records: Sequence[tok.CorpusRow]) -> tok.PreparedPlan:
        return _fair_plan(original_prepare(candidate, records))

    def fair_boundary(
        model: tok.TwelveSixDecoder,
        candidate: str,
        records: Sequence[tok.CorpusRow],
    ) -> dict[str, Any]:
        transition = _cross_document_transition_bits(model, candidate, records)
        if candidate == "A":
            return {
                "status": "MEASURED_CONTROL_NO_EOS",
                "documents": min(len(records), tok.BOUNDARY_DOCS),
                "document_start_prediction": None,
                "document_end_prediction": None,
                "cross_document_attention_influence": None,
                "cross_document_transition_loss": transition,
                "eos_is_attention_reset": False,
            }
        result = dict(original_boundary(model, candidate, records))
        result["cross_document_transition_loss"] = transition
        result["eos_is_attention_reset"] = False
        return result

    def fair_packing_identity(
        candidate: str, plan: tok.PreparedPlan
    ) -> dict[str, Any]:
        base = original_packing_identity(candidate, plan)
        payload = dict(base["payload"])
        payload.update(
            {
                "optimized_valid_target_budget": plan.stats[
                    "optimized_valid_target_budget"
                ],
                "optimization_mask_policy": plan.stats["optimization_mask_policy"],
            }
        )
        return {"payload": payload, "sha256": tok.hash_json(payload)}

    def fair_run(*args: Any, **kwargs: Any):
        result, model, trainer, plan, run_id = original_run(*args, **kwargs)
        if trainer.tokens_seen != MATCHED_OPTIMIZED_VALID_TOKENS:
            raise RuntimeError(
                f"matched Trainer token budget drift: {trainer.tokens_seen}"
            )
        telemetry = result["training"]["observer"]
        optimization = telemetry["optimization"]
        result["optimized_valid_token_budget_exact"] = trainer.tokens_seen
        result["final_model_state_sha256"] = tok._tensor_state_sha256(model)
        result["gradient_behavior"] = {
            "gradient_norm_min": optimization["gradient_norm_min"],
            "gradient_norm_max": optimization["gradient_norm_max"],
            "all_reported_gradient_norms_finite": all(
                value is None or math.isfinite(float(value))
                for value in (
                    optimization["gradient_norm_min"],
                    optimization["gradient_norm_max"],
                )
            ),
        }
        return result, model, trainer, plan, run_id

    tok.prepare_plan = fair_prepare
    tok.boundary_metrics = fair_boundary
    tok.packing_identity = fair_packing_identity
    tok._run_one_matched = fair_run
    return {
        "original_prepare": original_prepare,
        "original_boundary": original_boundary,
        "original_packing_identity": original_packing_identity,
        "original_run": original_run,
    }


def _matched_context():
    built_manifest, rows = tok.ensure_corpus()
    ordered_train = tok._weighted_order(rows, "train")
    ordered_val = tok._weighted_order(rows, "validation")
    train_records = tok.select_source_slice(ordered_train, tok.MATCHED_SOURCE_BYTES)
    train_panel = tok.select_source_slice(
        ordered_train, min(tok.HELDOUT_PANEL_BYTES, tok.MATCHED_SOURCE_BYTES)
    )
    val_panel = tok.select_source_slice(ordered_val, tok.HELDOUT_PANEL_BYTES)
    return built_manifest, train_records, train_panel, val_panel


def _split_proof(
    train_records: Sequence[tok.CorpusRow], val_panel: Sequence[tok.CorpusRow]
) -> dict[str, Any]:
    train_ids = {row.record_id for row in train_records}
    val_ids = {row.record_id for row in val_panel}
    train_hashes = {row.content_sha256 for row in train_records}
    val_hashes = {row.content_sha256 for row in val_panel}
    record_overlap = sorted(train_ids & val_ids)
    content_overlap = sorted(train_hashes & val_hashes)
    if record_overlap or content_overlap:
        raise RuntimeError("held-out records/content leaked into training source slice")
    if any(row.split != "train" for row in train_records):
        raise RuntimeError("non-train record entered matched training source slice")
    if any(row.split != "validation" for row in val_panel):
        raise RuntimeError("non-validation record entered held-out panel")
    return {
        "training_record_validation_record_overlap": record_overlap,
        "training_content_validation_content_overlap": content_overlap,
        "evaluation_records_passed_to_training_packer": False,
        "cross_document_packing_input_split": "train_only",
        "reserved_split_boundary_continuation_possible": False,
        "proof": (
            "packer is invoked on the train-only fixed source slice; validation is "
            "materialized separately and used only by evaluation"
        ),
    }


def phase_matched(out: Path, source_sha: str) -> None:
    _install_matched_patches()
    built, train_records, train_panel, val_panel = _matched_context()
    split_proof = _split_proof(train_records, val_panel)

    tok.phase_matched(out, source_sha)
    matched_path = out / "matched.json"
    matched = json.loads(matched_path.read_text(encoding="utf-8"))

    results = matched["results"]
    for result in results:
        optimized = int(result["training"]["observer"]["counters"]["optimized_tokens"])
        seen = int(result["training"]["tokens_seen_including_eos"])
        if optimized != MATCHED_OPTIMIZED_VALID_TOKENS or seen != optimized:
            raise RuntimeError(
                f"{result['scale']}/{result['candidate']} token budget mismatch: "
                f"observer={optimized}, trainer={seen}"
            )
        if not result["gradient_behavior"]["all_reported_gradient_norms_finite"]:
            raise RuntimeError("non-finite gradient evidence in matched run")

    original_control = next(
        result
        for result in results
        if result["scale"] == "1m" and result["candidate"] == "A"
    )
    replay_result, replay_model, _, _, _ = tok._run_one_matched(
        out,
        source_sha,
        "1m",
        "A",
        train_records,
        train_panel,
        val_panel,
    )
    replay_hash = tok._tensor_state_sha256(replay_model)
    replay_equal = (
        replay_hash == original_control["final_model_state_sha256"]
        and replay_result["heldout"]["final_bpb"]
        == original_control["heldout"]["final_bpb"]
        and replay_result["train_panel"]["final_bpb"]
        == original_control["train_panel"]["final_bpb"]
        and replay_result["generation_after"] == original_control["generation_after"]
    )
    if not replay_equal:
        raise RuntimeError("incumbent strict-isolation control replay diverged")

    matched["data106"] = {
        "schema": SCHEMA,
        "source_sha": source_sha,
        "tok115_parent_sha": TOK115_PARENT,
        "fixed_optimized_valid_token_budget": MATCHED_OPTIMIZED_VALID_TOKENS,
        "fixed_source_document_slice": True,
        "fixed_model_geometry_per_scale": True,
        "fixed_optimizer_seed_and_precision": True,
        "split_proof": split_proof,
        "corpus_identity_sha256": built["corpus_identity_sha256"],
        "arm_mapping": {
            "A_strict_document_isolation": "TOK115 A",
            "B_eos_packed_normal_causal_attention": "TOK115 C",
            "C_current_incumbent_control": "independent TOK115 A replay",
            "extra_diagnostic_eos_document_isolated": "TOK115 B",
        },
        "incumbent_control_replay": {
            "status": "PASS" if replay_equal else "FAIL",
            "original_final_model_state_sha256": original_control[
                "final_model_state_sha256"
            ],
            "replay_final_model_state_sha256": replay_hash,
            "heldout_final_bpb_exact": replay_result["heldout"]["final_bpb"],
            "generation_after_exact_match": (
                replay_result["generation_after"]
                == original_control["generation_after"]
            ),
        },
        "eos_semantics": {
            "eos_id": tok.EXPERIMENTAL_EOS_ID,
            "normal_causal_attention_through_eos": True,
            "attention_reset_at_eos": False,
        },
    }
    _write_json(matched_path, matched)
    print(
        json.dumps(
            {
                "data106_fixed_budget": MATCHED_OPTIMIZED_VALID_TOKENS,
                "control_replay": "PASS",
                "split_proof": "PASS",
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def _data106_recommendation(matched: Mapping[str, Any]) -> dict[str, Any]:
    one_m = {
        result["candidate"]: result
        for result in matched["results"]
        if result["scale"] == "1m"
    }
    strict = one_m["A"]
    packed = one_m["C"]
    eligible = []
    for name, result in (("strict_isolation", strict), ("eos_packed", packed)):
        if (
            result["heldout"]["decrease"] > 0
            and result["train_panel"]["decrease"] > 0
            and result["gradient_behavior"]["all_reported_gradient_norms_finite"]
        ):
            eligible.append((name, result))
    if not eligible:
        return {
            "selected_for_next_campaign": "strict_isolation",
            "decision_rule": "FAIL_CLOSED_NO_ELIGIBLE_CANDIDATE",
        }
    winner_name, winner = min(
        eligible, key=lambda pair: pair[1]["heldout"]["final_bpb"]
    )
    return {
        "selected_for_next_campaign": winner_name,
        "decision_rule": (
            "LOWEST_COMMON_HELDOUT_CONTENT_BPB_AMONG_FINITE_FIXED_BUDGET_ARMS"
        ),
        "strict_final_bpb": strict["heldout"]["final_bpb"],
        "eos_packed_final_bpb": packed["heldout"]["final_bpb"],
        "eos_packed_minus_strict_bpb": (
            packed["heldout"]["final_bpb"] - strict["heldout"]["final_bpb"]
        ),
        "single_seed_boundary": True,
        "eos_is_attention_reset": False,
        "selected_parent_candidate": winner["candidate"],
    }


def _augment_final(out: Path, source_sha: str) -> None:
    final_path = out / "final-report.json"
    final = json.loads(final_path.read_text(encoding="utf-8"))
    matched = final["matched_experiment"]
    data106 = matched.get("data106")
    if not isinstance(data106, Mapping):
        raise RuntimeError("final report lost fixed-budget DATA-106 matched evidence")

    pre_opt = final["training"]["pre_resume_observer"]["optimization"]
    post_opt = final["training"]["post_resume_observer"]["optimization"]
    train_loss_first = pre_opt["loss_first"]
    train_loss_final = post_opt["loss_final"]
    if train_loss_first is None or train_loss_final is None:
        raise RuntimeError("substantial run lacks actual Trainer loss evidence")
    train_loss_decreased = float(train_loss_final) < float(train_loss_first)
    if not train_loss_decreased:
        raise RuntimeError(
            f"substantial Trainer loss did not decrease: {train_loss_first} -> {train_loss_final}"
        )

    recommendation = _data106_recommendation(matched)
    selected_vocab = int(final["tokenizer"]["vocab_size"])
    parameter_tax = matched["parameter_tax"]["1m"]
    final.setdefault("worker_ids", []).append("DATA-106-DOCUMENT-BOUNDARIES")
    final["exact_experimental_branch"] = BRANCH
    final["lineage"]["tok115_parent"] = TOK115_PARENT
    final["data106"] = {
        **dict(data106),
        "recommendation": recommendation,
        "parameter_comparison": parameter_tax,
        "migration_and_checkpoint_compatibility": {
            "vocab256_to_vocab257_in_place_resume": "FORBIDDEN_FAIL_CLOSED",
            "vocab257_to_vocab256_in_place_resume": "FORBIDDEN_FAIL_CLOSED",
            "reason": (
                "tokenizer config/vocab hashes, ModelSpec vocabulary and tied "
                "embedding shape differ; D05 compatibility must reject before mutation"
            ),
            "same_tokenizer_same_modelspec_resume": "D05_VERIFIED_ONLY",
            "eos_attention_reset": False,
            "chat_instruction_tokens": False,
        },
    }
    final["learning_proof"]["actual_trainer_loss_first"] = train_loss_first
    final["learning_proof"]["actual_trainer_loss_final"] = train_loss_final
    final["learning_proof"]["actual_trainer_loss_decreased"] = train_loss_decreased
    final["milestone100_status"] = (
        "PARTIAL_FAIL_CLOSED_REAL_WORLD_REPRESENTATIVE_CORPUS_NOT_MET"
    )
    final["milestone100_required_proof"] = {
        "random_initialization": final["model"]["random_initialization"],
        "exact_parameter_count": final["model"]["parameter_count"],
        "versioned_tokenizer": final["tokenizer"]["version"],
        "train_loss_decreased": train_loss_decreased,
        "heldout_bpb_decreased": final["learning_proof"]["heldout_bpb_decrease"] > 0,
        "multiple_checkpoints": all(
            name in final["checkpoints"] for name in ("quarter", "midpoint", "final")
        ),
        "fresh_process_resume": final["fresh_process_resume"]["different_process"],
        "evaluation_non_mutation": final["learning_proof"]["evaluation_non_mutating"],
        "generation_before_after": bool(final["generation_before"])
        and bool(final["generation_after"]),
        "retained_exact_checkpoint": final["checkpoints"]["final"]["checkpoint_id"],
        "machine_manifest": bool(final["machine_manifest"]),
        "real_world_representative_corpus": False,
        "real_world_representative_corpus_reason": (
            "DATA-25 is substantial deterministic project-authored UK/EN/code data "
            "with zero external training-eligible sources at this identity"
        ),
    }
    final["reproduction_command"] = (
        "PYTHONPATH=src python tools/run_data106_tok115_convergence.py --phase matched "
        f"--output-dir {out} --source-sha {source_sha} && "
        "PYTHONPATH=src python tools/run_data106_tok115_convergence.py --phase start "
        f"--output-dir {out} --source-sha {source_sha} && "
        "PYTHONPATH=src python tools/run_data106_tok115_convergence.py --phase resume "
        f"--output-dir {out} --source-sha {source_sha}"
    )
    if selected_vocab not in {256, 257}:
        raise RuntimeError("unexpected selected tokenizer vocabulary")
    _write_json(final_path, final)


def parse_args():
    parser = tok.argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("matched", "start", "resume"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--torch-threads", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.torch_threads <= 0:
        raise ValueError("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    source_sha = args.source_sha.strip()
    actual_sha = tok._git_sha()
    if source_sha != actual_sha:
        raise RuntimeError(f"source SHA mismatch: {source_sha} != {actual_sha}")
    if len(source_sha) != 40:
        raise RuntimeError("source SHA must be exact 40-hex Git identity")
    if args.phase == "matched":
        phase_matched(args.output_dir, source_sha)
    elif args.phase == "start":
        tok.phase_start(args.output_dir, source_sha)
    else:
        tok.phase_resume(args.output_dir, source_sha)
        _augment_final(args.output_dir, source_sha)


if __name__ == "__main__":
    main()
