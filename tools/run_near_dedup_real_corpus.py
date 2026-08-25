from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from twelve_six.data.near_dedup import (
    REPORT_SCHEMA,
    calibration_records,
    canonical_json_bytes,
    load_calibration,
    policy_candidates,
    run_datatrove_policy,
    score_calibration,
    select_policy,
    sha256_bytes,
    surviving_corpus_identity,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data10_records(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 17:
        raise ValueError(f"unexpected DATA-10 mechanics corpus layout: {len(lines)} lines")
    records: list[dict] = []
    for index in range(6):
        lang = "uk" if index < 3 else "en"
        text = lines[index]
        records.append(
            {
                "id": f"data10-{lang}-{index + 1:02d}",
                "text": text,
                "metadata": {
                    "source_id": "data10_project_authored_mechanics",
                    "raw_identity": sha256_bytes(text.encode("utf-8")),
                    "modality": "natural",
                    "language": lang,
                    "authority": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
                },
            }
        )
    code_groups = [
        ("python-function", lines[6:8]),
        ("python-class", lines[8:14]),
        ("sql-query", lines[14:17]),
    ]
    for name, group in code_groups:
        text = "\n".join(group) + "\n"
        records.append(
            {
                "id": f"data10-code-{name}",
                "text": text,
                "metadata": {
                    "source_id": "data10_project_authored_mechanics",
                    "raw_identity": sha256_bytes(text.encode("utf-8")),
                    "modality": "code",
                    "language": "code",
                    "authority": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
                },
            }
        )
    return records


def _fresh(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _determinism_projection(execution: dict) -> dict:
    return {
        "survivor_ids": execution["survivor_ids"],
        "removed_ids": execution["removed_ids"],
        "clusters": execution["clusters"],
        "cluster_statistics": execution["cluster_statistics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="DATA-30 calibrated DataTrove near-dedup execution")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--data10-corpus", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    _fresh(args.workspace)
    calibration = load_calibration(args.calibration)
    candidates = policy_candidates()
    candidate_results: dict[str, list[dict]] = {"natural": [], "code": []}
    scored_for_selection = {"natural": [], "code": []}

    for modality in ("natural", "code"):
        records = calibration_records(calibration, modality)
        for policy in candidates[modality]:
            execution = run_datatrove_policy(
                records,
                policy=policy,
                workspace=args.workspace / "calibration" / modality / policy.name,
                exercise_skip_completed=True,
            )
            metrics = score_calibration(calibration, modality=modality, execution=execution)
            candidate_results[modality].append(
                {
                    "policy": policy.manifest(),
                    "metrics": metrics,
                    "cluster_statistics": execution["cluster_statistics"],
                    "restart": execution["restart"],
                }
            )
            scored_for_selection[modality].append((policy, metrics))

    code_metrics = {policy.name: metrics for policy, metrics in scored_for_selection["code"]}
    code_incumbent = code_metrics["code_incumbent_band_5g_14x8"]
    code_strict = code_metrics["code_strict_5g_10x10"]
    code_strict_required = (
        code_strict["recall"] >= 0.75
        and code_strict["false_removal_risk"] <= 0.25
        and (
            code_incumbent["recall"] < 0.75
            or code_incumbent["false_removal_risk"] > 0.25
            or code_strict["false_removal_risk"] < code_incumbent["false_removal_risk"]
        )
    )
    code_preferred = (
        "code_strict_5g_10x10" if code_strict_required else "code_incumbent_band_5g_14x8"
    )
    selected = {
        "natural": select_policy(
            scored_for_selection["natural"],
            preferred_policy_name="natural_incumbent_9g_14x8",
        ),
        "code": select_policy(
            scored_for_selection["code"],
            preferred_policy_name=code_preferred,
        ),
    }

    selected_validation: dict[str, dict] = {}
    for modality in ("natural", "code"):
        records = calibration_records(calibration, modality)
        first = run_datatrove_policy(
            records,
            policy=selected[modality],
            workspace=args.workspace / "selected-validation" / modality,
            exercise_skip_completed=True,
        )
        rerun = run_datatrove_policy(
            records,
            policy=selected[modality],
            workspace=args.workspace / "selected-validation" / modality,
            exercise_skip_completed=True,
        )
        fresh = run_datatrove_policy(
            records,
            policy=selected[modality],
            workspace=args.workspace / "selected-validation-fresh" / modality,
            exercise_skip_completed=True,
        )
        first_projection = _determinism_projection(first)
        selected_validation[modality] = {
            "full_skip_completed_rerun_identical": first_projection == _determinism_projection(rerun),
            "fresh_workspace_representative_selection_identical": first_projection == _determinism_projection(fresh),
            "signature_skip_completed_verified": first["restart"]["signature_rerun_byte_identical"],
        }

    data10_records = _data10_records(args.data10_corpus)
    natural_records = [record for record in data10_records if record["metadata"]["modality"] == "natural"]
    code_records = [record for record in data10_records if record["metadata"]["modality"] == "code"]
    corpus_exec = {
        "natural": run_datatrove_policy(
            natural_records,
            policy=selected["natural"],
            workspace=args.workspace / "current-corpus" / "natural",
            exercise_skip_completed=True,
        ),
        "code": run_datatrove_policy(
            code_records,
            policy=selected["code"],
            workspace=args.workspace / "current-corpus" / "code",
            exercise_skip_completed=True,
        ),
    }
    survivor_ids = set(corpus_exec["natural"]["survivor_ids"]) | set(corpus_exec["code"]["survivor_ids"])
    survivors = [record for record in data10_records if record["id"] in survivor_ids]
    input_corpus_identity = _sha256_file(args.data10_corpus)
    corpus_identity = surviving_corpus_identity(
        survivors,
        selected_policies=selected,
        input_corpus_identity=input_corpus_identity,
    )

    false_positive_review = []
    for modality in ("natural", "code"):
        selected_name = selected[modality].name
        for result in candidate_results[modality]:
            if result["policy"]["name"] == selected_name:
                false_positive_review.extend(result["metrics"]["false_positive_review_sample"])
                break

    input_records = sum(item["input_records"] for item in corpus_exec.values())
    removed_records = sum(item["removed_records"] for item in corpus_exec.values())
    input_bytes = sum(item["input_bytes"] for item in corpus_exec.values())
    removed_bytes = sum(item["removed_bytes"] for item in corpus_exec.values())
    report = {
        "schema_version": REPORT_SCHEMA,
        "worker_id": "DATA-30-NEAR-DEDUP",
        "execution_class": "LOCAL_FREE",
        "paid_cost_usd": 0,
        "engine": {
            "name": "DataTrove MinHash",
            "version": corpus_exec["natural"]["engine"]["version"],
            "wheel_sha256": corpus_exec["natural"]["engine"]["wheel_sha256"],
            "second_dedup_engine_created": False,
        },
        "calibration": {
            "calibration_sha256": _sha256_file(args.calibration),
            "categories_required": [
                "true_near_copy",
                "boilerplate",
                "translation",
                "code_fork",
                "legitimate_similar_document",
            ],
            "candidate_results": candidate_results,
            "selected_policies": {name: policy.manifest() for name, policy in selected.items()},
            "code_policy_separation_evidence": {
                "strict_policy_required_by_calibration": code_strict_required,
                "preferred_after_calibration": code_preferred,
                "incumbent_metrics": code_incumbent,
                "strict_metrics": code_strict,
            },
            "selection_gate": {"minimum_recall": 0.75, "maximum_false_removal_risk": 0.25},
            "determinism_and_restart": selected_validation,
            "false_positive_review_sample": false_positive_review[:10],
        },
        "bounded_current_corpus_pass": {
            "input_path": str(args.data10_corpus),
            "input_sha256": input_corpus_identity,
            "authority": "PROJECT_AUTHORED_SYNTHETIC_ONLY",
            "representative_external_corpus": False,
            "all_currently_committed_data10_mechanics_records_processed": True,
            "input_records": input_records,
            "removed_records": removed_records,
            "survivor_records": input_records - removed_records,
            "input_bytes": input_bytes,
            "removed_bytes": removed_bytes,
            "document_reduction_ratio": removed_records / input_records,
            "byte_reduction_ratio": 0.0 if input_bytes == 0 else removed_bytes / input_bytes,
            "by_modality": corpus_exec,
            "surviving_corpus": corpus_identity,
        },
        "real_candidate_corpus_gate": {
            "external_training_eligible_sources_available_in_composed_input": 0,
            "status": "BLOCKED_NO_REAL_TRAINING_ELIGIBLE_UK_EN_CODE_CORPUS",
            "real_full_corpus_pass_claimed": False,
            "real_surviving_corpus_identity": None,
            "required_reentry": "rerun this same DATA-30 tool after DATA-21/22/23/24/25 provide manifested rights-approved corpus shards",
        },
        "truth_boundary": {
            "lexical_minhash_only": True,
            "semantic_deduplication_claimed": False,
            "translations_treated_as_semantic_only_preserve_calibration": True,
            "model_training_performed": False,
            "current_bounded_pass_is_mechanics_smoke_not_representative_corpus_evidence": True,
        },
    }
    core = dict(report)
    report["report_sha256"] = sha256_bytes(canonical_json_bytes(core))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json_bytes(report))
    print(json.dumps({
        "report_sha256": report["report_sha256"],
        "selected_natural": selected["natural"].name,
        "selected_code": selected["code"].name,
        "current_input_records": input_records,
        "current_removed_records": removed_records,
        "surviving_corpus_identity": corpus_identity["surviving_corpus_identity"],
        "real_corpus_status": report["real_candidate_corpus_gate"]["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
