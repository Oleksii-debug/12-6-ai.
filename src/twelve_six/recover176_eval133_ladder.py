"""RECOVER-176: run immutable EVAL-133 against the MILESTONE-150 learned Base ladder."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from twelve_six.checkpoint import load_trainer_checkpoint, verify_checkpoint
from twelve_six.en_raw_diagnostic import (
    AUTHORITY as EVAL133_AUTHORITY,
    PHENOMENA,
    RESERVED_INDEX_SHA256,
    RESERVATION_SHA256,
    SUITE_ID,
    SUITE_SHA256,
    SUITE_VERSION,
    _state_hash,
    evaluate_model,
    load_suite,
    validate_reservation,
)
from twelve_six.eval_reservations import canonical_json_sha256, training_text_collisions
from twelve_six.milestone150_learned_base_ladder import (
    SCALE_ORDER,
    init_spec,
    model_spec,
    trainer_config,
    validate_ladder,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer

SCHEMA = "12-6.recover176-eval133-learned-base-ladder.v1"
AUTHORITY = "LOCAL_FREE_M150_LEARNED_BASE_LADDER_WITH_IMMUTABLE_EVAL133_EN_RAW"
REPOSITORY = "Oleksii-debug/12-6-ai."
RECOVERY_BRANCH = "recover176/eval133-en-m150-ladder-20260826"
M150_BRANCH = "milestone150/learned-base-ladder-v1-20260826"
M150_SOURCE_SHA = "1037439f65c48529904be170064bf69d0c75d18b"
EXPECTED_PARAMS = {"100k": 95_568, "500k": 467_808, "1m": 1_037_696}

# Original Git blob object IDs from the accepted EVAL-133 branch.
IMMUTABLE_GIT_BLOBS = {
    "data/evaluation/eval133_en_raw_v1.jsonl": "49bb7fc41b56f70063c4819e3ab4eaefbae333a9",
    "data/evaluation/reserved/eval133_en_raw_v1.reservation.json": "0408b1287b2ee03fe2b8461f46dd571dc1b798f4",
    "data/evaluation/reserved/index.json": "f6542f1675f48b5675df940b791770dbe37e2e64",
    "src/twelve_six/cloze.py": "a5cdc860ece213242ddedaf17ea8d8b111e7e843",
    "src/twelve_six/en_raw_diagnostic.py": "60bba5e50acca81e99ef498b344acd727f4862ec",
    "src/twelve_six/eval_reservations.py": "b6336fe40154559ca283175cba3f321454952bdc",
    "tests/test_en_raw_diagnostic.py": "32d4ecdc25fa63182938c1a98543d03d7d533979",
}


class RecoveryError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RecoveryError(f"{path} must contain a JSON object")
    return value


def _git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    h = hashlib.sha1()
    h.update(f"blob {len(data)}\0".encode("ascii"))
    h.update(data)
    return h.hexdigest()


def validate_immutable_eval133(repo: Path) -> dict[str, Any]:
    observed = {}
    for relative, expected in IMMUTABLE_GIT_BLOBS.items():
        path = repo / relative
        if not path.is_file():
            raise RecoveryError(f"immutable EVAL-133 artifact missing: {relative}")
        actual = _git_blob_sha1(path)
        if actual != expected:
            raise RecoveryError(
                f"immutable EVAL-133 artifact drift: {relative}: {actual} != {expected}"
            )
        observed[relative] = {
            "git_blob_sha1": actual,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }

    items = load_suite(repo)
    reservation = validate_reservation(repo, items)
    if SUITE_ID != "eval133-en-raw-v1" or SUITE_VERSION != "1.0.0":
        raise RecoveryError("accepted EVAL-133 suite identity drift")
    if len(items) != 32 or {x["phenomenon"] for x in items} != set(PHENOMENA):
        raise RecoveryError("accepted EVAL-133 suite shape drift")
    return {
        "suite_id": SUITE_ID,
        "suite_version": SUITE_VERSION,
        "suite_sha256": SUITE_SHA256,
        "reservation_sha256": RESERVATION_SHA256,
        "reserved_index_sha256": RESERVED_INDEX_SHA256,
        "authority": EVAL133_AUTHORITY,
        "items": len(items),
        "phenomena": list(PHENOMENA),
        "immutable_git_blobs": observed,
        "reservation": reservation,
    }


def _jsonl_texts(path: Path) -> list[str]:
    if not path.is_file():
        return []
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row = json.loads(raw)
            if isinstance(row, Mapping) and isinstance(row.get("text"), str):
                rows.append(str(row["text"]))
    return rows


def _data25_texts(
    m150_evidence: Path, manifest: Mapping[str, Any]
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"train": [], "validation": []}
    corpus = m150_evidence / "corpus-a"
    for shard in manifest["shards"]:
        path = corpus / str(shard["path"])
        if _sha256_file(path) != shard["sha256"]:
            raise RecoveryError(f"DATA-25 shard hash mismatch: {shard['path']}")
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.strip():
                    continue
                row = json.loads(raw)
                split = row.get("split")
                text = row.get("text")
                if split not in result or not isinstance(text, str):
                    raise RecoveryError("DATA-25 shard row schema drift")
                result[str(split)].append(text)
    return result


def verify_reserved_exclusion(
    repo: Path, m150_evidence: Path, ladder: Mapping[str, Any]
) -> dict[str, Any]:
    manifest = _read_json(m150_evidence / "corpus-manifest.json")
    expected_corpus = ladder["truth_model"]["corpus_identity_sha256"]
    if manifest.get("corpus_identity_sha256") != expected_corpus:
        raise RecoveryError("DATA-25 corpus identity differs from M150 ladder truth")

    data25 = _data25_texts(m150_evidence, manifest)
    sources = {
        "data25_train": data25["train"],
        "data25_validation": data25["validation"],
        "legacy_s0_packaged_train": _jsonl_texts(
            repo / "data/s0/packaged/train.jsonl"
        ),
        "legacy_s0_packaged_validation": _jsonl_texts(
            repo / "data/s0/packaged/validation.jsonl"
        ),
    }

    scans = {}
    for name, texts in sources.items():
        collisions = training_text_collisions(repo, texts)
        if collisions:
            raise RecoveryError(
                f"EVAL-133 reserved material collision in {name}: {collisions}"
            )
        scans[name] = {
            "documents_scanned": len(texts),
            "collisions": [],
            "passed": True,
        }

    return {
        "status": "PASS",
        "method": (
            "NFKC+casefold+whitespace normalized exact-full-alternative and reserved "
            "context/full-continuation substring scan from immutable EVAL-133 reservation"
        ),
        "corpus_identity_sha256": expected_corpus,
        "sources": scans,
    }


def _trainer_hash(trainer: Trainer) -> str:
    state: Any = trainer.state_dict()
    if hasattr(state, "__dataclass_fields__"):
        state = asdict(state)
    return _state_hash(state)


def _evaluate_checkpoint(
    m150_evidence: Path,
    ladder: Mapping[str, Any],
    scale: str,
    step: int,
    roles: list[str],
    items: list[dict[str, str]],
) -> dict[str, Any]:
    scale_report = ladder["scales"][scale]
    spec = model_spec(scale)
    init = init_spec()
    if spec.parameter_count() != EXPECTED_PARAMS[scale]:
        raise RecoveryError(f"{scale} parameter count drift")
    if spec.identity_sha256() != scale_report["model"]["model_spec_sha256"]:
        raise RecoveryError(f"{scale} ModelSpec identity drift")

    checkpoint = m150_evidence / scale / f"checkpoint-{step:04d}"
    manifest = verify_checkpoint(checkpoint)
    identity = manifest["identity"]
    expected_checkpoint_id = None
    if "best" in roles:
        expected_checkpoint_id = scale_report["checkpoints"]["best_checkpoint_id"]
    if "final" in roles:
        final_id = scale_report["checkpoints"]["final_checkpoint_id"]
        if expected_checkpoint_id is not None and final_id != expected_checkpoint_id:
            raise RecoveryError(f"{scale} one checkpoint cannot have divergent role IDs")
        expected_checkpoint_id = final_id
    if (
        expected_checkpoint_id is not None
        and manifest["checkpoint_id"] != expected_checkpoint_id
    ):
        raise RecoveryError(f"{scale} checkpoint role identity mismatch")

    tokenizer = ByteTokenizer()
    run_manifest = _read_json(m150_evidence / scale / "run-manifest.json")
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, trainer_config(), device="cpu")
    loaded = load_trainer_checkpoint(
        checkpoint,
        model=model,
        trainer=trainer,
        strict_model=True,
        restore_rng=False,
        expected_git_sha=M150_SOURCE_SHA,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        expected_dataset_manifest_hash=ladder["truth_model"]["corpus_identity_sha256"],
        expected_run_manifest_hash=run_manifest["identity_sha256"],
        expected_seed=1337,
    )
    if int(loaded.manifest["identity"]["step"]) != step:
        raise RecoveryError(f"{scale} loaded checkpoint step mismatch")

    before = _trainer_hash(trainer)
    result = evaluate_model(model, tokenizer, items)
    after = _trainer_hash(trainer)
    if before != after:
        raise RecoveryError(f"{scale} EVAL-133 mutated Trainer state")

    return {
        "roles": roles,
        "step": step,
        "checkpoint": f"{scale}/checkpoint-{step:04d}",
        "checkpoint_id": manifest["checkpoint_id"],
        "checkpoint_identity": identity,
        "model_non_mutation": True,
        "trainer_non_mutation": True,
        "metrics": result,
    }


def _role_steps(scale_report: Mapping[str, Any]) -> list[tuple[int, list[str]]]:
    roles_by_step: dict[int, list[str]] = {0: ["random_init"]}
    best = int(scale_report["evaluation"]["best_step"])
    final = 1000
    roles_by_step.setdefault(best, []).append("best")
    roles_by_step.setdefault(final, []).append("final")
    return sorted(roles_by_step.items())


def _scale_trend(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for scale in SCALE_ORDER:
        checkpoints = results[scale]["checkpoints"]
        random_row = next(x for x in checkpoints if "random_init" in x["roles"])
        best_row = next(x for x in checkpoints if "best" in x["roles"])
        random_metrics = random_row["metrics"]
        best_metrics = best_row["metrics"]
        row: dict[str, Any] = {
            "scale": scale,
            "parameters": EXPECTED_PARAMS[scale],
            "best_step": best_row["step"],
            "random_init_accuracy": random_metrics["accuracy"],
            "best_accuracy": best_metrics["accuracy"],
            "accuracy_delta_vs_random": (
                best_metrics["accuracy"] - random_metrics["accuracy"]
            ),
            "random_init_mean_log_likelihood_margin": random_metrics[
                "mean_log_likelihood_margin"
            ],
            "best_mean_log_likelihood_margin": best_metrics[
                "mean_log_likelihood_margin"
            ],
            "mean_log_likelihood_margin_delta_vs_random": (
                best_metrics["mean_log_likelihood_margin"]
                - random_metrics["mean_log_likelihood_margin"]
            ),
            "best_mean_normalized_margin_nats_per_token": best_metrics[
                "mean_normalized_margin_nats_per_token"
            ],
            "best_mean_normalized_margin_nats_per_utf8_byte": best_metrics[
                "mean_normalized_margin_nats_per_utf8_byte"
            ],
        }
        if previous is not None:
            row["adjacent_scale_change"] = {
                "from": previous["scale"],
                "accuracy": row["best_accuracy"] - previous["best_accuracy"],
                "mean_log_likelihood_margin": (
                    row["best_mean_log_likelihood_margin"]
                    - previous["best_mean_log_likelihood_margin"]
                ),
                "normalized_margin_nats_per_token": (
                    row["best_mean_normalized_margin_nats_per_token"]
                    - previous["best_mean_normalized_margin_nats_per_token"]
                ),
            }
        rows.append(row)
        previous = row
    return rows


def _execution_environment(repo: Path) -> dict[str, Any]:
    rels = (
        "requirements/locks/linux-x86_64/toolchain.lock.txt",
        "requirements/locks/linux-x86_64/runtime.lock.txt",
        "requirements/locks/linux-x86_64/dev.lock.txt",
    )
    files = {relative: _sha256_file(repo / relative) for relative in rels}
    return {
        "purpose": "universal local CPU execution for runtime plus tests",
        "identity": "canonical-linux-x86_64-full-locked-toolchain-runtime-dev",
        "hash_locked": True,
        "tests_installed_from_dev_lock": True,
        "files": files,
        "combined_sha256": canonical_json_sha256(files),
    }


def run(
    repo: Path,
    m150_evidence: Path,
    output: Path,
    expected_m150_source_sha: str = M150_SOURCE_SHA,
) -> dict[str, Any]:
    repo = repo.resolve()
    m150_evidence = m150_evidence.resolve()
    if expected_m150_source_sha != M150_SOURCE_SHA:
        raise RecoveryError("RECOVER-176 is pinned to the accepted M150 exact head")

    immutable = validate_immutable_eval133(repo)
    ladder = validate_ladder(
        m150_evidence / "ladder-report.json",
        expected_source_sha=M150_SOURCE_SHA,
    )
    if ladder["source"]["branch"] != M150_BRANCH:
        raise RecoveryError("unexpected M150 ladder branch identity")
    if ladder["ten_million"]["status"] != (
        "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE"
    ):
        raise RecoveryError("10M truth boundary unexpectedly changed")

    exclusion = verify_reserved_exclusion(repo, m150_evidence, ladder)
    items = load_suite(repo)
    scale_results = {}
    for scale in SCALE_ORDER:
        if int(ladder["scales"][scale]["model"]["parameter_count"]) != EXPECTED_PARAMS[scale]:
            raise RecoveryError(f"{scale} ladder parameter count drift")
        checkpoints = [
            _evaluate_checkpoint(m150_evidence, ladder, scale, step, roles, items)
            for step, roles in _role_steps(ladder["scales"][scale])
        ]
        scale_results[scale] = {
            "parameters": EXPECTED_PARAMS[scale],
            "model_spec_sha256": ladder["scales"][scale]["model"][
                "model_spec_sha256"
            ],
            "checkpoints": checkpoints,
        }

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "source": {
            "repository": REPOSITORY,
            "recovery_branch": RECOVERY_BRANCH,
            "recovery_source_sha": _git_head(repo),
            "m150_branch": M150_BRANCH,
            "m150_source_sha": M150_SOURCE_SHA,
            "paid_compute": False,
        },
        "execution_environment": _execution_environment(repo),
        "eval133": {
            "identity": immutable,
            "reserved_evaluation_exclusion": exclusion,
            "scales": scale_results,
            "scale_trend_best_checkpoint": _scale_trend(scale_results),
            "interpretation_boundary": {
                "raw_lm_minimal_pair_diagnostic": True,
                "instruction_following_claim": False,
                "broad_english_proficiency_claim": False,
                "intelligence_claim": False,
                "alignment_claim": False,
                "production_readiness_claim": False,
            },
        },
        "learned_base_ladder_v1": ladder,
        "claims": {
            "learned_base_ladder": True,
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "paid_compute": False,
            "instruction_following": False,
            "intelligence": False,
            "alignment": False,
            "production_readiness": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _write_json(output, report)
    return report


def validate_report(
    path: Path, expected_m150_source_sha: str = M150_SOURCE_SHA
) -> dict[str, Any]:
    report = _read_json(path)
    unsigned = {k: v for k, v in report.items() if k != "report_sha256"}
    if report.get("report_sha256") != canonical_json_sha256(unsigned):
        raise RecoveryError("RECOVER-176 report self hash mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise RecoveryError("RECOVER-176 report schema/authority mismatch")
    if report["source"]["m150_source_sha"] != expected_m150_source_sha:
        raise RecoveryError("RECOVER-176 M150 source mismatch")
    identity = report["eval133"]["identity"]
    expected_ids = (
        (identity["suite_sha256"], SUITE_SHA256),
        (identity["reservation_sha256"], RESERVATION_SHA256),
        (identity["reserved_index_sha256"], RESERVED_INDEX_SHA256),
    )
    if any(actual != expected for actual, expected in expected_ids):
        raise RecoveryError("RECOVER-176 immutable EVAL-133 identity mismatch")
    if report["eval133"]["reserved_evaluation_exclusion"]["status"] != "PASS":
        raise RecoveryError("RECOVER-176 reserved-evaluation exclusion failed")
    if set(report["eval133"]["scales"]) != set(SCALE_ORDER):
        raise RecoveryError("RECOVER-176 comparable scale set drift")
    for scale in SCALE_ORDER:
        rows = report["eval133"]["scales"][scale]["checkpoints"]
        roles = {role for row in rows for role in row["roles"]}
        if not {"random_init", "best", "final"}.issubset(roles):
            raise RecoveryError(f"{scale} missing random/best/final EVAL-133 coverage")
        if any(
            row["model_non_mutation"] is not True
            or row["trainer_non_mutation"] is not True
            for row in rows
        ):
            raise RecoveryError(f"{scale} non-mutation evidence failed")
        for row in rows:
            metrics = row["metrics"]
            if (
                "per_phenomenon" not in metrics
                or set(metrics["per_phenomenon"]) != set(PHENOMENA)
            ):
                raise RecoveryError(f"{scale} per-phenomenon coverage drift")
            for key in (
                "accuracy",
                "mean_log_likelihood_margin",
                "mean_normalized_margin_nats_per_token",
                "mean_normalized_margin_nats_per_utf8_byte",
            ):
                if key not in metrics:
                    raise RecoveryError(f"{scale} missing EVAL-133 metric: {key}")
    if report["learned_base_ladder_v1"]["ten_million"]["status"] != (
        "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE"
    ):
        raise RecoveryError("RECOVER-176 incorrectly promoted 10M")
    for key in (
        "foreign_pretrained_weights",
        "sft",
        "rlhf",
        "dpo",
        "paid_compute",
        "instruction_following",
        "intelligence",
        "alignment",
        "production_readiness",
    ):
        if report["claims"][key] is not False:
            raise RecoveryError(f"unsupported claim enabled: {key}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--repo-root", type=Path, required=True)
    run_p.add_argument("--m150-evidence", type=Path, required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--expected-m150-source-sha", default=M150_SOURCE_SHA)
    val_p = sub.add_parser("validate")
    val_p.add_argument("report", type=Path)
    val_p.add_argument("--expected-m150-source-sha", default=M150_SOURCE_SHA)
    args = parser.parse_args(argv)

    if args.cmd == "run":
        report = run(
            args.repo_root,
            args.m150_evidence,
            args.output,
            args.expected_m150_source_sha,
        )
        print(
            json.dumps(
                {
                    "validation_ready": True,
                    "m150_source_sha": report["source"]["m150_source_sha"],
                    "suite_sha256": report["eval133"]["identity"]["suite_sha256"],
                    "reserved_exclusion": report["eval133"][
                        "reserved_evaluation_exclusion"
                    ]["status"],
                    "scale_trend": report["eval133"][
                        "scale_trend_best_checkpoint"
                    ],
                    "ten_million": report["learned_base_ladder_v1"][
                        "ten_million"
                    ]["status"],
                    "report_sha256": report["report_sha256"],
                },
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
        )
    else:
        report = validate_report(args.report, args.expected_m150_source_sha)
        print(
            json.dumps(
                {
                    "validation": "PASS",
                    "report_sha256": report["report_sha256"],
                    "ten_million": report["learned_base_ladder_v1"]["ten_million"][
                        "status"
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
