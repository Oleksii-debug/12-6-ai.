"""RECOVER-176: frozen EVAL-133 over the terminal M150 learned Base incumbent."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import torch

from twelve_six.checkpoint import load_trainer_checkpoint, verify_checkpoint
from twelve_six.data.corpus_v01 import verify_rebuild
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
from twelve_six.inference.first_party import load_first_party_backend
from twelve_six.milestone100_first_learned import _state_hash as m150_model_state_hash
from twelve_six.milestone150_learned_base_ladder import (
    SCALE_ORDER,
    SEED,
    init_spec,
    model_spec,
    trainer_config,
    validate_ladder,
)
from twelve_six.model import TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer
from twelve_six.training import Trainer

SCHEMA = "12-6.recover176-eval133-learned-base-ladder.v2"
AUTHORITY = "LOCAL_FREE_M150_LEARNED_BASE_LADDER_WITH_IMMUTABLE_EVAL133_EN_RAW"
REPOSITORY = "Oleksii-debug/12-6-ai."
RECOVERY_BRANCH = "recover176/eval133-en-m150-ladder-20260826"
M150_BRANCH = "milestone150/learned-base-ladder-v1-20260826"
M150_SOURCE_SHA = "5838cd16869dcfcf762368d8673eddf52d51b7e3"
M150_RUN_ID = 32937411703
M150_ARTIFACT_ID = 9595677772
M150_ARTIFACT_DIGEST = "sha256:c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71"
M150_ARTIFACT_NAME = "milestone150-learned-base-ladder-v1"
M150_LADDER_REPORT_SHA256 = "1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a"
M150_CORPUS_ID = "422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8"
M150_EVALUATION_ID = "7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113"
EXPECTED_PARAMS = {"100k": 95_568, "500k": 467_808, "1m": 1_037_696}
CORPUS_CONFIG = Path("configs/data/corpus_v01.json")

# Original Git blob object IDs from the accepted EVAL-133 branch. These are
# intentionally unchanged by EVAL-216; recovery fixes the consumer only.
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


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


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


def _artifact_root(path: Path) -> Path:
    for candidate in (path, path / "milestone150-evidence"):
        if (candidate / "ladder-report.json").is_file():
            return candidate
    raise RecoveryError(f"M150 ladder-report.json not found under {path}")


def validate_immutable_eval133(repo: Path) -> dict[str, Any]:
    observed: dict[str, Any] = {}
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
    if len(items) != 32 or {row["phenomenon"] for row in items} != set(PHENOMENA):
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
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if isinstance(row, Mapping) and isinstance(row.get("text"), str):
            rows.append(str(row["text"]))
    return rows


def verify_reserved_exclusion(repo: Path, expected_corpus_id: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="recover176-data25-") as temp:
        base = Path(temp)
        built = verify_rebuild(
            repo / CORPUS_CONFIG,
            base / "corpus-a",
            base / "corpus-b",
        )
        if built.get("corpus_identity_sha256") != expected_corpus_id:
            raise RecoveryError("reconstructed DATA-25 identity differs from M150 producer")
        split_texts: dict[str, list[str]] = {"train": [], "validation": []}
        for shard in built["shards"]:
            path = base / "corpus-a" / str(shard["path"])
            if _sha256_file(path) != shard["sha256"]:
                raise RecoveryError(f"DATA-25 shard hash mismatch: {shard['path']}")
            for raw in path.read_text(encoding="utf-8").splitlines():
                if not raw.strip():
                    continue
                row = json.loads(raw)
                split = row.get("split")
                text = row.get("text")
                if split not in split_texts or not isinstance(text, str):
                    raise RecoveryError("DATA-25 shard row schema drift")
                split_texts[str(split)].append(text)

    sources = {
        "data25_train": split_texts["train"],
        "data25_validation": split_texts["validation"],
        "legacy_s0_packaged_train": _jsonl_texts(repo / "data/s0/packaged/train.jsonl"),
        "legacy_s0_packaged_validation": _jsonl_texts(repo / "data/s0/packaged/validation.jsonl"),
    }
    scans: dict[str, Any] = {}
    for name, texts in sources.items():
        collisions = training_text_collisions(repo, texts)
        if collisions:
            raise RecoveryError(f"EVAL-133 reserved material collision in {name}: {collisions}")
        scans[name] = {
            "documents_scanned": len(texts),
            "collisions": [],
            "passed": True,
        }
    return {
        "status": "PASS",
        "d06_training_rejection_verified": True,
        "method": (
            "immutable EVAL-133 reservation plus NFKC+casefold+whitespace normalized "
            "full-alternative/context-continuation collision scan over deterministic DATA-25 "
            "train+validation and retained legacy S0 packaged material"
        ),
        "corpus_identity_sha256": expected_corpus_id,
        "sources": scans,
    }


def _trainer_hash(trainer: Trainer) -> str:
    state: Any = trainer.state_dict()
    if hasattr(state, "__dataclass_fields__"):
        state = asdict(state)
    return _state_hash(state)


def _scale_model_spec_sha(scale_report: Mapping[str, Any]) -> str:
    model = scale_report.get("model")
    if not isinstance(model, Mapping):
        raise RecoveryError("M150 scale report model object missing")
    value = model.get("spec_sha256")
    if not isinstance(value, str):
        raise RecoveryError("M150 scale report model.spec_sha256 missing")
    return value


def _verify_tokenizer_identity(tokenizer: ByteTokenizer, ladder: Mapping[str, Any]) -> dict[str, Any]:
    expected = ladder["truth_model"]["tokenizer"]
    observed = {
        "version": tokenizer.identity.version,
        "config_sha256": tokenizer.identity.config_sha256,
        "vocab_sha256": tokenizer.identity.vocab_sha256,
        "vocab_size": tokenizer.identity.vocab_size,
        "special_tokens": dict(tokenizer.identity.special_tokens),
    }
    for key in ("version", "config_sha256", "vocab_sha256", "vocab_size"):
        if observed[key] != expected[key]:
            raise RecoveryError(f"canonical tokenizer identity mismatch: {key}")
    if observed["special_tokens"] != expected["special_tokens"]:
        raise RecoveryError("canonical tokenizer special-token identity mismatch")
    return {"status": "PASS", **observed}


def _completion_bpb(metrics: Mapping[str, Any]) -> dict[str, float | int]:
    rows = list(metrics["item_scores"])
    result: dict[str, float | int] = {}
    for side in ("preferred", "dispreferred"):
        total_nats = math.fsum(float(row[f"{side}_log_likelihood"]) for row in rows)
        total_bytes = sum(int(row[f"{side}_utf8_bytes"]) for row in rows)
        if total_bytes <= 0:
            raise RecoveryError("EVAL-133 completion byte accounting is empty")
        result[f"{side}_log_likelihood_nats"] = total_nats
        result[f"{side}_utf8_bytes"] = total_bytes
        result[f"{side}_conditional_bpb"] = -total_nats / (math.log(2.0) * total_bytes)
    return result


def _score_with_trainer(
    trainer: Trainer,
    tokenizer: ByteTokenizer,
    items: list[dict[str, str]],
) -> dict[str, Any]:
    model = trainer.model
    model_before = _state_hash(model.state_dict())
    mode_before = model.training
    trainer_before = _trainer_hash(trainer)
    tokens_before = int(trainer.tokens_seen)
    step_before = int(trainer.optimizer_step)
    metrics = evaluate_model(model, tokenizer, items)
    model_after = _state_hash(model.state_dict())
    trainer_after = _trainer_hash(trainer)
    tokens_after = int(trainer.tokens_seen)
    step_after = int(trainer.optimizer_step)
    if model_before != model_after or model.training != mode_before:
        raise RecoveryError("EVAL-133 mutated model state or mode")
    if trainer_before != trainer_after:
        raise RecoveryError("EVAL-133 mutated Trainer state")
    if tokens_after != tokens_before or step_after != step_before:
        raise RecoveryError("EVAL-133 changed optimized-token or optimizer-step counters")
    metrics["completion_bpb"] = _completion_bpb(metrics)
    return {
        "metrics": metrics,
        "non_mutation": {
            "model_state": True,
            "model_mode": True,
            "trainer_state": True,
            "model_state_sha256_before": model_before,
            "model_state_sha256_after": model_after,
            "trainer_state_sha256_before": trainer_before,
            "trainer_state_sha256_after": trainer_after,
            "optimized_tokens_before": tokens_before,
            "optimized_tokens_after": tokens_after,
            "optimized_tokens_delta": tokens_after - tokens_before,
            "optimizer_step_before": step_before,
            "optimizer_step_after": step_after,
            "optimizer_step_delta": step_after - step_before,
        },
    }


def _random_cell(
    scale: str,
    producer_root: Path,
    ladder: Mapping[str, Any],
    items: list[dict[str, str]],
) -> dict[str, Any]:
    scale_report = ladder["scales"][scale]
    spec = model_spec(scale)
    init = init_spec()
    if spec.parameter_count() != EXPECTED_PARAMS[scale]:
        raise RecoveryError(f"{scale} parameter count drift")
    if spec.identity_sha256() != _scale_model_spec_sha(scale_report):
        raise RecoveryError(f"{scale} ModelSpec identity drift")
    phase1 = _read_json(producer_root / scale / "phase1.json")
    torch.manual_seed(SEED)
    model = TwelveSixDecoder(spec, init)
    reconstructed = m150_model_state_hash(model)
    expected = phase1["model"]["random_init_state_sha256"]
    if reconstructed != expected:
        raise RecoveryError(f"{scale} producer random-init reconstruction mismatch")
    tokenizer = ByteTokenizer()
    tokenizer_proof = _verify_tokenizer_identity(tokenizer, ladder)
    trainer = Trainer(model, trainer_config(), device="cpu")
    scored = _score_with_trainer(trainer, tokenizer, items)
    return {
        "role": "random_init",
        "identity": {
            "kind": "deterministic_random_init_reconstruction",
            "producer_phase1_identity_sha256": phase1["identity_sha256"],
            "random_init_state_sha256": reconstructed,
            "model_spec_sha256": spec.identity_sha256(),
            "init_spec_sha256": init.identity_sha256(),
            "parameter_count": spec.parameter_count(),
            "seed": SEED,
        },
        "tokenizer": tokenizer_proof,
        "first_party_scorer": {
            "status": "PASS",
            "path": "twelve_six.en_raw_diagnostic.evaluate_model -> twelve_six.cloze.conditional_log_likelihood",
            "immutable_scorer_blob": IMMUTABLE_GIT_BLOBS["src/twelve_six/en_raw_diagnostic.py"],
            "immutable_cloze_blob": IMMUTABLE_GIT_BLOBS["src/twelve_six/cloze.py"],
        },
        **scored,
    }


def _learned_cell(
    scale: str,
    producer_root: Path,
    ladder: Mapping[str, Any],
    items: list[dict[str, str]],
) -> dict[str, Any]:
    scale_report = ladder["scales"][scale]
    if scale_report["fresh_verification"]["status"] != "PASS":
        raise RecoveryError(f"{scale} producer fresh verification is not PASS")
    spec = model_spec(scale)
    init = init_spec()
    if spec.identity_sha256() != _scale_model_spec_sha(scale_report):
        raise RecoveryError(f"{scale} ModelSpec identity drift")
    checkpoint = producer_root / "retained" / scale / "best"
    manifest = verify_checkpoint(checkpoint)
    expected_checkpoint = scale_report["checkpoints"]["best_checkpoint_id"]
    if manifest["checkpoint_id"] != expected_checkpoint:
        raise RecoveryError(f"{scale} retained best checkpoint identity mismatch")

    backend = load_first_party_backend(checkpoint)
    diagnostics = backend.diagnostics()
    if diagnostics["checkpoint_id"] != expected_checkpoint:
        raise RecoveryError(f"{scale} first-party checkpoint ID mismatch")
    if diagnostics["git_sha"] != M150_SOURCE_SHA:
        raise RecoveryError(f"{scale} first-party producer SHA mismatch")
    if diagnostics["model_spec_sha256"] != spec.identity_sha256():
        raise RecoveryError(f"{scale} first-party ModelSpec mismatch")
    if int(diagnostics["parameter_count"]) != EXPECTED_PARAMS[scale]:
        raise RecoveryError(f"{scale} first-party parameter count mismatch")
    ids = backend.encode(items[0]["context"])
    logits = list(backend.next_token_logits(ids))
    if len(logits) != 256 or not all(math.isfinite(float(value)) for value in logits):
        raise RecoveryError(f"{scale} first-party logits invalid")

    tokenizer = ByteTokenizer()
    tokenizer_proof = _verify_tokenizer_identity(tokenizer, ladder)
    if diagnostics["tokenizer_config_sha256"] != tokenizer.identity.config_sha256:
        raise RecoveryError(f"{scale} checkpoint tokenizer config mismatch")
    if diagnostics["tokenizer_vocab_sha256"] != tokenizer.identity.vocab_sha256:
        raise RecoveryError(f"{scale} checkpoint tokenizer vocab mismatch")
    if diagnostics["dataset_manifest_sha256"] != M150_CORPUS_ID:
        raise RecoveryError(f"{scale} checkpoint DATA-25 identity mismatch")

    run_manifest = _read_json(producer_root / scale / "run-manifest.json")
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
        expected_dataset_manifest_hash=M150_CORPUS_ID,
        expected_run_manifest_hash=run_manifest["identity_sha256"],
        expected_seed=SEED,
    )
    if loaded.manifest["checkpoint_id"] != expected_checkpoint:
        raise RecoveryError(f"{scale} Trainer load checkpoint ID mismatch")
    if m150_model_state_hash(model) != m150_model_state_hash(backend.model):
        raise RecoveryError(f"{scale} Trainer load and first-party backend model state differ")

    scored = _score_with_trainer(trainer, tokenizer, items)
    return {
        "role": "learned_best",
        "identity": {
            "kind": "m150_retained_best_checkpoint",
            "checkpoint_id": expected_checkpoint,
            "checkpoint_step": int(diagnostics["step"]),
            "optimized_tokens": int(diagnostics["tokens_seen"]),
            "producer_git_sha": diagnostics["git_sha"],
            "model_spec_sha256": diagnostics["model_spec_sha256"],
            "parameter_count": int(diagnostics["parameter_count"]),
            "run_manifest_sha256": diagnostics["run_manifest_sha256"],
            "producer_fresh_verification_status": scale_report["fresh_verification"]["status"],
        },
        "tokenizer": tokenizer_proof,
        "first_party_scorer": {
            "status": "PASS",
            "path": "verified M150 checkpoint + twelve_six.en_raw_diagnostic.evaluate_model -> twelve_six.cloze.conditional_log_likelihood",
            "backend_logits_path": "FirstPartyInferenceBackend.next_token_logits",
            "backend_logits_finite": True,
            "backend_vocab_size": len(logits),
            "immutable_scorer_blob": IMMUTABLE_GIT_BLOBS["src/twelve_six/en_raw_diagnostic.py"],
            "immutable_cloze_blob": IMMUTABLE_GIT_BLOBS["src/twelve_six/cloze.py"],
        },
        **scored,
    }


def _learned_vs_random(cells: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scale in SCALE_ORDER:
        random = cells[f"{scale}_random_init"]["metrics"]
        learned = cells[f"{scale}_learned_best"]["metrics"]
        result[scale] = {
            "pair_accuracy_delta": learned["accuracy"] - random["accuracy"],
            "raw_mean_log_likelihood_margin_delta": (
                learned["mean_log_likelihood_margin"] - random["mean_log_likelihood_margin"]
            ),
            "byte_normalized_accuracy_delta": (
                learned["byte_normalized_accuracy"] - random["byte_normalized_accuracy"]
            ),
            "byte_normalized_margin_delta_nats_per_utf8_byte": (
                learned["mean_normalized_margin_nats_per_utf8_byte"]
                - random["mean_normalized_margin_nats_per_utf8_byte"]
            ),
            "preferred_conditional_bpb_delta": (
                learned["completion_bpb"]["preferred_conditional_bpb"]
                - random["completion_bpb"]["preferred_conditional_bpb"]
            ),
            "context_bpb_delta": learned["context_bpb"]["aggregate"] - random["context_bpb"]["aggregate"],
            "negative_bpb_delta_means_lower_nll": True,
        }
    return result


def _scale_trend(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for scale in SCALE_ORDER:
        metrics = cells[f"{scale}_learned_best"]["metrics"]
        row: dict[str, Any] = {
            "scale": scale,
            "parameters": EXPECTED_PARAMS[scale],
            "pair_accuracy": metrics["accuracy"],
            "mean_log_likelihood_margin": metrics["mean_log_likelihood_margin"],
            "mean_normalized_margin_nats_per_utf8_byte": metrics[
                "mean_normalized_margin_nats_per_utf8_byte"
            ],
            "preferred_conditional_bpb": metrics["completion_bpb"]["preferred_conditional_bpb"],
            "context_bpb": metrics["context_bpb"]["aggregate"],
        }
        if previous is not None:
            row["adjacent_scale_delta"] = {
                "from": previous["scale"],
                "pair_accuracy": row["pair_accuracy"] - previous["pair_accuracy"],
                "raw_margin": row["mean_log_likelihood_margin"] - previous["mean_log_likelihood_margin"],
                "byte_margin": (
                    row["mean_normalized_margin_nats_per_utf8_byte"]
                    - previous["mean_normalized_margin_nats_per_utf8_byte"]
                ),
                "preferred_conditional_bpb": row["preferred_conditional_bpb"] - previous["preferred_conditional_bpb"],
                "context_bpb": row["context_bpb"] - previous["context_bpb"],
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
    producer_dir: Path,
    output: Path,
    producer_sha: str = M150_SOURCE_SHA,
    producer_run_id: int = M150_RUN_ID,
    producer_artifact_id: int = M150_ARTIFACT_ID,
    producer_artifact_digest: str = M150_ARTIFACT_DIGEST,
) -> dict[str, Any]:
    repo = repo.resolve()
    producer_root = _artifact_root(producer_dir.resolve())
    if producer_sha != M150_SOURCE_SHA or producer_run_id != M150_RUN_ID:
        raise RecoveryError("RECOVER-176 producer source/run differs from frozen M150 incumbent")
    if producer_artifact_id != M150_ARTIFACT_ID or producer_artifact_digest != M150_ARTIFACT_DIGEST:
        raise RecoveryError("RECOVER-176 producer artifact identity differs from frozen incumbent")

    immutable = validate_immutable_eval133(repo)
    ladder = validate_ladder(producer_root / "ladder-report.json", producer_sha)
    if ladder["report_sha256"] != M150_LADDER_REPORT_SHA256:
        raise RecoveryError("M150 ladder report identity drift")
    if ladder["source"]["branch"] != M150_BRANCH:
        raise RecoveryError("M150 producer branch identity drift")
    if ladder["truth_model"]["corpus_identity_sha256"] != M150_CORPUS_ID:
        raise RecoveryError("M150 DATA-25 identity drift")
    if ladder["truth_model"]["evaluation_identity"]["identity_sha256"] != M150_EVALUATION_ID:
        raise RecoveryError("M150 common evaluation identity drift")
    if ladder["ten_million"]["status"] != "INCOMPLETE_NO_COMPARABLE_LEARNED_EVIDENCE":
        raise RecoveryError("10M truth boundary unexpectedly changed")

    tokenizer_proof = _verify_tokenizer_identity(ByteTokenizer(), ladder)
    exclusion = verify_reserved_exclusion(repo, M150_CORPUS_ID)
    items = load_suite(repo)
    cells: dict[str, Any] = {}
    for scale in SCALE_ORDER:
        if int(ladder["scales"][scale]["model"]["parameter_count"]) != EXPECTED_PARAMS[scale]:
            raise RecoveryError(f"{scale} ladder parameter count drift")
        cells[f"{scale}_random_init"] = _random_cell(scale, producer_root, ladder, items)
        cells[f"{scale}_learned_best"] = _learned_cell(scale, producer_root, ladder, items)

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "producer": {
            "repository": REPOSITORY,
            "branch": M150_BRANCH,
            "git_sha": producer_sha,
            "workflow_run_id": producer_run_id,
            "artifact_id": producer_artifact_id,
            "artifact_name": M150_ARTIFACT_NAME,
            "artifact_digest": producer_artifact_digest,
            "ladder_report_sha256": ladder["report_sha256"],
            "corpus_identity_sha256": M150_CORPUS_ID,
            "evaluation_identity_sha256": M150_EVALUATION_ID,
        },
        "evaluator": {
            "repository": REPOSITORY,
            "branch": RECOVERY_BRANCH,
            "git_sha": _git_head(repo),
            "worker_id": "EVAL-216-ENGLISH-LADDER-RECOVERY",
            "execution_class": "LOCAL_FREE_GITHUB_HOSTED_CPU",
        },
        "execution_environment": _execution_environment(repo),
        "eval133": {
            "identity": immutable,
            "d06_reservation_and_exclusion": exclusion,
            "tokenizer_identity": tokenizer_proof,
            "evaluations": cells,
            "learned_vs_random": _learned_vs_random(cells),
            "learned_scale_trend": _scale_trend(cells),
            "interpretation_boundary": {
                "raw_lm_minimal_pair_diagnostic": True,
                "instruction_following_claim": False,
                "broad_english_proficiency_claim": False,
                "intelligence_claim": False,
                "alignment_claim": False,
                "production_readiness_claim": False,
            },
        },
        "producer_ladder": ladder,
        "ten_million": {
            "status": "NOT_EVALUATED_NO_TERMINAL_LEARNED_10M",
            "producer_status": ladder["ten_million"]["status"],
            "numeric_results_absent": True,
        },
        "claims": {
            "foreign_pretrained_weights": False,
            "sft": False,
            "rlhf": False,
            "dpo": False,
            "paid_compute": False,
            "instruction_following": False,
            "broad_english_proficiency": False,
            "intelligence": False,
            "alignment": False,
            "production_readiness": False,
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    _write_json(output, report)
    return report


def validate_report(path: Path, expected_evaluator_sha: str | None = None) -> dict[str, Any]:
    report = _read_json(path)
    expected_hash = report.get("report_sha256")
    unsigned = dict(report)
    unsigned.pop("report_sha256", None)
    if expected_hash != canonical_json_sha256(unsigned):
        raise RecoveryError("RECOVER-176 report self hash mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise RecoveryError("RECOVER-176 report schema/authority mismatch")
    producer = report["producer"]
    expected_producer = {
        "git_sha": M150_SOURCE_SHA,
        "workflow_run_id": M150_RUN_ID,
        "artifact_id": M150_ARTIFACT_ID,
        "artifact_digest": M150_ARTIFACT_DIGEST,
        "ladder_report_sha256": M150_LADDER_REPORT_SHA256,
        "corpus_identity_sha256": M150_CORPUS_ID,
        "evaluation_identity_sha256": M150_EVALUATION_ID,
    }
    for key, expected in expected_producer.items():
        if producer.get(key) != expected:
            raise RecoveryError(f"RECOVER-176 producer identity mismatch: {key}")
    if expected_evaluator_sha and report["evaluator"]["git_sha"] != expected_evaluator_sha:
        raise RecoveryError("RECOVER-176 evaluator source SHA mismatch")

    identity = report["eval133"]["identity"]
    for actual, expected in (
        (identity["suite_sha256"], SUITE_SHA256),
        (identity["reservation_sha256"], RESERVATION_SHA256),
        (identity["reserved_index_sha256"], RESERVED_INDEX_SHA256),
    ):
        if actual != expected:
            raise RecoveryError("RECOVER-176 immutable EVAL-133 identity mismatch")
    if report["eval133"]["d06_reservation_and_exclusion"]["status"] != "PASS":
        raise RecoveryError("RECOVER-176 D06 reservation/exclusion failed")
    if report["eval133"]["tokenizer_identity"]["status"] != "PASS":
        raise RecoveryError("RECOVER-176 tokenizer identity failed")

    cells = report["eval133"]["evaluations"]
    for scale in SCALE_ORDER:
        for role in ("random_init", "learned_best"):
            cell = cells[f"{scale}_{role}"]
            proof = cell["non_mutation"]
            if not (proof["model_state"] and proof["model_mode"] and proof["trainer_state"]):
                raise RecoveryError(f"{scale}/{role} non-mutation evidence failed")
            if proof["optimized_tokens_delta"] != 0 or proof["optimizer_step_delta"] != 0:
                raise RecoveryError(f"{scale}/{role} evaluation changed training counters")
            if cell["first_party_scorer"]["status"] != "PASS":
                raise RecoveryError(f"{scale}/{role} first-party scorer proof missing")
            metrics = cell["metrics"]
            if set(metrics["per_phenomenon"]) != set(PHENOMENA):
                raise RecoveryError(f"{scale}/{role} per-phenomenon coverage drift")
            for key in (
                "accuracy",
                "mean_log_likelihood_margin",
                "byte_normalized_accuracy",
                "mean_normalized_margin_nats_per_utf8_byte",
                "completion_bpb",
                "context_bpb",
            ):
                if key not in metrics:
                    raise RecoveryError(f"{scale}/{role} missing metric: {key}")
        learned = cells[f"{scale}_learned_best"]
        if learned["first_party_scorer"].get("backend_logits_finite") is not True:
            raise RecoveryError(f"{scale} first-party checkpoint logits proof missing")

    if report["ten_million"]["numeric_results_absent"] is not True:
        raise RecoveryError("RECOVER-176 must not emit 10M numerics")
    for key, value in report["claims"].items():
        if value is not False:
            raise RecoveryError(f"unsupported claim enabled: {key}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--repo-root", type=Path, required=True)
    run_p.add_argument("--producer-dir", type=Path, required=True)
    run_p.add_argument("--output", type=Path, required=True)
    run_p.add_argument("--producer-sha", default=M150_SOURCE_SHA)
    run_p.add_argument("--producer-run-id", type=int, default=M150_RUN_ID)
    run_p.add_argument("--producer-artifact-id", type=int, default=M150_ARTIFACT_ID)
    run_p.add_argument("--producer-artifact-digest", default=M150_ARTIFACT_DIGEST)
    val_p = sub.add_parser("validate")
    val_p.add_argument("report", type=Path)
    val_p.add_argument("--expected-evaluator-sha")
    args = parser.parse_args(argv)

    if args.command == "run":
        report = run(
            args.repo_root,
            args.producer_dir,
            args.output,
            args.producer_sha,
            args.producer_run_id,
            args.producer_artifact_id,
            args.producer_artifact_digest,
        )
        print(json.dumps({
            "validation_ready": True,
            "producer_sha": report["producer"]["git_sha"],
            "suite_sha256": report["eval133"]["identity"]["suite_sha256"],
            "d06_exclusion": report["eval133"]["d06_reservation_and_exclusion"]["status"],
            "scale_trend": report["eval133"]["learned_scale_trend"],
            "ten_million": report["ten_million"]["status"],
            "report_sha256": report["report_sha256"],
        }, ensure_ascii=False, sort_keys=True, indent=2))
    else:
        report = validate_report(args.report, args.expected_evaluator_sha)
        print(json.dumps({"validation": "PASS", "report_sha256": report["report_sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
