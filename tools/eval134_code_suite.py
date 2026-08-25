"""Execute the reserved EVAL-134 code diagnostic on learned Base controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from tools import research41_learned_scaling as r41
from twelve_six.checkpoint import save_trainer_checkpoint
from twelve_six.code_diagnostic import (
    canonical_json_sha256,
    load_suite,
    score_suite,
    serializable_scores,
    suite_file_sha256,
    summarize,
    tokenizer_diagnostics,
)
from twelve_six.data.pipeline import normalize_text
from twelve_six.model import InitSpec, TwelveSixDecoder
from twelve_six.tokenization.byte import ByteTokenizer
from twelve_six.training import Trainer

SCHEMA = "12-6.eval134-code-scaling.v1"
AUTHORITY = "RESERVED_MECHANISTIC_CODE_DIAGNOSTIC_LOCAL_FREE_ONLY"
SUITE_PATH = Path("eval/reserved/code_diag_v1/probes.jsonl")
MANIFEST_PATH = Path("eval/reserved/code_diag_v1/manifest.json")
REGISTRY_PATH = Path("data/s0/contamination_registry.json")
TRAIN_PATHS = (
    Path("data/s0/packaged/train.jsonl"),
    Path("data/synthetic/data10/uk-en-code-train.txt"),
)
SPEC_INDICES = (0, 2, 3)
EXPECTED_PARAMETERS = (95_568, 467_808, 1_038_464)
DEFAULT_BUDGET = 65_536


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def candidate_texts(probes) -> tuple[str, ...]:
    return tuple(probe.prefix + choice for probe in probes for choice in probe.choices)


def training_documents(root: Path) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for rel in TRAIN_PATHS:
        path = root / rel
        if not path.exists():
            raise RuntimeError(f"required training input missing: {rel}")
        if path.suffix == ".jsonl":
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                text = json.loads(line).get("text")
                if not isinstance(text, str):
                    raise RuntimeError(f"{rel}:{number}: missing text")
                docs.append((f"{rel}:{number}", text))
        else:
            text = path.read_text(encoding="utf-8")
            docs.append((str(rel), text))
            docs.extend(
                (f"{rel}:line:{number}", line)
                for number, line in enumerate(text.splitlines(), 1)
                if line
            )
    return docs


def hash_root(values: set[str]) -> str:
    return sha_text("\n".join(sorted(values)) + "\n")


def verify_reservation(root: Path) -> dict[str, Any]:
    probes = load_suite(root / SUITE_PATH)
    manifest = json.loads((root / MANIFEST_PATH).read_text(encoding="utf-8"))
    registry = json.loads((root / REGISTRY_PATH).read_text(encoding="utf-8"))
    if manifest.get("schema") != "12-6.code-diagnostic-suite.v1":
        raise RuntimeError("suite manifest schema drift")
    if manifest.get("data_sha256") != suite_file_sha256(root / SUITE_PATH):
        raise RuntimeError("suite data hash drift")
    identity_payload = dict(manifest)
    identity = identity_payload.pop("suite_identity_sha256", None)
    if identity != canonical_json_sha256(identity_payload):
        raise RuntimeError("suite identity drift")
    if manifest.get("status") != "RESERVED_EVALUATION_ONLY":
        raise RuntimeError("suite is not reserved evaluation material")

    candidates = candidate_texts(probes)
    exact = {sha_text(value) for value in candidates}
    normalized = {sha_text(normalize_text(value)) for value in candidates}
    row = next(
        (
            item
            for item in registry.get("reserved_evaluation_suites", [])
            if item.get("suite_id") == manifest["suite_id"]
        ),
        None,
    )
    if row is None or row.get("suite_identity_sha256") != identity:
        raise RuntimeError("suite is not identity-bound in contamination registry")
    if row.get("data_sha256") != manifest["data_sha256"]:
        raise RuntimeError("registry suite data hash drift")
    if row.get("candidate_exact_sha256_root") != hash_root(exact):
        raise RuntimeError("reserved exact-candidate root drift")
    if row.get("candidate_normalized_sha256_root") != hash_root(normalized):
        raise RuntimeError("reserved normalized-candidate root drift")

    docs = training_documents(root)
    overlaps: list[dict[str, str]] = []
    for candidate in candidates:
        for source, text in docs:
            if candidate in text:
                overlaps.append({"source": source, "candidate_sha256": sha_text(candidate)})
    if overlaps:
        raise RuntimeError(f"reserved suite overlaps training input: {overlaps[:4]}")
    if "qzv_" in "\n".join(text for _, text in docs):
        raise RuntimeError("synthetic qzv_ namespace already exists in training")

    required = {
        "balanced_delimiters",
        "indentation_sensitive_continuation",
        "operator_type_syntax",
        "simple_function_call_structure",
        "variable_reuse",
        "string_comment_termination",
        "json_like_structure",
        "language_specific_syntax",
    }
    if {probe.category for probe in probes} != required:
        raise RuntimeError("required diagnostic category coverage drift")
    specific = {
        probe.language for probe in probes if probe.category == "language_specific_syntax"
    }
    if specific != {"python", "sql"}:
        raise RuntimeError("language-specific syntax coverage drift")
    return {
        "suite_id": manifest["suite_id"],
        "suite_identity_sha256": identity,
        "data_sha256": manifest["data_sha256"],
        "items": len(probes),
        "candidate_continuations": len(candidates),
        "exact_registry_hashes_verified": len(exact),
        "normalized_registry_hashes_verified": len(normalized),
        "training_documents_scanned": len(docs),
        "training_overlap_count": 0,
        "synthetic_identifier_namespace_absent_from_training": True,
    }


def model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(model.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def score_nonmutating(model, trainer, tokenizer, probes):
    state_before = model_hash(model)
    counters_before = (trainer.optimizer_step, trainer.tokens_seen)
    mode_before = model.training
    scores = score_suite(model, tokenizer, probes)
    if model_hash(model) != state_before:
        raise RuntimeError("evaluation mutated model state")
    if (trainer.optimizer_step, trainer.tokens_seen) != counters_before:
        raise RuntimeError("evaluation mutated Trainer counters")
    if model.training != mode_before:
        raise RuntimeError("evaluation failed to restore model mode")
    return scores


def train_cell(root: Path, output: Path, source_sha: str, control, index: int, budget: int):
    spec = control["specs"][index]
    tokenizer = control["tok"]
    probes = load_suite(root / SUITE_PATH)
    init = InitSpec()
    random.seed(r41.SEED)
    torch.manual_seed(r41.SEED)
    model = TwelveSixDecoder(spec, init)
    trainer = Trainer(model, control["cfg"], device="cpu")

    random_scores = score_nonmutating(model, trainer, tokenizer, probes)
    token_streams = r41.streams(tokenizer)
    names, offsets = r41.schedule(control["plan"], control["cfg"].max_steps)
    started = time.perf_counter()
    final_loss = None
    while trainer.tokens_seen < budget:
        step = trainer.optimizer_step
        if step >= len(names):
            raise RuntimeError("training schedule exhausted")
        batch = r41.batch(token_streams[names[step]], offsets[step])
        metrics = trainer.train_microbatch({"input_ids": batch})
        if not metrics.optimizer_stepped:
            raise RuntimeError("uncommitted optimization update")
        final_loss = float(metrics.update_loss or metrics.loss)
        if not math.isfinite(final_loss):
            raise RuntimeError("non-finite training loss")
    wall = time.perf_counter() - started
    learned_scores = score_nonmutating(model, trainer, tokenizer, probes)

    checkpoint_identity, _, _ = r41.binding(source_sha, spec, init, control, trainer)
    checkpoint = output / "checkpoints" / str(spec.parameter_count())
    saved = save_trainer_checkpoint(
        checkpoint, model=model, trainer=trainer, identity=checkpoint_identity
    )
    random_summary = summarize(random_scores)
    learned_summary = summarize(learned_scores)
    return {
        "parameters": spec.parameter_count(),
        "model_spec_sha256": spec.identity_sha256(),
        "requested_learned_budget": budget,
        "optimized_tokens": trainer.tokens_seen,
        "optimizer_steps": trainer.optimizer_step,
        "training_wall_seconds": wall,
        "final_train_loss": final_loss,
        "checkpoint": {
            "path": str(checkpoint.relative_to(output)),
            "checkpoint_id": saved["checkpoint_id"],
            "format": saved["format"],
            "format_version": saved["format_version"],
        },
        "random_init": {
            "summary": random_summary,
            "scores": serializable_scores(random_scores),
        },
        "learned": {
            "summary": learned_summary,
            "scores": serializable_scores(learned_scores),
        },
        "delta": {
            "raw_accuracy": learned_summary["overall"]["raw_accuracy"]
            - random_summary["overall"]["raw_accuracy"],
            "byte_normalized_accuracy": learned_summary["overall"]["byte_normalized_accuracy"]
            - random_summary["overall"]["byte_normalized_accuracy"],
            "correct_bpb": learned_summary["overall"]["mean_correct_bits_per_source_byte"]
            - random_summary["overall"]["mean_correct_bits_per_source_byte"],
        },
    }


def run(root: Path, output: Path, source_sha: str, budget: int) -> dict[str, Any]:
    if budget <= 0 or git_head(root) != source_sha:
        raise RuntimeError("invalid budget or source SHA")
    reservation = verify_reservation(root)
    control = r41.control(root)
    probes = load_suite(root / SUITE_PATH)
    byte_tokenizer = ByteTokenizer()
    segmentation = tokenizer_diagnostics(control["tok"], byte_tokenizer, probes)
    models = [
        train_cell(root, output, source_sha, control, index, budget)
        for index in SPEC_INDICES
    ]
    if tuple(row["parameters"] for row in models) != EXPECTED_PARAMETERS:
        raise RuntimeError("selected parameter ladder drift")
    report = {
        "schema": SCHEMA,
        "authority": AUTHORITY,
        "repository": r41.REPO,
        "source_sha": source_sha,
        "suite": reservation,
        "control_identity_sha256": control["identity"],
        "tokenizer": {
            "learned": {
                "algorithm": "bpe",
                "vocab_size": control["tok"].vocab_size,
                "config_sha256": control["tok"].identity.config_sha256,
                "vocab_sha256": control["tok"].identity.vocab_sha256,
            },
            "byte_reference": {
                "algorithm": "utf8-byte",
                "vocab_size": byte_tokenizer.vocab_size,
                "config_sha256": byte_tokenizer.identity.config_sha256,
                "vocab_sha256": byte_tokenizer.identity.vocab_sha256,
            },
            "probe_segmentation": segmentation,
            "interpretation": "Likelihoods use BPE controls; byte tokenization is the segmentation reference. Source-byte NLL/BPB is the tokenizer-length-normalized model metric.",
        },
        "training": {
            "dataset_id": control["data"]["dataset_id"],
            "data_authority": control["data"]["training_authority"],
            "train_snapshot_sha256": control["data"]["train_snapshot_sha256"],
            "learned_budget_requested": budget,
            "seed": r41.SEED,
            "foreign_pretrained_weights_used": False,
            "instruction_or_sft_used": False,
            "paid_compute_used": False,
        },
        "models": models,
        "truth_boundary": {
            "mechanistic_code_modelling_probe": True,
            "memorization_control": "qzv_ identifiers and reserved 7xxx/8xxx literals are absent from training inputs; exact full-candidate overlap is zero",
            "code_generation_capability_claim": False,
            "instruction_following_benchmark": False,
            "representative_code_corpus_claim": False,
            "model_promotion_claim": False,
        },
    }
    report["report_identity_sha256"] = canonical_json_sha256(report)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "eval134-code-scaling-report.json", report)
    return report


def validate_report(path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    identity = report.pop("report_identity_sha256", None)
    if identity != canonical_json_sha256(report):
        raise RuntimeError("report identity mismatch")
    if report.get("schema") != SCHEMA or report.get("authority") != AUTHORITY:
        raise RuntimeError("report schema/authority mismatch")
    if report["suite"]["training_overlap_count"] != 0:
        raise RuntimeError("contaminated diagnostic report")
    if tuple(row["parameters"] for row in report["models"]) != EXPECTED_PARAMETERS:
        raise RuntimeError("scale ladder mismatch")
    for row in report["models"]:
        for phase in ("random_init", "learned"):
            overall = row[phase]["summary"]["overall"]
            if not 0.0 <= overall["raw_accuracy"] <= 1.0:
                raise RuntimeError("invalid accuracy")
            if not 0.0 <= overall["byte_normalized_accuracy"] <= 1.0:
                raise RuntimeError("invalid normalized accuracy")
            if not math.isfinite(overall["mean_correct_bits_per_source_byte"]):
                raise RuntimeError("invalid BPB")
    if report["truth_boundary"]["code_generation_capability_claim"] is not False:
        raise RuntimeError("truth boundary weakened")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=Path("."))
    execute = sub.add_parser("run")
    execute.add_argument("--repo-root", type=Path, default=Path("."))
    execute.add_argument("--source-sha", required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    execute.add_argument("--learned-budget", type=int, default=DEFAULT_BUDGET)
    execute.add_argument("--torch-threads", type=int, default=2)
    validate = sub.add_parser("validate")
    validate.add_argument("report", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_reservation(args.repo_root.resolve()), sort_keys=True))
        return 0
    if args.command == "validate":
        validate_report(args.report)
        print("eval134_report=VALID")
        return 0
    torch.set_num_threads(args.torch_threads)
    report = run(
        args.repo_root.resolve(),
        args.output_dir.resolve(),
        args.source_sha,
        args.learned_budget,
    )
    print(json.dumps({
        "report_identity_sha256": report["report_identity_sha256"],
        "models": [
            {
                "parameters": row["parameters"],
                "random_raw_accuracy": row["random_init"]["summary"]["overall"]["raw_accuracy"],
                "learned_raw_accuracy": row["learned"]["summary"]["overall"]["raw_accuracy"],
                "random_bpb": row["random_init"]["summary"]["overall"]["mean_correct_bits_per_source_byte"],
                "learned_bpb": row["learned"]["summary"]["overall"]["mean_correct_bits_per_source_byte"],
            }
            for row in report["models"]
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
