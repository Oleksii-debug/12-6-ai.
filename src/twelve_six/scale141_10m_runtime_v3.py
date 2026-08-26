"""SCALE-141 authoritative entrypoint with stable identity and launch gating.

The v2 runtime owns actual-token accounting and memory-bounded evaluation. This
entrypoint adds fail-closed corrections found during pre-execution audit:

* dataclass configuration is normalized to JSON-native types before run-manifest
  hashing/comparison, so tuple-valued AdamW fields survive the fresh process;
* the train-vs-heldout gap remains only a generalization proxy, while a separate
  hash-only training-passage continuation probe records memorization progression
  without changing DATA-25 or emitting source text;
* CI-165 requires a SHA/config-bound cheap launch envelope before either long
  training phase can execute;
* CHECKPOINT-211 maps rolling recovery saves onto immutable D05 generations and
  advances only a small atomically replaced pointer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from twelve_six import scale141_10m_runtime_v2 as v2
from twelve_six import scale141_memorization as memorization
from twelve_six.checkpoint import hash_json
from twelve_six.launch_gate import require_launch_envelope_from_env
from twelve_six.scale141_recovery import (
    publish_recovery_generation,
    resolve_recovery_generation,
)

SCHEMA = v2.SCHEMA
_V2_RUN_MANIFEST = v2._run_manifest_v2
_BASE_EVAL_POINT = v2.core._eval_point
_ORIGINAL_SAVE = v2.core._save
_ORIGINAL_CHECKPOINT_PATH = v2._checkpoint_path
_LAUNCH_BINDING = {"workflow": "scale141-10m-learned-continuation", "scale": "10m"}

_PHASE1_BOUNDARY_RECOVERY: dict[str, Any] | None = None
_RESUME_RECOVERY_PATH: Path | None = None


def _json_normalize(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _json_stable_run_manifest(*args, **kwargs) -> dict[str, Any]:
    value = _V2_RUN_MANIFEST(*args, **kwargs)
    value["diagnostics"] = {
        "memorization": {
            "probe_id": memorization.PROBE_ID,
            "type": "hash-only short-continuation recovery/NLL on deterministic project-owned training passages",
            "samples_per_modality": memorization.SAMPLES_PER_MODALITY,
            "continuation_tokens": memorization.WIDTH,
            "text_emitted": False,
            "canary_injection": False,
            "canary_injection_reason": "preserve exact reconstructed DATA-25 corpus identity",
            "privacy_leakage_claim": "NONE",
        }
    }
    value.pop("identity_sha256", None)
    value = _json_normalize(value)
    value["identity_sha256"] = hash_json(value)
    return value


def _eval_point_with_memorization(
    model,
    checkpoint: Path,
    corpus: Path,
    manifest: dict[str, Any],
    tok,
    trainer,
    interval_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    value = _BASE_EVAL_POINT(
        model, checkpoint, corpus, manifest, tok, trainer, interval_rows
    )
    rows_by_modality = {
        modality: list(v2.core._rows(corpus, manifest, "train", modality))
        for modality in ("uk", "en", "code")
    }
    probe = memorization.hashed_training_probe(
        model,
        tok,
        rows_by_modality,
        seed=v2.core.SEED,
        context_tokens=v2.SEQ,
    )
    value["memorization"]["metric_role"] = (
        "TRAIN_HELDOUT_GAP_IS_GENERALIZATION_PROXY_NOT_MEMORIZATION_ALONE"
    )
    value["memorization"]["hash_only_training_passage_probe"] = probe
    value["memorization"]["privacy_leakage_claim"] = "NONE"
    return value


def _publish_recovery_for_save(path: Path, args: tuple[Any, ...]) -> dict[str, Any]:
    if len(args) != 8:
        raise v2.Scale141RuntimeError("unexpected SCALE-141 checkpoint save signature")
    source_sha, _spec, _tok, _manifest, run, _cfg, trainer, _locks = args
    return publish_recovery_generation(
        path.parent / "recovery",
        save_generation=lambda destination: _ORIGINAL_SAVE(destination, *args),
        expected_source_sha=source_sha,
        expected_run_manifest_hash=run["identity_sha256"],
        expected_step=trainer.optimizer_step,
        expected_tokens_seen=trainer.tokens_seen,
    )


def _recovery_aware_save(path: Path, *args) -> dict[str, Any]:
    global _PHASE1_BOUNDARY_RECOVERY
    path = Path(path)
    if path.name == "recovery-latest":
        reference = _publish_recovery_for_save(path, args)
        return {
            "step": reference["optimizer_step"],
            "tokens_seen": reference["tokens_seen"],
            "checkpoint_id": reference["checkpoint_id"],
            "recovery_generation": reference["generation"],
            "pointer_sha256": reference["pointer_sha256"],
        }

    result = _ORIGINAL_SAVE(path, *args)
    resume_boundary_name = _ORIGINAL_CHECKPOINT_PATH(
        path.parent, v2.RESUME_TOKEN_TARGET
    ).name
    if path.name == resume_boundary_name:
        _PHASE1_BOUNDARY_RECOVERY = _publish_recovery_for_save(path, args)
    return result


def _resume_aware_checkpoint_path(out: Path, token_target: int) -> Path:
    if token_target == v2.RESUME_TOKEN_TARGET and _RESUME_RECOVERY_PATH is not None:
        return _RESUME_RECOVERY_PATH
    return _ORIGINAL_CHECKPOINT_PATH(out, token_target)


def _install() -> None:
    # v2._install_runtime_contract resolves _run_manifest_v2 at call time, so
    # replacing it here makes phase1 and the separate resume process install the
    # same JSON-stable builder. _eval_point is not reset by v2 and is patched here.
    v2._run_manifest_v2 = _json_stable_run_manifest
    v2.core._eval_point = _eval_point_with_memorization
    v2.core._save = _recovery_aware_save
    v2._checkpoint_path = _ORIGINAL_CHECKPOINT_PATH


def _require_launch_gate(repo: Path) -> None:
    require_launch_envelope_from_env(repo, expected_binding=_LAUNCH_BINDING)


def _validate_phase1_self_hash(value: dict[str, Any]) -> None:
    supplied = value.get("identity_sha256")
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    if supplied != hash_json(unsigned):
        raise v2.Scale141RuntimeError("phase1 report self-hash mismatch")


def phase1(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    global _PHASE1_BOUNDARY_RECOVERY
    _require_launch_gate(repo)
    _install()
    _PHASE1_BOUNDARY_RECOVERY = None
    value = v2.phase1(repo, source_sha, out)
    reference = _PHASE1_BOUNDARY_RECOVERY
    if reference is None:
        raise v2.Scale141RuntimeError(
            "phase1 reached its boundary without publishing an exact recovery generation"
        )
    if reference["optimizer_step"] != value["optimizer_step"]:
        raise v2.Scale141RuntimeError("phase1 recovery optimizer-step mismatch")
    if reference["tokens_seen"] != value["tokens_seen"]:
        raise v2.Scale141RuntimeError("phase1 recovery optimized-token mismatch")
    value["recovery_resume"] = reference
    value.pop("identity_sha256", None)
    value["identity_sha256"] = hash_json(value)
    v2.core._write_json(out / "phase1.json", value)
    return value


def resume(repo: Path, source_sha: str, out: Path) -> dict[str, Any]:
    global _RESUME_RECOVERY_PATH
    _require_launch_gate(repo)
    _install()
    phase1_report = v2.core._read_json(out / "phase1.json")
    _validate_phase1_self_hash(phase1_report)
    reference = phase1_report.get("recovery_resume")
    if not isinstance(reference, dict):
        raise v2.Scale141RuntimeError("phase1 recovery reference is missing")
    resolution = resolve_recovery_generation(
        out / "recovery",
        expected_reference=reference,
        expected_source_sha=source_sha,
        expected_run_manifest_hash=reference.get("run_manifest_hash"),
        expected_step=phase1_report["optimizer_step"],
        expected_tokens_seen=phase1_report["tokens_seen"],
    )
    _RESUME_RECOVERY_PATH = resolution.path
    v2._checkpoint_path = _resume_aware_checkpoint_path
    try:
        value = v2.resume(repo, source_sha, out)
    finally:
        _RESUME_RECOVERY_PATH = None
        v2._checkpoint_path = _ORIGINAL_CHECKPOINT_PATH

    value["fresh_process_resume"]["recovery_generation"] = reference["generation"]
    value["fresh_process_resume"]["recovery_checkpoint_id"] = reference["checkpoint_id"]
    value["fresh_process_resume"]["recovery_pointer_sha256"] = reference[
        "pointer_sha256"
    ]
    value["fresh_process_resume"]["recovery_resolution"] = "EXACT_PHASE1_REFERENCE"
    value.pop("report_sha256", None)
    value["report_sha256"] = hash_json(value)
    v2.core._write_json(out / "report.json", value)
    return value


def validate(path: Path, expected_source_sha: str | None = None) -> dict[str, Any]:
    report = v2.validate(path, expected_source_sha)
    recovery = report.get("fresh_process_resume", {})
    if recovery.get("recovery_resolution") != "EXACT_PHASE1_REFERENCE":
        raise v2.Scale141RuntimeError("fresh-process recovery generation was not exact")
    for target in v2.EVAL_TOKEN_TARGETS:
        point = report["scheduled"].get(str(target))
        if point is None:
            raise v2.Scale141RuntimeError(
                f"missing scheduled evaluation at optimized-token target {target}"
            )
        probe = point["memorization"].get("hash_only_training_passage_probe")
        if not isinstance(probe, dict):
            raise v2.Scale141RuntimeError("missing hash-only memorization probe")
        if probe.get("text_emitted") is not False:
            raise v2.Scale141RuntimeError("memorization probe emitted source text")
        if probe.get("model_non_mutation_passed") is not True:
            raise v2.Scale141RuntimeError("memorization probe mutation gate failed")
        if probe.get("canary_injection") is not False:
            raise v2.Scale141RuntimeError("SCALE-141 corpus identity was altered by canaries")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("phase1", "resume"):
        p = sub.add_parser(name)
        p.add_argument("--repo-root", type=Path, default=Path("."))
        p.add_argument("--source-sha", required=True)
        p.add_argument("--output-dir", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("report", type=Path)
    p.add_argument("--expected-source-sha")
    args = parser.parse_args()
    if args.command == "phase1":
        value = phase1(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    elif args.command == "resume":
        value = resume(
            args.repo_root.resolve(), args.source_sha, args.output_dir.resolve()
        )
    else:
        value = validate(args.report, args.expected_source_sha)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
