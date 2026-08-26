"""Project detailed VERIFY-218 evidence into the downstream machine authority contract.

The bridge preserves the scientific verifier payload, adds a fresh-process D02/D05
resume proof, independently resolves the retained best/final roles from the exact
producer artifact, and emits the stable fields consumed by RUNTIME-225.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from twelve_six.checkpoint import hash_json
from twelve_six.verify218_learned_10m import (
    EXPECTED_CORPUS_ID,
    EXPECTED_MODEL_SPEC_SHA256,
    EXPECTED_PARAMETER_COUNT,
    PRODUCER_ARTIFACT_ID,
    PRODUCER_ARTIFACT_ZIP_SHA256,
    PRODUCER_SHA,
    STATE,
    WORKER,
)

PRODUCER_ARTIFACT_NAME = "learn217-terminal-10m-learned-base"
PRODUCER_WORKFLOW_RUN_ID = 32_952_787_070


class Verify218BridgeError(RuntimeError):
    """Fail-closed downstream-authority projection error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Verify218BridgeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Verify218BridgeError(f"cannot read JSON object {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _verify_detailed_authority(value: Mapping[str, Any]) -> None:
    supplied = value.get("identity_sha256")
    unsigned = dict(value)
    unsigned.pop("identity_sha256", None)
    _require(isinstance(supplied, str), "detailed VERIFY-218 identity hash missing")
    _require(hash_json(unsigned) == supplied, "detailed VERIFY-218 identity hash mismatch")
    _require(value.get("worker") == WORKER, "detailed verifier worker mismatch")
    _require(value.get("state") == STATE, "detailed verifier state mismatch")
    boundaries = value.get("boundaries")
    _require(isinstance(boundaries, Mapping), "detailed verifier boundaries missing")
    _require(boundaries.get("training_executed") is False, "verifier executed training")
    _require(boundaries.get("optimizer_updates") == 0, "verifier performed optimizer updates")
    _require(
        boundaries.get("foreign_pretrained_weights") is False,
        "verifier admitted foreign/pretrained weights",
    )
    _require(boundaries.get("evaluation_mutated_model") is False, "evaluation mutated model")


def _resolve_roles(artifact_root: Path, detailed: Mapping[str, Any]) -> dict[str, Any]:
    evidence = artifact_root / "scale141-evidence"
    fresh = _read_json(evidence / "fresh-verification.json")
    retained = _read_json(evidence / "retained" / "index.json")
    ladder = fresh.get("ladder_common_evaluation")
    _require(isinstance(ladder, Mapping), "producer ladder evidence missing")
    scheduled = ladder.get("all_scheduled")
    _require(isinstance(scheduled, Mapping) and scheduled, "producer scheduled evidence missing")

    candidates: list[tuple[float, int]] = []
    for raw_target, payload in scheduled.items():
        try:
            target = int(raw_target)
        except (TypeError, ValueError) as exc:
            raise Verify218BridgeError("producer scheduled target is not an integer") from exc
        if target <= 0:
            continue
        _require(isinstance(payload, Mapping), f"scheduled evidence {target} missing")
        evaluation = payload.get("evaluation")
        _require(isinstance(evaluation, Mapping), f"scheduled evaluation {target} missing")
        try:
            bpb = float(evaluation["bits_per_byte"])
        except (KeyError, TypeError, ValueError) as exc:
            raise Verify218BridgeError(f"scheduled BPB {target} is invalid") from exc
        candidates.append((bpb, target))
    _require(bool(candidates), "producer has no learned scheduled evaluations")
    resolved_best_target = min(candidates, key=lambda item: (item[0], item[1]))[1]
    resolved_final_target = max(target for _, target in candidates)

    roles = retained.get("roles")
    _require(isinstance(roles, Mapping), "retained roles missing")
    best_index = roles.get("best")
    final_index = roles.get("final")
    _require(isinstance(best_index, Mapping), "retained best role missing")
    _require(isinstance(final_index, Mapping), "retained final role missing")
    _require(
        int(best_index.get("target_optimized_tokens", -1)) == resolved_best_target,
        "retained best role does not resolve from scheduled evidence",
    )
    _require(
        int(final_index.get("target_optimized_tokens", -1)) == resolved_final_target,
        "retained final role does not resolve to final scheduled target",
    )

    checkpoints = detailed.get("checkpoints")
    _require(isinstance(checkpoints, Mapping), "detailed checkpoints missing")
    best = checkpoints.get("best")
    final = checkpoints.get("final")
    _require(isinstance(best, Mapping), "detailed best checkpoint missing")
    _require(isinstance(final, Mapping), "detailed final checkpoint missing")
    _require(
        best.get("checkpoint_id") == best_index.get("checkpoint_id"),
        "best checkpoint ID differs from retained index",
    )
    _require(
        final.get("checkpoint_id") == final_index.get("checkpoint_id"),
        "final checkpoint ID differs from retained index",
    )
    return {
        "best_target_optimized_tokens": resolved_best_target,
        "final_target_optimized_tokens": resolved_final_target,
        "best_checkpoint_id": best["checkpoint_id"],
        "final_checkpoint_id": final["checkpoint_id"],
    }


def _fresh_process_resume(checkpoint: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.verify218_resume_probe",
            "--checkpoint",
            str(checkpoint.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise Verify218BridgeError("fresh-process recovery probe returned invalid JSON") from exc
    _require(isinstance(value, dict), "fresh-process recovery probe must return an object")
    _require(value.get("pid") != os.getpid(), "recovery probe did not use a fresh process")
    _require(value.get("training_executed") is False, "recovery probe executed training")
    _require(value.get("optimizer_updates") == 0, "recovery probe performed optimizer update")
    _require(
        value.get("checkpoint_safe_after_restore") is True,
        "fresh-process trainer is not checkpoint-safe after restore",
    )
    optimizer = value.get("optimizer_state")
    _require(isinstance(optimizer, Mapping), "fresh-process optimizer metadata missing")
    _require(
        int(optimizer.get("populated_parameters", 0)) > 0,
        "fresh-process optimizer state is empty",
    )
    _require(
        int(optimizer.get("tensor_leaves", 0)) > 0,
        "fresh-process optimizer tensor state is empty",
    )
    return value


def bridge(*, detailed_path: Path, artifact_root: Path, output_path: Path) -> dict[str, Any]:
    detailed = _read_json(detailed_path.resolve())
    _verify_detailed_authority(detailed)
    artifact_root = artifact_root.resolve()
    role_resolution = _resolve_roles(artifact_root, detailed)
    resume = _fresh_process_resume(
        artifact_root / "scale141-evidence" / "retained" / "recovery-current"
    )

    model = detailed.get("model")
    data = detailed.get("data_and_eval")
    checkpoints = detailed.get("checkpoints")
    _require(isinstance(model, Mapping), "detailed model evidence missing")
    _require(isinstance(data, Mapping), "detailed data/evaluation evidence missing")
    _require(isinstance(checkpoints, Mapping), "detailed checkpoint evidence missing")
    best = checkpoints.get("best")
    _require(isinstance(best, Mapping), "detailed best checkpoint missing")
    identity = best.get("identity")
    _require(isinstance(identity, Mapping), "detailed best checkpoint identity missing")

    _require(model.get("model_spec_sha256") == EXPECTED_MODEL_SPEC_SHA256, "ModelSpec mismatch")
    _require(model.get("parameter_count") == EXPECTED_PARAMETER_COUNT, "parameter count mismatch")
    _require(data.get("corpus_identity_sha256") == EXPECTED_CORPUS_ID, "corpus mismatch")
    _require(
        data.get("best_improved_over_reconstructed_random_init") is True,
        "best checkpoint failed learned-improvement gate",
    )
    _require(
        data.get("final_improved_over_reconstructed_random_init") is True,
        "final checkpoint failed learned-improvement gate",
    )

    output = dict(detailed)
    output.update(
        {
            "worker_id": WORKER,
            "status": STATE,
            "verified_learned_10m": True,
            "foreign_pretrained_weights": False,
            "mechanics_only_checkpoint": False,
            "one_step_smoke": False,
            "tokenizer": {
                "version": model["tokenizer_version"],
                "config_sha256": model["tokenizer_config_sha256"],
                "vocab_sha256": model["tokenizer_vocab_sha256"],
            },
            "corpus_identity_sha256": data["corpus_identity_sha256"],
            "source": {
                "artifact_id": PRODUCER_ARTIFACT_ID,
                "artifact_name": PRODUCER_ARTIFACT_NAME,
                "artifact_digest": f"sha256:{PRODUCER_ARTIFACT_ZIP_SHA256}",
                "workflow_run_id": PRODUCER_WORKFLOW_RUN_ID,
                "source_sha": PRODUCER_SHA,
            },
            "checkpoint": {
                "role": "best",
                "checkpoint_id": best["checkpoint_id"],
                "step": int(identity["step"]),
                "tokens_seen": int(identity["tokens_seen"]),
            },
            "gates": {
                "checkpoint_integrity": True,
                "fresh_process_resume": True,
                "finite_first_party_logits": True,
                "heldout_bpb": True,
                "evaluation_non_mutation": True,
                "greedy_generation": True,
                "best_final_role_resolution": True,
            },
            "fresh_process_resume": resume,
            "role_resolution": role_resolution,
        }
    )
    output.pop("identity_sha256", None)
    output["identity_sha256"] = hash_json(output)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = bridge(
        detailed_path=args.detailed,
        artifact_root=args.artifact_root,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "worker_id": result["worker_id"],
                "status": result["status"],
                "identity_sha256": result["identity_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
