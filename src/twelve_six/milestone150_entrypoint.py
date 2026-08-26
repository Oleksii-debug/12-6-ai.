"""Fresh-process-safe entrypoint for MILESTONE-150.

The core ladder manifest is hash-canonical through JSON, but TrainerConfig contains
Python tuples (notably AdamW betas). JSON persistence turns those tuples into
lists. A fresh process therefore compares the same JSON data model rather than
Python container implementation details.

This entrypoint also strengthens retained-checkpoint verification without changing
the ladder architecture: immediately before the incumbent verifier runs it
reconstructs the complete run manifest from the live source/config and requires
exact equality with the persisted self-hashed manifest. If best and final are the
same physical checkpoint, their two recorded generation roles must also be
identical before fresh generation is accepted transitively for both roles.

No field or identity check is weakened.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from twelve_six import milestone150_learned_base_ladder as ladder


_ORIGINAL_RUN_MANIFEST = ladder._run_manifest
_ORIGINAL_VERIFY_SCALE = ladder.verify_scale


def json_normalize(value: Any) -> Any:
    """Return the canonical JSON data-model representation of ``value``."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def normalized_run_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Build the incumbent run manifest and normalize only its representation."""
    value = _ORIGINAL_RUN_MANIFEST(*args, **kwargs)
    normalized = json_normalize(value)
    if not isinstance(normalized, dict):
        raise TypeError("MILESTONE-150 run manifest must normalize to an object")
    if normalized.get("identity_sha256") != value.get("identity_sha256"):
        raise RuntimeError("JSON normalization changed run-manifest identity")
    return normalized


def require_fresh_manifest_match(
    persisted: dict[str, Any], expected: dict[str, Any]
) -> None:
    """Fail closed unless persisted and freshly reconstructed manifests are exact."""
    ladder._check_self_hash(persisted)
    ladder._check_self_hash(expected)
    if persisted != expected:
        raise ladder.LadderError("fresh run-manifest reconstruction mismatch")


def require_shared_checkpoint_generation_snapshot(report: dict[str, Any]) -> None:
    """When best == final checkpoint, require both retained generation roles to match."""
    best_step = int(report["evaluation"]["best_step"])
    if best_step != ladder.MAX_STEPS:
        return
    generation = report["generation"]
    if generation["best_checkpoint"] != generation["final_checkpoint"]:
        raise ladder.LadderError(
            "best/final generation snapshots diverge for the same retained checkpoint"
        )


def validate_fresh_reproducibility_contract(
    repo: Path, source_sha: str, out: Path, scale: str
) -> dict[str, Any]:
    """Reconstruct reproducibility truth immediately before retained verification."""
    manifest, tok, eval_id = ladder._common_truth(repo, source_sha, out, build=False)
    spec = ladder.model_spec(scale)
    init = ladder.init_spec()
    cfg = ladder.trainer_config()
    locks = ladder.m100._locks(repo)
    expected = normalized_run_manifest(
        source_sha,
        scale,
        spec,
        init,
        tok,
        manifest,
        eval_id,
        cfg,
        locks,
    )
    persisted = ladder._read_json(out / scale / "run-manifest.json")
    require_fresh_manifest_match(persisted, expected)

    preverify = ladder._read_json(out / scale / "report.preverify.json")
    ladder._check_self_hash(preverify)
    require_shared_checkpoint_generation_snapshot(preverify)
    return {
        "run_manifest_identity_sha256": expected["identity_sha256"],
        "fresh_reconstruction_match": True,
        "shared_best_final_generation_role_checked": int(
            preverify["evaluation"]["best_step"]
        )
        == ladder.MAX_STEPS,
    }


def verified_scale(repo: Path, source_sha: str, out: Path, scale: str) -> dict[str, Any]:
    """Run fresh reproducibility gates, then the incumbent retained verifier."""
    validate_fresh_reproducibility_contract(repo, source_sha, out, scale)
    return _ORIGINAL_VERIFY_SCALE(repo, source_sha, out, scale)


def install_fresh_process_manifest_normalization() -> None:
    """Install representation and verification compatibility repairs for this process."""
    ladder._run_manifest = normalized_run_manifest
    ladder.verify_scale = verified_scale


def main(argv: list[str] | None = None) -> int:
    install_fresh_process_manifest_normalization()
    return ladder.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
