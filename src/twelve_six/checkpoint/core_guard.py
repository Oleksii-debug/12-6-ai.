"""Install D05 semantic guards on direct ``checkpoint.core`` load entry points.

``core.py`` remains the serialization/immutable-byte authority.  The public
package API layers semantic compatibility preflight on top.  This installer
ensures callers that import the historical core load functions directly cannot
bypass that preflight while the checks are being folded into core proper.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import core, hardening


def install_core_guards() -> None:
    if getattr(core, "_D05_SEMANTIC_GUARDS_INSTALLED", False):
        return

    original_load_verified = core.load_verified_checkpoint

    def guarded_load_verified_checkpoint(
        verified: core.VerifiedCheckpoint,
        *,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        strict_model: bool = True,
        restore_rng: bool = True,
        expected_git_sha: str | None = None,
        expected_model_spec_hash: str | None = None,
        expected_tokenizer_hash: str | None = None,
        expected_tokenizer_vocab_hash: str | None = None,
        expected_dataset_manifest_hash: str | None = None,
        expected_run_manifest_hash: str | None = None,
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
    ) -> core.LoadResult:
        manifest = verified.manifest
        hardening._validate_loaded_identity(manifest)
        identity = manifest["identity"]
        expectations = {
            "step": expected_step,
            "tokens_seen": expected_tokens_seen,
        }
        mismatches = {
            key: {"expected": expected, "actual": identity.get(key)}
            for key, expected in expectations.items()
            if expected is not None and identity.get(key) != expected
        }
        if mismatches:
            raise core.CheckpointCompatibilityError(
                f"checkpoint counter identity mismatch: {mismatches}"
            )

        hardening._decode_and_preflight(
            verified,
            model=model,
            optimizer=optimizer,
            strict_model=strict_model,
        )
        return original_load_verified(
            verified,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict_model=strict_model,
            restore_rng=restore_rng,
            expected_git_sha=expected_git_sha,
            expected_model_spec_hash=expected_model_spec_hash,
            expected_tokenizer_hash=expected_tokenizer_hash,
            expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
            expected_dataset_manifest_hash=expected_dataset_manifest_hash,
            expected_run_manifest_hash=expected_run_manifest_hash,
        )

    def guarded_load_checkpoint(
        directory: str | Path,
        *,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        strict_model: bool = True,
        restore_rng: bool = True,
        expected_git_sha: str | None = None,
        expected_model_spec_hash: str | None = None,
        expected_tokenizer_hash: str | None = None,
        expected_tokenizer_vocab_hash: str | None = None,
        expected_dataset_manifest_hash: str | None = None,
        expected_run_manifest_hash: str | None = None,
        expected_step: int | None = None,
        expected_tokens_seen: int | None = None,
    ) -> core.LoadResult:
        verified = hardening.prepare_checkpoint_load(directory)
        return guarded_load_verified_checkpoint(
            verified,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict_model=strict_model,
            restore_rng=restore_rng,
            expected_git_sha=expected_git_sha,
            expected_model_spec_hash=expected_model_spec_hash,
            expected_tokenizer_hash=expected_tokenizer_hash,
            expected_tokenizer_vocab_hash=expected_tokenizer_vocab_hash,
            expected_dataset_manifest_hash=expected_dataset_manifest_hash,
            expected_run_manifest_hash=expected_run_manifest_hash,
            expected_step=expected_step,
            expected_tokens_seen=expected_tokens_seen,
        )

    core.load_verified_checkpoint = guarded_load_verified_checkpoint
    core.load_checkpoint = guarded_load_checkpoint
    core.verify_checkpoint = hardening.verify_checkpoint
    core._D05_SEMANTIC_GUARDS_INSTALLED = True
