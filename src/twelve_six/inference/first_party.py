"""Verified first-party checkpoint adapter for ModelSpec-driven inference."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from twelve_six.checkpoint import (
    CheckpointCompatibilityError,
    load_verified_checkpoint,
    prepare_checkpoint_load,
)
from twelve_six.model import ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

from .torch_backend import TorchInferenceBackend


class FirstPartyInferenceBackend(TorchInferenceBackend):
    """Verified D01+D04+D05 composition exposed through the generic protocol."""

    def __init__(
        self,
        model: TwelveSixDecoder,
        tokenizer: ByteTokenizer,
        *,
        manifest: Mapping[str, Any],
        checkpoint_path: Path,
    ) -> None:
        super().__init__(model, tokenizer)
        self.manifest = dict(manifest)
        self.checkpoint_path = checkpoint_path

    def diagnostics(self) -> dict[str, object]:
        """Return privacy-safe checkpoint and runtime identities for evidence."""

        identity = self.manifest["identity"]
        return {
            "backend": "first_party_torch",
            "source": "checkpoint",
            "checkpoint_id": self.manifest["checkpoint_id"],
            "git_sha": identity["git_sha"],
            "model_spec_sha256": identity["model_spec_hash"],
            "parameter_count": identity["parameter_count"],
            "vocab_size": self.model.spec.vocab_size,
            "max_context_tokens": self.max_context_tokens,
            "tokenizer_version": self.tokenizer.identity.version,
            "tokenizer_config_sha256": identity["tokenizer_hash"],
            "tokenizer_vocab_sha256": identity["tokenizer_vocab_hash"],
            "dataset_manifest_sha256": identity["dataset_manifest_hash"],
            "run_manifest_sha256": identity["run_manifest_hash"],
            "step": identity["step"],
            "tokens_seen": identity["tokens_seen"],
            "device": str(next(self.model.parameters()).device),
        }


def _checkpoint_spec(manifest: Mapping[str, Any]) -> ModelSpec:
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise CheckpointCompatibilityError("checkpoint identity is missing")
    raw_spec = identity.get("model_spec")
    if not isinstance(raw_spec, Mapping):
        raise CheckpointCompatibilityError("checkpoint ModelSpec is missing")
    try:
        spec = ModelSpec.from_dict(dict(raw_spec))
    except (TypeError, ValueError) as exc:
        raise CheckpointCompatibilityError(f"checkpoint ModelSpec is invalid: {exc}") from exc

    if spec.identity_sha256() != identity.get("model_spec_hash"):
        raise CheckpointCompatibilityError("checkpoint ModelSpec semantic identity mismatch")
    if spec.parameter_count() != identity.get("parameter_count"):
        raise CheckpointCompatibilityError("checkpoint parameter count does not match ModelSpec")

    training_config = identity.get("training_config")
    if isinstance(training_config, Mapping):
        training = training_config.get("training")
        if isinstance(training, Mapping):
            declared_context = training.get("context_length")
            if declared_context is not None and declared_context != spec.max_seq_len:
                raise CheckpointCompatibilityError(
                    "checkpoint training context_length does not match ModelSpec max_seq_len"
                )
    return spec


def _require_byte_tokenizer(manifest: Mapping[str, Any], spec: ModelSpec) -> ByteTokenizer:
    tokenizer = ByteTokenizer()
    identity = manifest["identity"]
    expected = tokenizer.identity
    if identity.get("tokenizer_hash") != expected.config_sha256:
        raise CheckpointCompatibilityError(
            "checkpoint tokenizer config is incompatible with canonical s0-byte-v1"
        )
    if identity.get("tokenizer_vocab_hash") != expected.vocab_sha256:
        raise CheckpointCompatibilityError(
            "checkpoint tokenizer vocabulary is incompatible with canonical s0-byte-v1"
        )
    if spec.vocab_size != expected.vocab_size:
        raise CheckpointCompatibilityError(
            "checkpoint ModelSpec vocabulary size does not match canonical tokenizer"
        )

    training_config = identity.get("training_config")
    if isinstance(training_config, Mapping):
        data = training_config.get("data")
        if isinstance(data, Mapping):
            declared_version = data.get("tokenizer_version")
            if declared_version is not None and declared_version != expected.version:
                raise CheckpointCompatibilityError(
                    "checkpoint tokenizer version does not match canonical s0-byte-v1"
                )
    return tokenizer


def load_first_party_backend(
    checkpoint: Path,
    *,
    device: str | torch.device = "cpu",
    spec_validator: Callable[[ModelSpec], object] | None = None,
) -> FirstPartyInferenceBackend:
    """Verify one checkpoint snapshot, bind ModelSpec/tokenizer, and expose inference.

    The checkpoint directory is read exactly once into D05's immutable
    ``VerifiedCheckpoint`` snapshot. ModelSpec/tokenizer validation, applied
    weights, and backend diagnostics therefore derive from the same bytes.
    An optional ModelSpec validator runs before model allocation. RNG state is
    intentionally not restored for inference.
    """

    checkpoint = Path(checkpoint)
    verified = prepare_checkpoint_load(checkpoint)
    manifest = verified.manifest
    spec = _checkpoint_spec(manifest)
    if spec_validator is not None:
        spec_validator(spec)
    tokenizer = _require_byte_tokenizer(manifest, spec)

    model = TwelveSixDecoder(spec)
    loaded = load_verified_checkpoint(
        verified,
        model=model,
        restore_rng=False,
        expected_model_spec_hash=spec.identity_sha256(),
        expected_tokenizer_hash=tokenizer.identity.config_sha256,
        expected_tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
    )
    model.to(torch.device(device))
    model.eval()
    return FirstPartyInferenceBackend(
        model,
        tokenizer,
        manifest=loaded.manifest,
        checkpoint_path=checkpoint,
    )
