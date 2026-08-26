from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from twelve_six.model import InitSpec, ModelSpec, TwelveSixDecoder
from twelve_six.tokenization import ByteTokenizer

from .contracts import GenerationConfig, GenerationResult
from .first_party import FirstPartyInferenceBackend, load_first_party_backend
from .generation import generate
from .torch_backend import TorchInferenceBackend

TWENTY_M_TARGET_PARAMETERS = 20_000_000
TWENTY_M_MIN_PARAMETERS = 18_000_000
TWENTY_M_MAX_PARAMETERS = 22_000_000

InferenceSource = Literal["random_init", "checkpoint"]


def validate_20m_spec(spec: ModelSpec) -> ModelSpec:
    """Fail closed when a runtime is accidentally pointed at a non-20M model."""

    parameters = spec.parameter_count()
    if not TWENTY_M_MIN_PARAMETERS <= parameters <= TWENTY_M_MAX_PARAMETERS:
        raise ValueError(
            "ModelSpec is outside the maintained ~20M inference band: "
            f"parameters={parameters} expected=[{TWENTY_M_MIN_PARAMETERS}, "
            f"{TWENTY_M_MAX_PARAMETERS}]"
        )
    return spec


def load_20m_model_spec(path: str | Path) -> ModelSpec:
    """Load a ModelSpec from a raw spec or a containing project JSON artifact."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("ModelSpec JSON root must be an object")

    raw_spec: object
    if "model_spec" in payload:
        raw_spec = payload["model_spec"]
    elif "model" in payload:
        raw_spec = payload["model"]
    else:
        raw_spec = payload
    if not isinstance(raw_spec, Mapping):
        raise ValueError("ModelSpec JSON must contain an object-valued model/model_spec")

    try:
        spec = ModelSpec.from_dict(dict(raw_spec))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ModelSpec JSON: {exc}") from exc
    return validate_20m_spec(spec)


@dataclass(slots=True)
class TwentyMInference:
    """Raw-completion library API for the maintained first-party ~20M path."""

    backend: TorchInferenceBackend
    source: InferenceSource
    init_seed: int | None = None

    @property
    def model_spec(self) -> ModelSpec:
        return self.backend.model.spec

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        return generate(self.backend, prompt, config)

    def diagnostics(self) -> dict[str, object]:
        backend_diagnostics = getattr(self.backend, "diagnostics", None)
        if callable(backend_diagnostics):
            payload = dict(backend_diagnostics())
        else:
            payload = {
                "backend": "first_party_torch",
                "model_spec_sha256": self.model_spec.identity_sha256(),
                "parameter_count": self.model_spec.parameter_count(),
                "vocab_size": self.model_spec.vocab_size,
                "max_context_tokens": self.backend.max_context_tokens,
                "device": str(next(self.backend.model.parameters()).device),
            }
        payload.update(
            {
                "model_family": "20m",
                "source": self.source,
                "learned_weights": self.source == "checkpoint",
                "init_seed": self.init_seed,
            }
        )
        return payload

    @classmethod
    def from_random_init(
        cls,
        spec: ModelSpec,
        *,
        init_spec: InitSpec | None = None,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> "TwentyMInference":
        validate_20m_spec(spec)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")

        tokenizer = ByteTokenizer()
        if spec.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "20M random-init ModelSpec vocabulary must match the canonical "
                f"first-party tokenizer: model={spec.vocab_size} "
                f"tokenizer={tokenizer.vocab_size}"
            )

        # Initialize on CPU under a forked RNG so construction is deterministic
        # without mutating the caller's global RNG stream. Device transfer is
        # intentionally separate from parameter initialization.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = TwelveSixDecoder(spec, init_spec=init_spec)
        model.to(torch.device(device))
        model.eval()
        return cls(
            backend=TorchInferenceBackend(model, tokenizer),
            source="random_init",
            init_seed=seed,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> "TwentyMInference":
        backend: FirstPartyInferenceBackend = load_first_party_backend(
            Path(checkpoint),
            device=device,
            spec_validator=validate_20m_spec,
        )
        return cls(backend=backend, source="checkpoint", init_seed=None)


def open_20m_inference(
    *,
    checkpoint: str | Path | None = None,
    model_spec: ModelSpec | None = None,
    init_spec: InitSpec | None = None,
    init_seed: int = 0,
    device: str | torch.device = "cpu",
) -> TwentyMInference:
    """Open exactly one learned-checkpoint or random-init 20M inference source."""

    if (checkpoint is None) == (model_spec is None):
        raise ValueError("provide exactly one of checkpoint or model_spec")
    if checkpoint is not None:
        if init_spec is not None:
            raise ValueError("init_spec is only valid for random-init inference")
        return TwentyMInference.from_checkpoint(checkpoint, device=device)
    assert model_spec is not None
    return TwentyMInference.from_random_init(
        model_spec,
        init_spec=init_spec,
        seed=init_seed,
        device=device,
    )
