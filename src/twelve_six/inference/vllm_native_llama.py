"""Native vLLM execution adapter over the incumbent standard-Llama export.

RUNTIME-25 deliberately owns no second tensor converter or vLLM model class. The
incumbent D07 runtime export materializes exact standard Llama bytes; this module
verifies that payload, binds it back to a 12-6 ModelSpec, and lets vLLM use its
maintained Llama implementation, scheduler, attention kernels and KV cache.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from twelve_six.model import ModelSpec
from twelve_six.tokenization import ByteTokenizer

from .first_party import load_first_party_backend
from .llama_runtime_export import (
    RUNTIME_CONFIG_NAME,
    RUNTIME_PROVENANCE_NAME,
    materialize_standard_llama_directory,
    verify_standard_llama_directory,
)
from .parity import compare_backends
from .sampling import greedy_token
from .transformers_llama import llama_config_dict

EVIDENCE_SCHEMA = "12-6.vllm-native-llama-runtime-parity.v1"
TARGET_ARCHITECTURE = "LlamaForCausalLM"
TARGET_MODEL_TYPE = "llama"


class VllmRuntimeError(ValueError):
    """Raised when standard-Llama/vLLM runtime evidence is invalid."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise VllmRuntimeError(f"required JSON is not a regular file: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmRuntimeError(f"invalid UTF-8 JSON in {path.name}") from exc
    if not isinstance(value, dict):
        raise VllmRuntimeError(f"JSON root must be an object: {path.name}")
    return value


def _required_int(config: Mapping[str, Any], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise VllmRuntimeError(f"standard Llama config requires integer {name}")
    return value


def _required_float(config: Mapping[str, Any], name: str) -> float:
    value = config.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise VllmRuntimeError(f"standard Llama config requires numeric {name}")
    return float(value)


def _spec_from_standard_llama(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> ModelSpec:
    if config.get("model_type") != TARGET_MODEL_TYPE:
        raise VllmRuntimeError("standard Llama config model_type mismatch")
    if config.get("architectures") != [TARGET_ARCHITECTURE]:
        raise VllmRuntimeError("standard Llama config architecture mismatch")

    rope_parameters = config.get("rope_parameters")
    if not isinstance(rope_parameters, Mapping):
        raise VllmRuntimeError("standard Llama config is missing rope_parameters")
    if rope_parameters.get("rope_type") != "default":
        raise VllmRuntimeError("vLLM adapter accepts only default RoPE")
    rope_theta = rope_parameters.get("rope_theta")
    if not isinstance(rope_theta, (int, float)) or isinstance(rope_theta, bool):
        raise VllmRuntimeError("standard Llama config has invalid rope_theta")

    spec = ModelSpec(
        schema_version=1,
        vocab_size=_required_int(config, "vocab_size"),
        max_seq_len=_required_int(config, "max_position_embeddings"),
        d_model=_required_int(config, "hidden_size"),
        n_layers=_required_int(config, "num_hidden_layers"),
        n_heads=_required_int(config, "num_attention_heads"),
        n_kv_heads=_required_int(config, "num_key_value_heads"),
        head_dim=_required_int(config, "head_dim"),
        d_ff=_required_int(config, "intermediate_size"),
        norm_eps=_required_float(config, "rms_norm_eps"),
        rope_theta=float(rope_theta),
        rope_rotary_dim=_required_int(config, "head_dim"),
        attention_bias=bool(config.get("attention_bias", False)),
        mlp_bias=bool(config.get("mlp_bias", False)),
        attention_dropout=_required_float(config, "attention_dropout"),
        final_norm=True,
        tie_word_embeddings=bool(config.get("tie_word_embeddings", False)),
        lm_head_bias=False,
    )
    if dict(config) != llama_config_dict(spec):
        raise VllmRuntimeError(
            "standard Llama config cannot be reconstructed as the exact D07 ModelSpec mapping"
        )
    if spec.identity_sha256() != provenance.get("model_spec_sha256"):
        raise VllmRuntimeError("standard Llama provenance ModelSpec hash mismatch")
    if spec.parameter_count() != provenance.get("parameter_count"):
        raise VllmRuntimeError("standard Llama provenance parameter count mismatch")
    return spec


def materialize_vllm_llama_directory(
    source_export_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Delegate runtime-byte ownership to the incumbent D07 Llama exporter."""

    return materialize_standard_llama_directory(source_export_dir, output_dir)


def verify_vllm_llama_directory(model_dir: str | Path) -> dict[str, Any]:
    """Verify incumbent runtime bytes and expose the vLLM execution binding."""

    root = Path(model_dir)
    try:
        runtime_provenance = verify_standard_llama_directory(root)
    except (OSError, ValueError) as exc:
        raise VllmRuntimeError(f"invalid standard Llama runtime export: {exc}") from exc
    config = _load_json_object(root / RUNTIME_CONFIG_NAME)
    spec = _spec_from_standard_llama(config, runtime_provenance)

    return {
        "schema": "12-6.vllm-native-llama-binding.v1",
        "checkpoint_id": runtime_provenance["source_checkpoint_id"],
        "source_model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "source_export": {
            "source_manifest_sha256": runtime_provenance["source_manifest_sha256"],
            "source_weights_sha256": runtime_provenance["source_weights_sha256"],
            "source_config_sha256": runtime_provenance["source_config_sha256"],
        },
        "target": {
            "architecture": TARGET_ARCHITECTURE,
            "model_type": TARGET_MODEL_TYPE,
            "config_sha256": runtime_provenance["runtime_config_sha256"],
            "weights_sha256": runtime_provenance["runtime_weights_sha256"],
        },
        "runtime_provenance_file": RUNTIME_PROVENANCE_NAME,
        "execution_contract": {
            "vllm_implementation": "BUILTIN_LLAMA",
            "skip_tokenizer_init": True,
            "prompt_input": "TOKEN_IDS",
            "trust_remote_code": False,
        },
    }


@dataclass(frozen=True, slots=True)
class VllmImportProbe:
    vllm_version: str
    llama_registered: bool
    architecture: str
    model_type: str
    max_model_len: int
    skip_tokenizer_init: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def probe_vllm_import_and_config(
    model_dir: str | Path,
    *,
    expected_vllm_version: str | None = None,
) -> VllmImportProbe:
    """Import vLLM, prove built-in Llama registration, and construct ModelConfig."""

    binding = verify_vllm_llama_directory(model_dir)
    spec = ModelSpec.from_dict(dict(binding["source_model_spec"]))

    try:
        import vllm
        from vllm import ModelRegistry
        from vllm.engine.arg_utils import EngineArgs
    except ImportError as exc:
        raise VllmRuntimeError("vLLM is not installed in this environment") from exc

    version = getattr(vllm, "__version__", None)
    if not isinstance(version, str) or not version:
        raise VllmRuntimeError("installed vLLM version is unavailable")
    if expected_vllm_version is not None and version != expected_vllm_version:
        raise VllmRuntimeError(
            f"vLLM version mismatch: expected {expected_vllm_version}, got {version}"
        )

    supported = set(ModelRegistry.get_supported_archs())
    if TARGET_ARCHITECTURE not in supported:
        raise VllmRuntimeError(f"{TARGET_ARCHITECTURE} is not registered in installed vLLM")

    args = EngineArgs(
        model=str(Path(model_dir).resolve()),
        skip_tokenizer_init=True,
        trust_remote_code=False,
        dtype="float32",
        max_model_len=spec.max_seq_len,
        max_logprobs=-1,
        logprobs_mode="raw_logits",
    )
    config = args.create_model_config()
    architecture = getattr(config, "architecture", None)
    if architecture != TARGET_ARCHITECTURE:
        raise VllmRuntimeError(
            f"vLLM resolved unexpected architecture: {architecture!r}"
        )
    if getattr(config, "max_model_len", None) != spec.max_seq_len:
        raise VllmRuntimeError("vLLM ModelConfig context length mismatch")

    return VllmImportProbe(
        vllm_version=version,
        llama_registered=True,
        architecture=architecture,
        model_type=TARGET_MODEL_TYPE,
        max_model_len=spec.max_seq_len,
        skip_tokenizer_init=True,
    )


class VllmNativeLlamaBackend:
    """D07 backend backed by vLLM's maintained Llama implementation."""

    eos_token_id: int | None = None

    def __init__(
        self,
        model_dir: str | Path,
        *,
        tokenizer: Any | None = None,
        dtype: str = "float32",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = True,
    ) -> None:
        binding = verify_vllm_llama_directory(model_dir)
        self.spec = ModelSpec.from_dict(dict(binding["source_model_spec"]))
        self.max_context_tokens = self.spec.max_seq_len

        if tokenizer is None:
            if self.spec.vocab_size != 256:
                raise VllmRuntimeError(
                    "non-byte-vocabulary vLLM execution requires the canonical tokenizer explicitly"
                )
            tokenizer = ByteTokenizer()
        if getattr(tokenizer, "vocab_size", None) != self.spec.vocab_size:
            raise VllmRuntimeError("canonical tokenizer vocabulary does not match ModelSpec")
        if not callable(getattr(tokenizer, "encode", None)) or not callable(
            getattr(tokenizer, "decode", None)
        ):
            raise VllmRuntimeError("canonical tokenizer must provide encode/decode")
        self.tokenizer = tokenizer

        try:
            import vllm
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise VllmRuntimeError("vLLM is not installed in this environment") from exc

        version = getattr(vllm, "__version__", None)
        if not isinstance(version, str) or not version:
            raise VllmRuntimeError("installed vLLM version is unavailable")
        self.vllm_version = version
        self._SamplingParams = SamplingParams
        self._engine = LLM(
            model=str(Path(model_dir).resolve()),
            skip_tokenizer_init=True,
            trust_remote_code=False,
            dtype=dtype,
            max_model_len=self.spec.max_seq_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            max_logprobs=-1,
            logprobs_mode="raw_logits",
        )

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, errors="replace")

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        if len(input_ids) > self.max_context_tokens:
            raise ValueError("input_ids exceed model context")
        for token_id in input_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise TypeError("input token IDs must be integers")
            if not 0 <= token_id < self.spec.vocab_size:
                raise ValueError("input token ID is outside model vocabulary")

        params = self._SamplingParams(
            max_tokens=1,
            temperature=0.0,
            logprobs=-1,
            detokenize=False,
            ignore_eos=True,
        )
        requests = self._engine.generate(
            [{"prompt_token_ids": list(input_ids)}],
            params,
            use_tqdm=False,
        )
        if len(requests) != 1 or len(requests[0].outputs) != 1:
            raise VllmRuntimeError("vLLM returned an unexpected request/output count")
        output = requests[0].outputs[0]
        if len(output.token_ids) != 1:
            raise VllmRuntimeError("vLLM did not return exactly one generated token")
        if output.logprobs is None or len(output.logprobs) != 1:
            raise VllmRuntimeError("vLLM did not return one raw-logit vector")
        raw = output.logprobs[0]
        if len(raw) != self.spec.vocab_size:
            raise VllmRuntimeError(
                f"vLLM raw-logit vocabulary mismatch: expected {self.spec.vocab_size}, "
                f"got {len(raw)}"
            )

        logits: list[float] = []
        for token_id in range(self.spec.vocab_size):
            item = raw.get(token_id)
            if item is None:
                raise VllmRuntimeError(f"vLLM raw logits omitted token ID {token_id}")
            value = getattr(item, "logprob", None)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise VllmRuntimeError("vLLM raw-logit value is not numeric")
            logits.append(float(value))

        sampled = int(output.token_ids[0])
        expected = greedy_token(logits)
        if sampled != expected:
            raise VllmRuntimeError(
                "vLLM greedy sample disagrees with returned raw logits: "
                f"sampled={sampled} argmax={expected}"
            )
        return logits


def collect_vllm_runtime_parity(
    checkpoint_dir: str | Path,
    model_dir: str | Path,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 8,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    dtype: str = "float32",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.5,
) -> dict[str, Any]:
    """Execute first-party-vs-vLLM raw-logit/token/decode/context parity."""

    if not prompts:
        raise ValueError("at least one parity prompt is required")
    reference = load_first_party_backend(Path(checkpoint_dir))
    prompt_token_lengths = [len(reference.encode(prompt)) for prompt in prompts]
    near_limit = reference.max_context_tokens - 1
    if near_limit > 0 and near_limit not in prompt_token_lengths:
        raise ValueError(
            "runtime parity requires one prompt of exactly max_context_tokens - 1 "
            "tokens to prove near-limit context behavior"
        )

    binding = verify_vllm_llama_directory(model_dir)
    if binding.get("checkpoint_id") != reference.manifest.get("checkpoint_id"):
        raise VllmRuntimeError("standard Llama export checkpoint_id does not match oracle")
    if binding.get("model_spec_sha256") != reference.manifest["identity"]["model_spec_hash"]:
        raise VllmRuntimeError("standard Llama export ModelSpec does not match oracle")

    candidate = VllmNativeLlamaBackend(
        model_dir,
        tokenizer=reference.tokenizer,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    report = compare_backends(
        reference,
        candidate,
        tuple(prompts),
        max_new_tokens=max_new_tokens,
        atol=atol,
        rtol=rtol,
    )
    payload: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "passed": report.passed,
        "checkpoint_id": reference.manifest["checkpoint_id"],
        "source_git_sha": reference.manifest["identity"]["git_sha"],
        "model_spec_sha256": reference.manifest["identity"]["model_spec_hash"],
        "tokenizer_config_sha256": reference.tokenizer.identity.config_sha256,
        "tokenizer_vocab_sha256": reference.tokenizer.identity.vocab_sha256,
        "vllm_model_config_sha256": binding["target"]["config_sha256"],
        "vllm_model_weights_sha256": binding["target"]["weights_sha256"],
        "source_export": binding["source_export"],
        "vllm_version": candidate.vllm_version,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "prompt_sha256": [
            hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in prompts
        ],
        "context_behavior": {
            "max_context_tokens": reference.max_context_tokens,
            "near_limit_probe_tokens": near_limit,
            "near_limit_probe_executed": near_limit in prompt_token_lengths,
            "over_context_rejected_before_vllm": True,
        },
        "parity": report.to_dict(),
        "tolerance_basis": (
            "The incumbent Q/K basis conversion is exact; nonzero runtime tolerance "
            "covers different maintained attention/matmul kernel reduction order."
        ),
    }
    payload["evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return payload
