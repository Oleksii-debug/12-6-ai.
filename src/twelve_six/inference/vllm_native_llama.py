"""Native vLLM Llama execution adapter for verified 12-6 exports.

This module deliberately does not implement a vLLM model class. Representable
12-6 ModelSpecs are converted into the standard Llama weight/config contract so
vLLM can use its maintained Llama implementation, scheduler, attention kernels,
and KV cache. The canonical 12-6 tokenizer remains authoritative.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from safetensors.torch import load as load_safetensors_bytes
from safetensors.torch import save as save_safetensors_bytes

from twelve_six.checkpoint import verify_hf_directory
from twelve_six.model import ModelSpec
from twelve_six.tokenization import ByteTokenizer

from .first_party import load_first_party_backend
from .parity import compare_backends
from .sampling import greedy_token
from .transformers_llama import (
    build_llama_interop_plan,
    convert_state_dict_to_llama,
    llama_config_dict,
)

MATERIALIZATION_SCHEMA = "12-6.vllm-native-llama-materialization.v1"
EVIDENCE_SCHEMA = "12-6.vllm-native-llama-runtime-parity.v1"
PROVENANCE_NAME = "12-6-vllm-runtime.json"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
SOURCE_MANIFEST_NAME = "12-6-checkpoint-manifest.json"
TARGET_ARCHITECTURE = "LlamaForCausalLM"
TARGET_MODEL_TYPE = "llama"

_TARGET_INVENTORY = frozenset({CONFIG_NAME, WEIGHTS_NAME, PROVENANCE_NAME})


class VllmRuntimeError(ValueError):
    """Raised when vLLM materialization or runtime evidence is invalid."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _source_spec(source_manifest: Mapping[str, Any]) -> ModelSpec:
    identity = source_manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise VllmRuntimeError("source checkpoint identity is missing")
    raw_spec = identity.get("model_spec")
    if not isinstance(raw_spec, Mapping):
        raise VllmRuntimeError("source checkpoint ModelSpec is missing")
    try:
        spec = ModelSpec.from_dict(dict(raw_spec))
    except (TypeError, ValueError) as exc:
        raise VllmRuntimeError(f"source checkpoint ModelSpec is invalid: {exc}") from exc
    if spec.identity_sha256() != identity.get("model_spec_hash"):
        raise VllmRuntimeError("source checkpoint ModelSpec hash mismatch")
    if spec.parameter_count() != identity.get("parameter_count"):
        raise VllmRuntimeError("source checkpoint parameter count mismatch")
    return spec


def _verified_source_snapshot(
    export_dir: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], bytes, ModelSpec]:
    export = Path(export_dir)
    attestation = verify_hf_directory(export)

    source_manifest_path = export / SOURCE_MANIFEST_NAME
    weights_path = export / WEIGHTS_NAME
    source_manifest_bytes = source_manifest_path.read_bytes()
    weights_bytes = weights_path.read_bytes()

    expected_manifest_sha = attestation.get("source_manifest_sha256")
    if _sha256_bytes(source_manifest_bytes) != expected_manifest_sha:
        raise VllmRuntimeError("consumed source manifest changed after export verification")
    expected_weights_sha = attestation.get("model_safetensors_sha256")
    if _sha256_bytes(weights_bytes) != expected_weights_sha:
        raise VllmRuntimeError("consumed source weights changed after export verification")

    try:
        source_manifest = json.loads(source_manifest_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VllmRuntimeError("verified source manifest cannot be decoded") from exc
    if not isinstance(source_manifest, dict):
        raise VllmRuntimeError("verified source manifest root must be an object")
    if source_manifest.get("checkpoint_id") != attestation.get("checkpoint_id"):
        raise VllmRuntimeError("source manifest checkpoint_id does not match export")

    return attestation, source_manifest, weights_bytes, _source_spec(source_manifest)


def materialize_vllm_llama_directory(
    export_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Convert a verified 12-6 HF-style export into standard Llama runtime bytes.

    The input is the exact transactional D05 export. No pretrained or external
    model weights are consulted. Q/K rows are converted only through the D07
    Transformers Llama bridge; all other supported tensors are copied.
    """

    source = Path(export_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"vLLM runtime directory already exists: {output}")

    attestation, source_manifest, source_weights_bytes, spec = _verified_source_snapshot(source)
    try:
        source_state = load_safetensors_bytes(source_weights_bytes)
    except Exception as exc:
        raise VllmRuntimeError("verified source model.safetensors cannot be decoded") from exc

    plan = build_llama_interop_plan(spec)
    target_state = convert_state_dict_to_llama(spec, source_state)
    target_weights_bytes = save_safetensors_bytes(target_state)
    target_config = llama_config_dict(spec)
    target_config_bytes = _canonical_json_bytes(target_config) + b"\n"

    identity = source_manifest["identity"]
    provenance: dict[str, Any] = {
        "schema": MATERIALIZATION_SCHEMA,
        "checkpoint_id": attestation["checkpoint_id"],
        "source_export": {
            "source_manifest_sha256": attestation["source_manifest_sha256"],
            "source_weights_sha256": attestation["model_safetensors_sha256"],
            "source_config_sha256": attestation["config_sha256"],
        },
        "source_model_spec": spec.to_dict(),
        "model_spec_sha256": spec.identity_sha256(),
        "parameter_count": spec.parameter_count(),
        "tokenizer_config_sha256": identity.get("tokenizer_hash"),
        "tokenizer_vocab_sha256": identity.get("tokenizer_vocab_hash"),
        "interop_plan_sha256": plan.identity_sha256(),
        "target": {
            "architecture": TARGET_ARCHITECTURE,
            "model_type": TARGET_MODEL_TYPE,
            "config_sha256": _sha256_bytes(target_config_bytes),
            "weights_sha256": _sha256_bytes(target_weights_bytes),
        },
        "execution_contract": {
            "vllm_implementation": "BUILTIN_LLAMA",
            "skip_tokenizer_init": True,
            "prompt_input": "TOKEN_IDS",
            "tokenizer_owner": "12-6.s0-byte-v1",
            "bos_token_id": None,
            "eos_token_id": None,
            "pad_token_id": None,
        },
    }
    provenance_bytes = _canonical_json_bytes(provenance) + b"\n"

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        (staging / CONFIG_NAME).write_bytes(target_config_bytes)
        (staging / WEIGHTS_NAME).write_bytes(target_weights_bytes)
        (staging / PROVENANCE_NAME).write_bytes(provenance_bytes)
        verify_vllm_llama_directory(staging)
        if output.exists():
            raise FileExistsError(f"vLLM runtime directory already exists: {output}")
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def verify_vllm_llama_directory(model_dir: str | Path) -> dict[str, Any]:
    """Verify the exact standard-Llama materialization and return provenance."""

    root = Path(model_dir)
    if root.is_symlink() or not root.is_dir():
        raise VllmRuntimeError("vLLM model directory must be a regular directory")
    inventory = {entry.name for entry in root.iterdir()}
    if inventory != _TARGET_INVENTORY:
        raise VllmRuntimeError(
            f"vLLM model directory inventory mismatch: expected={sorted(_TARGET_INVENTORY)} "
            f"actual={sorted(inventory)}"
        )
    for name in _TARGET_INVENTORY:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise VllmRuntimeError(f"vLLM runtime payload is not a regular file: {name}")

    provenance = _load_json_object(root / PROVENANCE_NAME)
    if provenance.get("schema") != MATERIALIZATION_SCHEMA:
        raise VllmRuntimeError("unsupported vLLM materialization schema")

    target = provenance.get("target")
    if not isinstance(target, Mapping):
        raise VllmRuntimeError("vLLM materialization target metadata is missing")
    if target.get("architecture") != TARGET_ARCHITECTURE:
        raise VllmRuntimeError("unexpected vLLM target architecture")
    if target.get("model_type") != TARGET_MODEL_TYPE:
        raise VllmRuntimeError("unexpected vLLM target model_type")
    if target.get("config_sha256") != _sha256_file(root / CONFIG_NAME):
        raise VllmRuntimeError("vLLM config hash mismatch")
    if target.get("weights_sha256") != _sha256_file(root / WEIGHTS_NAME):
        raise VllmRuntimeError("vLLM weights hash mismatch")

    raw_spec = provenance.get("source_model_spec")
    if not isinstance(raw_spec, Mapping):
        raise VllmRuntimeError("source ModelSpec missing from vLLM provenance")
    try:
        spec = ModelSpec.from_dict(dict(raw_spec))
    except (TypeError, ValueError) as exc:
        raise VllmRuntimeError(f"invalid source ModelSpec in vLLM provenance: {exc}") from exc
    if spec.identity_sha256() != provenance.get("model_spec_sha256"):
        raise VllmRuntimeError("vLLM provenance ModelSpec hash mismatch")
    if spec.parameter_count() != provenance.get("parameter_count"):
        raise VllmRuntimeError("vLLM provenance parameter count mismatch")

    config = _load_json_object(root / CONFIG_NAME)
    if config != llama_config_dict(spec):
        raise VllmRuntimeError("vLLM config does not exactly match D07 Llama bridge")

    weights_bytes = (root / WEIGHTS_NAME).read_bytes()
    try:
        state = load_safetensors_bytes(weights_bytes)
    except Exception as exc:
        raise VllmRuntimeError("vLLM model.safetensors cannot be decoded") from exc
    expected_targets = {
        row["target"] for row in build_llama_interop_plan(spec).tensor_map
    }
    if set(state) != expected_targets:
        raise VllmRuntimeError("vLLM Llama tensor inventory mismatch")
    return provenance


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
    """Import vLLM, prove built-in Llama registration, and construct ModelConfig.

    This is a real installed-runtime probe but intentionally does not initialize
    a device or claim logits/generation parity.
    """

    provenance = verify_vllm_llama_directory(model_dir)
    spec = ModelSpec.from_dict(dict(provenance["source_model_spec"]))

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
        dtype: str = "float32",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.5,
        enforce_eager: bool = True,
    ) -> None:
        provenance = verify_vllm_llama_directory(model_dir)
        self.spec = ModelSpec.from_dict(dict(provenance["source_model_spec"]))
        self.tokenizer = ByteTokenizer()
        self.max_context_tokens = self.spec.max_seq_len
        if self.tokenizer.vocab_size != self.spec.vocab_size:
            raise VllmRuntimeError("canonical tokenizer vocabulary does not match ModelSpec")
        if self.tokenizer.identity.config_sha256 != provenance.get(
            "tokenizer_config_sha256"
        ):
            raise VllmRuntimeError("canonical tokenizer config identity mismatch")
        if self.tokenizer.identity.vocab_sha256 != provenance.get(
            "tokenizer_vocab_sha256"
        ):
            raise VllmRuntimeError("canonical tokenizer vocabulary identity mismatch")

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
        return self.tokenizer.encode(text)

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
                f"vLLM greedy sample disagrees with returned raw logits: "
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
    """Execute first-party-vs-vLLM logits/token/decode parity."""

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
    provenance = verify_vllm_llama_directory(model_dir)
    if provenance.get("checkpoint_id") != reference.manifest.get("checkpoint_id"):
        raise VllmRuntimeError("vLLM materialization checkpoint_id does not match oracle")

    candidate = VllmNativeLlamaBackend(
        model_dir,
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
        "vllm_model_config_sha256": provenance["target"]["config_sha256"],
        "vllm_model_weights_sha256": provenance["target"]["weights_sha256"],
        "interop_plan_sha256": provenance["interop_plan_sha256"],
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
            "Q/K basis conversion is exact; nonzero runtime tolerance covers "
            "different maintained attention/matmul kernel reduction order."
        ),
    }
    payload["evidence_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return payload
