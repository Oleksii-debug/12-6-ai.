"""Identity-bound fresh-process inference evidence for first-party 12-6 checkpoints.

This module does not define a second inference backend.  The child process loads
the checkpoint through :class:`FirstPartyInference`, which delegates to the
maintained D05 verified-checkpoint path, and returns only private probe data to
the parent.  The parent verifies OS process separation and emits a compact,
text-free receipt suitable for downstream D06 binding.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .api import FirstPartyInference
from .contracts import GenerationConfig

SCHEMA_VERSION = "d07-fresh-process-inference-v1"
PRODUCER_IDENTITY = "twelve_six.inference.fresh_process"
_CHILD_REQUEST_SCHEMA = "d07-fresh-process-child-request-v1"
_CHILD_RESPONSE_SCHEMA = "d07-fresh-process-child-response-v1"

_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "producer_identity",
        "checkpoint_identity",
        "checkpoint_provenance_identity",
        "git_sha",
        "model_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
        "optimizer_step",
        "tokens_seen",
        "prompt_suite_identity",
        "prompt_payload_sha256",
        "generation_config_identity",
        "output_fingerprint",
        "process_run_identity",
        "child_pid",
        "parent_pid",
        "challenge_sha256",
        "receipt_identity",
    }
)

_CHILD_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "checkpoint",
        "parent_pid",
        "challenge",
        "prompt_token_ids",
        "prompt_suite_identity",
        "prompt_payload_sha256",
        "generation_config",
        "generation_config_identity",
        "cache_mode",
    }
)

_CHILD_RESPONSE_KEYS = frozenset(
    {
        "schema_version",
        "parent_pid",
        "child_pid",
        "challenge",
        "prompt_suite_identity",
        "prompt_payload_sha256",
        "generation_config_identity",
        "generated_token_ids",
        "stop_reason",
        "diagnostics",
    }
)

_PROVENANCE_FIELDS = (
    "checkpoint_identity",
    "git_sha",
    "model_spec_sha256",
    "tokenizer_config_sha256",
    "tokenizer_vocab_sha256",
    "dataset_manifest_sha256",
    "run_manifest_sha256",
    "optimizer_step",
    "tokens_seen",
)


class FreshProcessEvidenceError(RuntimeError):
    """Raised when fresh-process evidence cannot be produced or verified."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256_identity(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_sha256_identity(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    return (
        len(digest) == 64
        and digest == digest.lower()
        and all(char in "0123456789abcdef" for char in digest)
    )


def _is_hex(value: Any, *, lengths: tuple[int, ...]) -> bool:
    return (
        isinstance(value, str)
        and len(value) in lengths
        and value == value.lower()
        and all(char in "0123456789abcdef" for char in value)
    )


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalize_generation_config(config: GenerationConfig, cache_mode: str) -> dict[str, Any]:
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be GenerationConfig")
    if config.sample:
        raise ValueError("fresh-process terminal probe requires deterministic sample=False")
    if config.max_new_tokens <= 0:
        raise ValueError("fresh-process terminal probe requires max_new_tokens > 0")
    if cache_mode not in {"static", "stateless"}:
        raise ValueError("cache_mode must be 'static' or 'stateless'")
    payload = asdict(config)
    payload["stop_token_ids"] = list(config.stop_token_ids)
    payload["stop_strings"] = list(config.stop_strings)
    payload["cache_mode"] = cache_mode
    return payload


def generation_config_identity(
    config: GenerationConfig, *, cache_mode: str = "stateless"
) -> str:
    """Return the canonical identity of the deterministic generation contract."""

    return _sha256_identity(_normalize_generation_config(config, cache_mode))


def prompt_payload_identity(prompt_token_ids: Sequence[int]) -> str:
    """Hash exact prompt token IDs without retaining prompt text in evidence."""

    if isinstance(prompt_token_ids, (str, bytes)) or not isinstance(
        prompt_token_ids, Sequence
    ):
        raise TypeError("prompt_token_ids must be a sequence of integers")
    values: list[int] = []
    for token_id in prompt_token_ids:
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("prompt_token_ids must contain integers")
        if token_id < 0:
            raise ValueError("prompt_token_ids must be non-negative")
        values.append(token_id)
    if not values:
        raise ValueError("prompt_token_ids must be non-empty")
    return _sha256_identity({"prompt_token_ids": values})


def _checkpoint_provenance_identity(receipt: Mapping[str, Any]) -> str:
    return _sha256_identity({key: receipt.get(key) for key in _PROVENANCE_FIELDS})


def _process_run_identity(receipt: Mapping[str, Any]) -> str:
    return _sha256_identity(
        {
            "checkpoint_identity": receipt.get("checkpoint_identity"),
            "prompt_suite_identity": receipt.get("prompt_suite_identity"),
            "prompt_payload_sha256": receipt.get("prompt_payload_sha256"),
            "generation_config_identity": receipt.get("generation_config_identity"),
            "output_fingerprint": receipt.get("output_fingerprint"),
            "child_pid": receipt.get("child_pid"),
            "parent_pid": receipt.get("parent_pid"),
            "challenge_sha256": receipt.get("challenge_sha256"),
        }
    )


def _receipt_identity(receipt: Mapping[str, Any]) -> str:
    return _sha256_identity(
        {key: receipt.get(key) for key in sorted(_RECEIPT_KEYS - {"receipt_identity"})}
    )


def validate_fresh_process_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_identity: str,
    expected_checkpoint_identity: str,
    expected_prompt_suite_identity: str,
    expected_prompt_payload_sha256: str,
    expected_generation_config_identity: str,
) -> list[str]:
    """Validate one already-produced receipt against externally bound expectations.

    The expected receipt identity is mandatory.  This prevents a caller from
    modifying semantic fields and merely recomputing a new self-consistent hash
    while claiming that it is the previously authorized evidence object.
    """

    blockers: list[str] = []
    if not isinstance(receipt, Mapping):
        return ["d07.fresh_process.receipt_not_mapping"]
    keys = set(receipt)
    if keys != _RECEIPT_KEYS:
        if missing := sorted(_RECEIPT_KEYS - keys):
            blockers.append("d07.fresh_process.receipt_keys_missing:" + ",".join(missing))
        if extra := sorted(keys - _RECEIPT_KEYS):
            blockers.append("d07.fresh_process.receipt_keys_unknown:" + ",".join(extra))

    if receipt.get("schema_version") != SCHEMA_VERSION:
        blockers.append("d07.fresh_process.schema_version_mismatch")
    if receipt.get("producer_identity") != PRODUCER_IDENTITY:
        blockers.append("d07.fresh_process.producer_identity_mismatch")

    checkpoint_identity = receipt.get("checkpoint_identity")
    if not _is_hex(checkpoint_identity, lengths=(64,)):
        blockers.append("d07.fresh_process.checkpoint_identity_invalid")
    if checkpoint_identity != expected_checkpoint_identity:
        blockers.append("d07.fresh_process.checkpoint_identity_mismatch")

    for key, lengths in (
        ("git_sha", (40, 64)),
        ("model_spec_sha256", (64,)),
        ("tokenizer_config_sha256", (64,)),
        ("tokenizer_vocab_sha256", (64,)),
        ("dataset_manifest_sha256", (64,)),
        ("run_manifest_sha256", (64,)),
    ):
        if not _is_hex(receipt.get(key), lengths=lengths):
            blockers.append(f"d07.fresh_process.{key}_invalid")

    if not _nonnegative_int(receipt.get("optimizer_step")):
        blockers.append("d07.fresh_process.optimizer_step_invalid")
    if not _nonnegative_int(receipt.get("tokens_seen")):
        blockers.append("d07.fresh_process.tokens_seen_invalid")

    if not _nonempty_text(receipt.get("prompt_suite_identity")):
        blockers.append("d07.fresh_process.prompt_suite_identity_invalid")
    if receipt.get("prompt_suite_identity") != expected_prompt_suite_identity:
        blockers.append("d07.fresh_process.prompt_suite_identity_mismatch")

    for key, expected in (
        ("prompt_payload_sha256", expected_prompt_payload_sha256),
        ("generation_config_identity", expected_generation_config_identity),
    ):
        if not _is_sha256_identity(receipt.get(key)):
            blockers.append(f"d07.fresh_process.{key}_invalid")
        if receipt.get(key) != expected:
            blockers.append(f"d07.fresh_process.{key}_mismatch")

    if not _is_sha256_identity(receipt.get("output_fingerprint")):
        blockers.append("d07.fresh_process.output_fingerprint_invalid")
    if not _is_sha256_identity(receipt.get("challenge_sha256")):
        blockers.append("d07.fresh_process.challenge_sha256_invalid")

    child_pid = receipt.get("child_pid")
    parent_pid = receipt.get("parent_pid")
    if not _positive_int(child_pid):
        blockers.append("d07.fresh_process.child_pid_invalid")
    if not _positive_int(parent_pid):
        blockers.append("d07.fresh_process.parent_pid_invalid")
    if _positive_int(child_pid) and _positive_int(parent_pid) and child_pid == parent_pid:
        blockers.append("d07.fresh_process.process_not_fresh")

    expected_provenance = _checkpoint_provenance_identity(receipt)
    if not _is_sha256_identity(receipt.get("checkpoint_provenance_identity")):
        blockers.append("d07.fresh_process.checkpoint_provenance_identity_invalid")
    elif receipt.get("checkpoint_provenance_identity") != expected_provenance:
        blockers.append("d07.fresh_process.checkpoint_provenance_identity_mismatch")

    expected_run = _process_run_identity(receipt)
    if not _is_sha256_identity(receipt.get("process_run_identity")):
        blockers.append("d07.fresh_process.process_run_identity_invalid")
    elif receipt.get("process_run_identity") != expected_run:
        blockers.append("d07.fresh_process.process_run_identity_mismatch")

    actual_receipt_identity = receipt.get("receipt_identity")
    if not _is_sha256_identity(actual_receipt_identity):
        blockers.append("d07.fresh_process.receipt_identity_invalid")
    else:
        if actual_receipt_identity != _receipt_identity(receipt):
            blockers.append("d07.fresh_process.receipt_identity_self_mismatch")
        if actual_receipt_identity != expected_receipt_identity:
            blockers.append("d07.fresh_process.receipt_identity_expected_mismatch")
    return sorted(set(blockers))


def _child_response(request: Mapping[str, Any]) -> dict[str, Any]:
    if set(request) != _CHILD_REQUEST_KEYS:
        raise FreshProcessEvidenceError("fresh-process child request has non-canonical keys")
    if request.get("schema_version") != _CHILD_REQUEST_SCHEMA:
        raise FreshProcessEvidenceError("fresh-process child request schema mismatch")

    parent_pid = request.get("parent_pid")
    if (
        not _positive_int(parent_pid)
        or parent_pid == os.getpid()
        or parent_pid != os.getppid()
    ):
        raise FreshProcessEvidenceError("fresh-process parent PID is invalid")
    challenge = request.get("challenge")
    if not _is_hex(challenge, lengths=(64,)):
        raise FreshProcessEvidenceError("fresh-process challenge must be exact 64-hex")
    prompt_suite_identity = request.get("prompt_suite_identity")
    if not _nonempty_text(prompt_suite_identity):
        raise FreshProcessEvidenceError("prompt_suite_identity must be non-empty")

    raw_tokens = request.get("prompt_token_ids")
    if not isinstance(raw_tokens, list):
        raise FreshProcessEvidenceError("prompt_token_ids must be a JSON list")
    prompt_token_ids: list[int] = []
    for token_id in raw_tokens:
        if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
            raise FreshProcessEvidenceError("prompt_token_ids contain an invalid value")
        prompt_token_ids.append(token_id)
    if not prompt_token_ids:
        raise FreshProcessEvidenceError("prompt_token_ids must be non-empty")
    if prompt_payload_identity(prompt_token_ids) != request.get("prompt_payload_sha256"):
        raise FreshProcessEvidenceError("prompt payload identity mismatch")

    raw_config = request.get("generation_config")
    if not isinstance(raw_config, Mapping):
        raise FreshProcessEvidenceError("generation_config must be a mapping")
    config_keys = set(GenerationConfig.__dataclass_fields__)
    if set(raw_config) != config_keys:
        raise FreshProcessEvidenceError("generation_config keys are not canonical")
    normalized_config = dict(raw_config)
    normalized_config["stop_token_ids"] = tuple(normalized_config["stop_token_ids"])
    normalized_config["stop_strings"] = tuple(normalized_config["stop_strings"])
    config = GenerationConfig(**normalized_config)
    cache_mode = request.get("cache_mode")
    if generation_config_identity(config, cache_mode=cache_mode) != request.get(
        "generation_config_identity"
    ):
        raise FreshProcessEvidenceError("generation config identity mismatch")

    checkpoint = request.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise FreshProcessEvidenceError("checkpoint path is missing")
    inference = FirstPartyInference.from_checkpoint(Path(checkpoint))
    result = inference.generate_token_ids(
        prompt_token_ids,
        config,
        cache_mode=cache_mode,
    )
    return {
        "schema_version": _CHILD_RESPONSE_SCHEMA,
        "parent_pid": parent_pid,
        "child_pid": os.getpid(),
        "challenge": challenge,
        "prompt_suite_identity": prompt_suite_identity,
        "prompt_payload_sha256": request["prompt_payload_sha256"],
        "generation_config_identity": request["generation_config_identity"],
        "generated_token_ids": list(result.generated_token_ids),
        "stop_reason": result.stop_reason,
        "diagnostics": inference.diagnostics(),
    }


def _parse_child_response(
    payload: str,
    *,
    process_pid: int,
    parent_pid: int,
    challenge: str,
    expected_checkpoint_identity: str,
    expected_prompt_suite_identity: str,
    expected_prompt_payload_sha256: str,
    expected_generation_config_identity: str,
) -> dict[str, Any]:
    try:
        response = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FreshProcessEvidenceError("fresh-process child did not emit valid JSON") from exc
    if not isinstance(response, Mapping) or set(response) != _CHILD_RESPONSE_KEYS:
        raise FreshProcessEvidenceError("fresh-process child response is not closed-world")
    if response.get("schema_version") != _CHILD_RESPONSE_SCHEMA:
        raise FreshProcessEvidenceError("fresh-process child response schema mismatch")
    if response.get("parent_pid") != parent_pid:
        raise FreshProcessEvidenceError("fresh-process parent PID mismatch")
    if response.get("child_pid") != process_pid or process_pid == parent_pid:
        raise FreshProcessEvidenceError("fresh-process OS process separation not proven")
    if response.get("challenge") != challenge:
        raise FreshProcessEvidenceError("fresh-process challenge mismatch")
    if response.get("prompt_suite_identity") != expected_prompt_suite_identity:
        raise FreshProcessEvidenceError("fresh-process prompt-suite identity mismatch")
    if response.get("prompt_payload_sha256") != expected_prompt_payload_sha256:
        raise FreshProcessEvidenceError("fresh-process prompt payload identity mismatch")
    if response.get("generation_config_identity") != expected_generation_config_identity:
        raise FreshProcessEvidenceError("fresh-process generation config identity mismatch")

    diagnostics = response.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise FreshProcessEvidenceError("fresh-process diagnostics are missing")
    required_diagnostics = {
        "checkpoint_id",
        "git_sha",
        "model_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
        "step",
        "tokens_seen",
        "source_kind",
    }
    if not required_diagnostics <= set(diagnostics):
        raise FreshProcessEvidenceError("fresh-process checkpoint diagnostics are incomplete")
    if diagnostics.get("source_kind") != "checkpoint":
        raise FreshProcessEvidenceError("fresh-process source is not a verified checkpoint")
    if diagnostics.get("checkpoint_id") != expected_checkpoint_identity:
        raise FreshProcessEvidenceError("fresh-process loaded the wrong checkpoint")

    generated = response.get("generated_token_ids")
    if not isinstance(generated, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in generated
    ):
        raise FreshProcessEvidenceError("fresh-process generated token IDs are invalid")
    stop_reason = response.get("stop_reason")
    if not _nonempty_text(stop_reason):
        raise FreshProcessEvidenceError("fresh-process stop reason is invalid")

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "producer_identity": PRODUCER_IDENTITY,
        "checkpoint_identity": diagnostics["checkpoint_id"],
        "checkpoint_provenance_identity": "",
        "git_sha": diagnostics["git_sha"],
        "model_spec_sha256": diagnostics["model_spec_sha256"],
        "tokenizer_config_sha256": diagnostics["tokenizer_config_sha256"],
        "tokenizer_vocab_sha256": diagnostics["tokenizer_vocab_sha256"],
        "dataset_manifest_sha256": diagnostics["dataset_manifest_sha256"],
        "run_manifest_sha256": diagnostics["run_manifest_sha256"],
        "optimizer_step": diagnostics["step"],
        "tokens_seen": diagnostics["tokens_seen"],
        "prompt_suite_identity": expected_prompt_suite_identity,
        "prompt_payload_sha256": expected_prompt_payload_sha256,
        "generation_config_identity": expected_generation_config_identity,
        "output_fingerprint": _sha256_identity(
            {
                "generated_token_ids": generated,
                "stop_reason": stop_reason,
            }
        ),
        "process_run_identity": "",
        "child_pid": process_pid,
        "parent_pid": parent_pid,
        "challenge_sha256": _sha256_identity({"challenge": challenge}),
        "receipt_identity": "",
    }
    receipt["checkpoint_provenance_identity"] = _checkpoint_provenance_identity(receipt)
    receipt["process_run_identity"] = _process_run_identity(receipt)
    receipt["receipt_identity"] = _receipt_identity(receipt)
    return receipt


def run_fresh_process_probe(
    checkpoint: str | Path,
    *,
    expected_checkpoint_identity: str,
    prompt_token_ids: Sequence[int],
    prompt_suite_identity: str,
    config: GenerationConfig,
    cache_mode: str = "stateless",
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Spawn a new Python process and return one verified text-free receipt."""

    if not _is_hex(expected_checkpoint_identity, lengths=(64,)):
        raise ValueError("expected_checkpoint_identity must be exact lowercase 64-hex")
    if not _nonempty_text(prompt_suite_identity):
        raise ValueError("prompt_suite_identity must be non-empty")
    if (
        not isinstance(timeout_seconds, (int, float))
        or isinstance(timeout_seconds, bool)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be > 0")

    prompt_ids = list(prompt_token_ids)
    prompt_hash = prompt_payload_identity(prompt_ids)
    config_payload = _normalize_generation_config(config, cache_mode)
    config_identity = _sha256_identity(config_payload)
    child_config = dict(config_payload)
    child_config.pop("cache_mode")

    parent_pid = os.getpid()
    challenge = secrets.token_hex(32)
    request = {
        "schema_version": _CHILD_REQUEST_SCHEMA,
        "checkpoint": str(Path(checkpoint).resolve()),
        "parent_pid": parent_pid,
        "challenge": challenge,
        "prompt_token_ids": prompt_ids,
        "prompt_suite_identity": prompt_suite_identity,
        "prompt_payload_sha256": prompt_hash,
        "generation_config": child_config,
        "generation_config_identity": config_identity,
        "cache_mode": cache_mode,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "twelve_six.inference.fresh_process", "--child"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(
            json.dumps(request, sort_keys=True, separators=(",", ":")),
            timeout=float(timeout_seconds),
        )
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.communicate()
        raise FreshProcessEvidenceError("fresh-process inference probe timed out") from exc
    if process.returncode != 0:
        detail = stderr.strip()[-2000:]
        raise FreshProcessEvidenceError(
            "fresh-process inference probe failed"
            + (f": {detail}" if detail else "")
        )

    receipt = _parse_child_response(
        stdout,
        process_pid=process.pid,
        parent_pid=parent_pid,
        challenge=challenge,
        expected_checkpoint_identity=expected_checkpoint_identity,
        expected_prompt_suite_identity=prompt_suite_identity,
        expected_prompt_payload_sha256=prompt_hash,
        expected_generation_config_identity=config_identity,
    )
    blockers = validate_fresh_process_receipt(
        receipt,
        expected_receipt_identity=receipt["receipt_identity"],
        expected_checkpoint_identity=expected_checkpoint_identity,
        expected_prompt_suite_identity=prompt_suite_identity,
        expected_prompt_payload_sha256=prompt_hash,
        expected_generation_config_identity=config_identity,
    )
    if blockers:
        raise FreshProcessEvidenceError(
            "fresh-process receipt failed self-validation: " + ";".join(blockers)
        )
    return receipt


def _main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["--child"]:
        print(
            "fresh-process module is an internal evidence worker; expected --child",
            file=sys.stderr,
        )
        return 2
    try:
        request = json.loads(sys.stdin.read())
        if not isinstance(request, Mapping):
            raise FreshProcessEvidenceError("fresh-process child request must be a mapping")
        response = _child_response(request)
    except (FreshProcessEvidenceError, OSError, TypeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
