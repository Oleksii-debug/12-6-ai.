"""Durable, replayable evidence for verified first-party Base generation.

This module records checkpoint/runtime identity plus deterministic generation traces
without changing the canonical generation or sampling implementation. The trace
backend delegates to the existing D07 backend and only fingerprints each logits
call. Evidence is self-hashed and can be replayed against a freshly loaded D05
checkpoint to detect checkpoint, tokenizer, context, sampling, token, logit, or
decode drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import GenerationConfig, InferenceBackend
from .first_party import load_first_party_backend
from .generation import generate
from .sampling import greedy_token

EVIDENCE_SCHEMA = "12-6.first-party-inference-evidence.v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_REQUIRED_DIAGNOSTICS = (
    "backend",
    "checkpoint_id",
    "git_sha",
    "model_spec_sha256",
    "parameter_count",
    "vocab_size",
    "max_context_tokens",
    "tokenizer_version",
    "tokenizer_config_sha256",
    "tokenizer_vocab_sha256",
    "dataset_manifest_sha256",
    "run_manifest_sha256",
    "step",
    "tokens_seen",
    "device",
)
_ROOT_KEYS = {"schema", "checkpoint", "probes", "claims", "evidence_sha256"}
_PROBE_KEYS = {
    "name",
    "prompt_utf8_sha256",
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "config",
    "generated_token_ids",
    "generated_token_ids_sha256",
    "output_utf8_sha256",
    "stop_reason",
    "step_trace",
    "step_trace_sha256",
    "record_sha256",
}
_STEP_KEYS = {
    "input_length",
    "input_ids_sha256",
    "logit_count",
    "logits_float64_hex_sha256",
    "greedy_token_id",
}
_CONFIG_KEYS = {
    "max_new_tokens",
    "sample",
    "temperature",
    "top_k",
    "top_p",
    "seed",
    "stop_token_ids",
    "stop_strings",
    "strip_stop_strings",
}
_STOP_REASONS = {"max_new_tokens", "context_limit", "eos", "stop_token", "stop_string"}
_CLAIMS = {
    "raw_base_completion_only": True,
    "instruction_or_chat_semantics": False,
    "cross_hardware_bitwise_reproducibility": False,
    "promotion_authority": False,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_diagnostics(diagnostics: object) -> dict[str, object]:
    if not isinstance(diagnostics, dict):
        raise TypeError("inference evidence checkpoint must be an object")
    if set(diagnostics) != set(_REQUIRED_DIAGNOSTICS):
        raise ValueError("inference evidence checkpoint identity fields are incomplete")
    if diagnostics["backend"] != "first_party_torch":
        raise ValueError("inference evidence requires the canonical first_party_torch backend")
    _require_sha256(diagnostics["checkpoint_id"], field="checkpoint_id")
    git_sha = diagnostics["git_sha"]
    if not isinstance(git_sha, str) or _GIT_SHA.fullmatch(git_sha) is None:
        raise ValueError("git_sha must be a lowercase full Git object id")
    for field in (
        "model_spec_sha256",
        "tokenizer_config_sha256",
        "tokenizer_vocab_sha256",
        "dataset_manifest_sha256",
        "run_manifest_sha256",
    ):
        _require_sha256(diagnostics[field], field=field)
    for field in ("parameter_count", "vocab_size", "max_context_tokens"):
        value = diagnostics[field]
        if not _is_int(value) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    for field in ("step", "tokens_seen"):
        value = diagnostics[field]
        if not _is_int(value) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer")
    for field in ("tokenizer_version", "device"):
        value = diagnostics[field]
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string")
    return diagnostics


def _diagnostics(backend: InferenceBackend) -> dict[str, object]:
    diagnostics_method = getattr(backend, "diagnostics", None)
    if not callable(diagnostics_method):
        raise TypeError("first-party inference evidence requires backend diagnostics()")
    raw = diagnostics_method()
    if not isinstance(raw, Mapping):
        raise TypeError("backend diagnostics() must return a mapping")
    missing = [key for key in _REQUIRED_DIAGNOSTICS if key not in raw]
    if missing:
        raise ValueError(f"backend diagnostics missing required fields: {missing}")
    projected = {key: raw[key] for key in _REQUIRED_DIAGNOSTICS}
    return _validate_diagnostics(projected)


def _config_payload(config: GenerationConfig) -> dict[str, object]:
    payload = asdict(config)
    payload["stop_token_ids"] = list(config.stop_token_ids)
    payload["stop_strings"] = list(config.stop_strings)
    return payload


def _config_from_payload(payload: object) -> GenerationConfig:
    if not isinstance(payload, Mapping):
        raise TypeError("generation config evidence must be an object")
    if set(payload) != _CONFIG_KEYS:
        raise ValueError("generation config evidence has missing or unknown fields")
    stop_token_ids = payload["stop_token_ids"]
    stop_strings = payload["stop_strings"]
    if not isinstance(stop_token_ids, list) or not all(_is_int(value) for value in stop_token_ids):
        raise TypeError("stop_token_ids evidence must be an integer array")
    if not isinstance(stop_strings, list) or not all(
        isinstance(value, str) for value in stop_strings
    ):
        raise TypeError("stop_strings evidence must be a string array")
    return GenerationConfig(
        max_new_tokens=payload["max_new_tokens"],
        sample=payload["sample"],
        temperature=payload["temperature"],
        top_k=payload["top_k"],
        top_p=payload["top_p"],
        seed=payload["seed"],
        stop_token_ids=tuple(stop_token_ids),
        stop_strings=tuple(stop_strings),
        strip_stop_strings=payload["strip_stop_strings"],
    )


class _TracingBackend:
    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend
        self.eos_token_id = backend.eos_token_id
        self.max_context_tokens = backend.max_context_tokens
        self.steps: list[dict[str, object]] = []

    def encode(self, text: str) -> list[int]:
        return self._backend.encode(text)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._backend.decode(token_ids)

    def next_token_logits(self, input_ids: Sequence[int]) -> Sequence[float]:
        logits = [float(value) for value in self._backend.next_token_logits(input_ids)]
        if not logits:
            raise ValueError("backend returned zero logits")
        if any(not math.isfinite(value) for value in logits):
            raise FloatingPointError("inference evidence refuses non-finite logits")
        logit_hex = [value.hex() for value in logits]
        self.steps.append(
            {
                "input_length": len(input_ids),
                "input_ids_sha256": _sha256(list(input_ids)),
                "logit_count": len(logits),
                "logits_float64_hex_sha256": _sha256(logit_hex),
                "greedy_token_id": greedy_token(logits),
            }
        )
        return logits


def collect_probe(
    backend: InferenceBackend,
    *,
    name: str,
    prompt: str,
    config: GenerationConfig,
) -> dict[str, object]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("probe name must be a non-empty string")
    if not isinstance(prompt, str):
        raise TypeError("probe prompt must be text")

    tracing = _TracingBackend(backend)
    result = generate(tracing, prompt, config)
    prompt_ids = list(result.prompt_token_ids)
    generated_ids = list(result.generated_token_ids)
    record: dict[str, object] = {
        "name": name,
        "prompt_utf8_sha256": _text_sha256(prompt),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": _sha256(prompt_ids),
        "config": _config_payload(config),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": _sha256(generated_ids),
        "output_utf8_sha256": _text_sha256(result.text),
        "stop_reason": result.stop_reason,
        "step_trace": tracing.steps,
        "step_trace_sha256": _sha256(tracing.steps),
    }
    record["record_sha256"] = _sha256(record)
    return record


def build_evidence(
    backend: InferenceBackend,
    probes: Sequence[tuple[str, str, GenerationConfig]],
) -> dict[str, object]:
    if not probes:
        raise ValueError("at least one inference probe is required")
    names = [name for name, _, _ in probes]
    if len(names) != len(set(names)):
        raise ValueError("inference probe names must be unique")

    payload: dict[str, object] = {
        "schema": EVIDENCE_SCHEMA,
        "checkpoint": _diagnostics(backend),
        "probes": [
            collect_probe(backend, name=name, prompt=prompt, config=config)
            for name, prompt, config in probes
        ],
        "claims": dict(_CLAIMS),
    }
    payload["evidence_sha256"] = _sha256(payload)
    return payload


def _validate_trace_step(
    step: object,
    *,
    index: int,
    prompt_length: int,
    vocab_size: int,
    max_context_tokens: int,
) -> None:
    if not isinstance(step, dict):
        raise TypeError("step_trace entries must be objects")
    if set(step) != _STEP_KEYS:
        raise ValueError("step_trace entry has missing or unknown fields")
    expected_length = prompt_length + index
    if step["input_length"] != expected_length:
        raise ValueError("step_trace input length is not contiguous")
    if expected_length >= max_context_tokens:
        raise ValueError("step_trace contains a logits call at or beyond context limit")
    _require_sha256(step["input_ids_sha256"], field="input_ids_sha256")
    _require_sha256(step["logits_float64_hex_sha256"], field="logits_float64_hex_sha256")
    if step["logit_count"] != vocab_size:
        raise ValueError("step_trace logit count does not match checkpoint vocabulary")
    greedy = step["greedy_token_id"]
    if not _is_int(greedy) or not 0 <= greedy < vocab_size:
        raise ValueError("step_trace greedy token is outside checkpoint vocabulary")


def _validate_probe(
    probe: object,
    *,
    names: set[str],
    vocab_size: int,
    max_context_tokens: int,
) -> None:
    if not isinstance(probe, dict):
        raise TypeError("inference probe evidence must be an object")
    if set(probe) != _PROBE_KEYS:
        raise ValueError("inference probe evidence has missing or unknown fields")
    name = probe["name"]
    if not isinstance(name, str) or not name or name in names:
        raise ValueError("inference probe names must be unique non-empty strings")
    names.add(name)
    record_hash = _require_sha256(probe["record_sha256"], field="record_sha256")
    unhashed_probe = dict(probe)
    del unhashed_probe["record_sha256"]
    if _sha256(unhashed_probe) != record_hash:
        raise ValueError(f"inference probe {name!r} self-hash mismatch")

    config = _config_from_payload(probe["config"])
    prompt_ids = probe["prompt_token_ids"]
    generated_ids = probe["generated_token_ids"]
    step_trace = probe["step_trace"]
    if not isinstance(prompt_ids, list) or not prompt_ids or not all(
        _is_int(value) and 0 <= value < vocab_size for value in prompt_ids
    ):
        raise ValueError("prompt_token_ids evidence must be a non-empty in-vocab integer array")
    if len(prompt_ids) > max_context_tokens:
        raise ValueError("prompt_token_ids exceed checkpoint context limit")
    if not isinstance(generated_ids, list) or not all(
        _is_int(value) and 0 <= value < vocab_size for value in generated_ids
    ):
        raise ValueError("generated_token_ids evidence must be an in-vocab integer array")
    if len(generated_ids) > config.max_new_tokens:
        raise ValueError("generated token count exceeds generation config")
    if not isinstance(step_trace, list) or len(step_trace) != len(generated_ids):
        raise ValueError("step_trace length must equal generated token count")

    _require_sha256(probe["prompt_utf8_sha256"], field="prompt_utf8_sha256")
    _require_sha256(probe["prompt_token_ids_sha256"], field="prompt_token_ids_sha256")
    _require_sha256(
        probe["generated_token_ids_sha256"], field="generated_token_ids_sha256"
    )
    _require_sha256(probe["output_utf8_sha256"], field="output_utf8_sha256")
    _require_sha256(probe["step_trace_sha256"], field="step_trace_sha256")
    if _sha256(prompt_ids) != probe["prompt_token_ids_sha256"]:
        raise ValueError(f"inference probe {name!r} prompt-token hash mismatch")
    if _sha256(generated_ids) != probe["generated_token_ids_sha256"]:
        raise ValueError(f"inference probe {name!r} generated-token hash mismatch")
    if _sha256(step_trace) != probe["step_trace_sha256"]:
        raise ValueError(f"inference probe {name!r} step-trace hash mismatch")
    if probe["stop_reason"] not in _STOP_REASONS:
        raise ValueError("inference probe has unsupported stop_reason")

    for index, step in enumerate(step_trace):
        _validate_trace_step(
            step,
            index=index,
            prompt_length=len(prompt_ids),
            vocab_size=vocab_size,
            max_context_tokens=max_context_tokens,
        )


def validate_evidence(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise TypeError("inference evidence must be a JSON object")
    if set(payload) != _ROOT_KEYS:
        raise ValueError("inference evidence has missing or unknown top-level fields")
    if payload.get("schema") != EVIDENCE_SCHEMA:
        raise ValueError("unsupported inference evidence schema")
    expected_hash = _require_sha256(payload.get("evidence_sha256"), field="evidence_sha256")
    unhashed = dict(payload)
    del unhashed["evidence_sha256"]
    if _sha256(unhashed) != expected_hash:
        raise ValueError("inference evidence self-hash mismatch")

    checkpoint = _validate_diagnostics(payload["checkpoint"])
    probes = payload["probes"]
    if not isinstance(probes, list) or not probes:
        raise ValueError("inference evidence must contain at least one probe")
    names: set[str] = set()
    for probe in probes:
        _validate_probe(
            probe,
            names=names,
            vocab_size=checkpoint["vocab_size"],
            max_context_tokens=checkpoint["max_context_tokens"],
        )

    if payload["claims"] != _CLAIMS:
        raise ValueError("inference evidence truth-boundary claims were weakened")
    return payload


def replay_evidence(
    payload: object,
    backend: InferenceBackend,
) -> dict[str, object]:
    validated = validate_evidence(payload)
    current_diagnostics = _diagnostics(backend)
    if current_diagnostics != validated["checkpoint"]:
        raise ValueError("loaded backend identity does not match inference evidence")

    replayed = 0
    for expected in validated["probes"]:
        prompt_ids = expected["prompt_token_ids"]
        prompt = backend.decode(prompt_ids)
        if backend.encode(prompt) != prompt_ids:
            raise ValueError(f"probe {expected['name']!r} prompt token round-trip mismatch")
        if _text_sha256(prompt) != expected["prompt_utf8_sha256"]:
            raise ValueError(f"probe {expected['name']!r} prompt UTF-8 hash mismatch")
        actual = collect_probe(
            backend,
            name=expected["name"],
            prompt=prompt,
            config=_config_from_payload(expected["config"]),
        )
        if actual != expected:
            raise ValueError(f"probe {expected['name']!r} replay diverged")
        replayed += 1

    return {
        "schema": "12-6.first-party-inference-replay-result.v1",
        "passed": True,
        "checkpoint_id": current_diagnostics["checkpoint_id"],
        "probes_replayed": replayed,
        "evidence_sha256": validated["evidence_sha256"],
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.evidence",
        description="Collect or replay exact first-party raw-Base inference evidence.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--checkpoint", type=Path, required=True)
    collect.add_argument("--prompt", action="append", required=True)
    collect.add_argument("--max-new-tokens", type=int, default=8)
    collect.add_argument("--sample-seed", type=int)
    collect.add_argument("--output", type=Path)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--checkpoint", type=Path, required=True)
    replay.add_argument("--evidence", type=Path, required=True)
    replay.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        backend = load_first_party_backend(args.checkpoint)
        if args.command == "collect":
            probes: list[tuple[str, str, GenerationConfig]] = []
            for index, prompt in enumerate(args.prompt):
                probes.append(
                    (
                        f"greedy-{index}",
                        prompt,
                        GenerationConfig(max_new_tokens=args.max_new_tokens, sample=False),
                    )
                )
                if args.sample_seed is not None:
                    probes.append(
                        (
                            f"sample-{index}-seed-{args.sample_seed}",
                            prompt,
                            GenerationConfig(
                                max_new_tokens=args.max_new_tokens,
                                sample=True,
                                seed=args.sample_seed,
                            ),
                        )
                    )
            evidence = build_evidence(backend, probes)
            rendered = json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=2,
            )
            if args.output is None:
                print(rendered)
            else:
                args.output.write_text(rendered + "\n", encoding="utf-8")
                checkpoint = evidence["checkpoint"]
                print(
                    f"evidence: PASS checkpoint_id={checkpoint['checkpoint_id']} "
                    f"probes={len(evidence['probes'])} sha256={evidence['evidence_sha256']}"
                )
            return 0

        report = replay_evidence(_load_json(args.evidence), backend)
        if args.json:
            print(json.dumps(report, sort_keys=True, allow_nan=False))
        else:
            print(
                f"replay: PASS checkpoint_id={report['checkpoint_id']} "
                f"probes={report['probes_replayed']} "
                f"evidence_sha256={report['evidence_sha256']}"
            )
        return 0
    except (
        FileNotFoundError,
        FloatingPointError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
