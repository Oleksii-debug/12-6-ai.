"""Machine-verifiable acceptance evidence for first-party 12-6 inference.

The collector composes the existing checkpoint loader, generation harness,
backend parity harness, and local raw-completions server. It deliberately does
not implement another model adapter or sampling path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from http.client import HTTPConnection
from pathlib import Path

from .contracts import GenerationConfig, GenerationResult, InferenceBackend
from .first_party import load_first_party_backend
from .generation import generate
from .openai_compat import completion_response
from .parity import compare_backends
from .server import make_server

ACCEPTANCE_SCHEMA = "12-6.inference-acceptance.v1"
ACCEPTANCE_AUTHORITY = "LOCAL_FREE_INFERENCE_EVIDENCE_NOT_PROMOTION"
_REQUIRED_DIAGNOSTIC_KEYS = (
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
)
_IDENTITY_KEYS = (
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
)


class InferenceAcceptanceError(RuntimeError):
    """Raised when an inference acceptance requirement fails closed."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json(value))


def _diagnostics(backend: object) -> dict[str, object]:
    provider = getattr(backend, "diagnostics", None)
    if not callable(provider):
        raise InferenceAcceptanceError("backend must expose callable diagnostics()")
    payload = provider()
    if not isinstance(payload, dict):
        raise InferenceAcceptanceError("backend diagnostics must be a dictionary")
    missing = [key for key in _REQUIRED_DIAGNOSTIC_KEYS if key not in payload]
    if missing:
        raise InferenceAcceptanceError(
            f"backend diagnostics missing required field(s): {', '.join(missing)}"
        )
    return payload


def _require_same_identity(
    reference: Mapping[str, object], candidate: Mapping[str, object]
) -> dict[str, object]:
    mismatches = {
        key: {"reference": reference.get(key), "candidate": candidate.get(key)}
        for key in _IDENTITY_KEYS
        if reference.get(key) != candidate.get(key)
    }
    if mismatches:
        raise InferenceAcceptanceError(
            f"reloaded backend identity mismatch: {json.dumps(mismatches, sort_keys=True)}"
        )
    return {key: reference[key] for key in _IDENTITY_KEYS}


def _generation_digest(result: GenerationResult) -> dict[str, object]:
    """Summarize a generation without persisting prompt or generated text."""

    return {
        "prompt_tokens": len(result.prompt_token_ids),
        "generated_tokens": len(result.generated_token_ids),
        "generated_token_ids_sha256": _sha256_json(list(result.generated_token_ids)),
        "text_sha256": _sha256_bytes(result.text.encode("utf-8")),
        "stop_reason": result.stop_reason,
    }


def _request_json(
    address: tuple[str, int],
    method: str,
    path: str,
    payload: object | None = None,
) -> tuple[int, dict[str, object]]:
    body = None if payload is None else _canonical_json(payload)
    headers = {} if body is None else {"Content-Type": "application/json"}
    connection = HTTPConnection(*address, timeout=10)
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read()
        status = response.status
    finally:
        connection.close()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InferenceAcceptanceError("server response was not valid UTF-8 JSON") from exc
    if not isinstance(parsed, dict):
        raise InferenceAcceptanceError("server response JSON must be an object")
    return status, parsed


def _completion_semantics(payload: Mapping[str, object]) -> dict[str, object]:
    choices = payload.get("choices")
    usage = payload.get("usage")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise InferenceAcceptanceError("completion response must contain exactly one choice")
    if not isinstance(usage, dict):
        raise InferenceAcceptanceError("completion response must contain usage")
    choice = choices[0]
    text = choice.get("text")
    if not isinstance(text, str):
        raise InferenceAcceptanceError("completion choice text must be a string")
    return {
        "object": payload.get("object"),
        "model": payload.get("model"),
        "finish_reason": choice.get("finish_reason"),
        "text_sha256": _sha256_bytes(text.encode("utf-8")),
        "usage": dict(usage),
    }


def _assert_context_rejection(backend: InferenceBackend) -> dict[str, object]:
    probe = "x" * (backend.max_context_tokens + 1)
    encoded = backend.encode(probe)
    if len(encoded) <= backend.max_context_tokens:
        raise InferenceAcceptanceError(
            "canonical context probe did not exceed backend max_context_tokens"
        )
    try:
        generate(backend, probe, GenerationConfig(max_new_tokens=1))
    except ValueError as exc:
        if "max_context_tokens" not in str(exc):
            raise InferenceAcceptanceError(
                "oversized context failed for an unexpected reason"
            ) from exc
        return {
            "passed": True,
            "context_limit": backend.max_context_tokens,
            "probe_tokens": len(encoded),
        }
    raise InferenceAcceptanceError("oversized prompt was not rejected")


def _collect_http_parity(
    backend: InferenceBackend,
    prompt: str,
    *,
    seed: int,
    max_new_tokens: int,
    model_name: str,
) -> dict[str, object]:
    server = make_server(
        backend,
        host="127.0.0.1",
        port=0,
        model_name=model_name,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    address = (str(host), int(port))
    try:
        health_status, health = _request_json(address, "GET", "/healthz")
        models_status, models = _request_json(address, "GET", "/v1/models")
        if health_status != 200 or health.get("model") != model_name:
            raise InferenceAcceptanceError("HTTP health identity check failed")
        model_rows = models.get("data")
        if (
            models_status != 200
            or not isinstance(model_rows, list)
            or len(model_rows) != 1
            or not isinstance(model_rows[0], dict)
            or model_rows[0].get("id") != model_name
        ):
            raise InferenceAcceptanceError("HTTP model identity check failed")

        requests = {
            "greedy": {
                "model": model_name,
                "prompt": prompt,
                "temperature": 0,
                "max_tokens": max_new_tokens,
                "seed": seed,
            },
            "seeded_sample": {
                "model": model_name,
                "prompt": prompt,
                "temperature": 0.8,
                "top_p": 0.95,
                "max_tokens": max_new_tokens,
                "seed": seed,
            },
        }
        parity: dict[str, object] = {}
        for label, request_payload in requests.items():
            expected = completion_response(
                backend,
                request_payload,
                response_id="cmpl-offline-acceptance",
                created=0,
                model_name=model_name,
            )
            status, actual = _request_json(
                address,
                "POST",
                "/v1/completions",
                request_payload,
            )
            if status != 200:
                raise InferenceAcceptanceError(
                    f"HTTP {label} completion failed with status {status}"
                )
            expected_semantics = _completion_semantics(expected)
            actual_semantics = _completion_semantics(actual)
            if expected_semantics != actual_semantics:
                raise InferenceAcceptanceError(
                    f"offline/HTTP {label} completion semantics diverged"
                )
            parity[label] = {
                "passed": True,
                "semantics_sha256": _sha256_json(actual_semantics),
                "finish_reason": actual_semantics["finish_reason"],
                "usage": actual_semantics["usage"],
            }

        wrong_model_status, _ = _request_json(
            address,
            "POST",
            "/v1/completions",
            {"model": "wrong-model", "prompt": prompt, "temperature": 0},
        )
        chat_status, _ = _request_json(
            address,
            "POST",
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": prompt}]},
        )
        context_probe = "x" * (backend.max_context_tokens + 1)
        context_status, context_error = _request_json(
            address,
            "POST",
            "/v1/completions",
            {
                "model": model_name,
                "prompt": context_probe,
                "temperature": 0,
                "max_tokens": 1,
            },
        )
        if wrong_model_status != 400:
            raise InferenceAcceptanceError("HTTP wrong-model request did not fail closed")
        if chat_status != 404:
            raise InferenceAcceptanceError("HTTP chat request did not fail closed")
        if context_status != 400:
            raise InferenceAcceptanceError("HTTP oversized context did not fail closed")
        error_payload = context_error.get("error")
        if not isinstance(error_payload, dict) or "max_context_tokens" not in str(
            error_payload.get("message")
        ):
            raise InferenceAcceptanceError(
                "HTTP oversized context returned the wrong error contract"
            )

        return {
            "loopback_only": address[0] == "127.0.0.1",
            "health_identity": True,
            "model_identity": True,
            "offline_http_parity": parity,
            "wrong_model_rejected": True,
            "chat_semantics_rejected": True,
            "oversized_context_rejected": True,
        }
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        if thread.is_alive():
            raise InferenceAcceptanceError("local acceptance server did not stop cleanly")


def collect_backend_acceptance(
    reference: InferenceBackend,
    reloaded: InferenceBackend,
    prompts: Sequence[str],
    *,
    seed: int = 20260825,
    max_new_tokens: int = 8,
    model_name: str = "12-6-base",
) -> dict[str, object]:
    """Collect fail-closed inference evidence for two independently loaded backends."""

    if not prompts or any(not isinstance(prompt, str) or not prompt for prompt in prompts):
        raise ValueError("prompts must contain at least one non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens < 1
    ):
        raise ValueError("max_new_tokens must be a positive integer")
    if not model_name:
        raise ValueError("model_name must not be empty")

    reference_diagnostics = _diagnostics(reference)
    reloaded_diagnostics = _diagnostics(reloaded)
    identity = _require_same_identity(reference_diagnostics, reloaded_diagnostics)

    parity = compare_backends(
        reference,
        reloaded,
        tuple(prompts),
        max_new_tokens=max_new_tokens,
        atol=0.0,
        rtol=0.0,
    )
    if not parity.passed:
        raise InferenceAcceptanceError(
            f"exact reload logit/token/decode parity failed: {parity.to_dict()}"
        )

    greedy_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        sample=False,
        seed=seed,
    )
    sample_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        sample=True,
        temperature=0.8,
        top_p=0.95,
        seed=seed,
    )
    prompt_evidence: list[dict[str, object]] = []
    for prompt in prompts:
        reference_greedy = generate(reference, prompt, greedy_config)
        reloaded_greedy = generate(reloaded, prompt, greedy_config)
        if reference_greedy != reloaded_greedy:
            raise InferenceAcceptanceError("greedy generation changed after checkpoint reload")

        first_sample = generate(reference, prompt, sample_config)
        repeated_sample = generate(reference, prompt, sample_config)
        reloaded_sample = generate(reloaded, prompt, sample_config)
        if first_sample != repeated_sample or first_sample != reloaded_sample:
            raise InferenceAcceptanceError(
                "seeded sampling is not repeatable across repeated/reloaded execution"
            )

        prompt_evidence.append(
            {
                "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "prompt_utf8_bytes": len(prompt.encode("utf-8")),
                "prompt_tokens": len(reference_greedy.prompt_token_ids),
                "greedy": _generation_digest(reference_greedy),
                "seeded_sample": _generation_digest(first_sample),
            }
        )

    context = _assert_context_rejection(reloaded)
    http = _collect_http_parity(
        reloaded,
        prompts[0],
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
    )

    report: dict[str, object] = {
        "schema": ACCEPTANCE_SCHEMA,
        "authority": ACCEPTANCE_AUTHORITY,
        "passed": True,
        "checkpoint": identity,
        "config": {
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "logit_atol": 0.0,
            "logit_rtol": 0.0,
            "model_name": model_name,
        },
        "prompts": prompt_evidence,
        "reload_parity": parity.to_dict(),
        "context": context,
        "http": http,
        "claims": {
            "raw_base_completion_semantics": True,
            "hidden_prompt_or_chat_template": False,
            "promotion_authority": False,
            "windows_nvda_live_execution": False,
        },
    }
    report["evidence_sha256"] = _sha256_json(report)
    validate_acceptance_report(report)
    return report


def collect_checkpoint_acceptance(
    checkpoint: str | Path,
    prompts: Sequence[str],
    *,
    seed: int = 20260825,
    max_new_tokens: int = 8,
    model_name: str = "12-6-base",
) -> dict[str, object]:
    """Load one verified checkpoint twice and collect D05/D07 acceptance evidence."""

    checkpoint = Path(checkpoint)
    reference = load_first_party_backend(checkpoint)
    reloaded = load_first_party_backend(checkpoint)
    return collect_backend_acceptance(
        reference,
        reloaded,
        prompts,
        seed=seed,
        max_new_tokens=max_new_tokens,
        model_name=model_name,
    )


def validate_acceptance_report(report: Mapping[str, object]) -> None:
    """Reject stale, incomplete, failed, or tampered acceptance evidence."""

    if report.get("schema") != ACCEPTANCE_SCHEMA:
        raise InferenceAcceptanceError("unsupported inference acceptance schema")
    if report.get("authority") != ACCEPTANCE_AUTHORITY:
        raise InferenceAcceptanceError("inference acceptance authority boundary mismatch")
    if report.get("passed") is not True:
        raise InferenceAcceptanceError("inference acceptance report is not PASS")
    expected_hash = report.get("evidence_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise InferenceAcceptanceError("inference acceptance evidence SHA-256 is missing")
    unhashed = dict(report)
    unhashed.pop("evidence_sha256", None)
    if _sha256_json(unhashed) != expected_hash:
        raise InferenceAcceptanceError("inference acceptance evidence SHA-256 mismatch")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise InferenceAcceptanceError("inference acceptance checkpoint identity is missing")
    missing = [key for key in _IDENTITY_KEYS if key not in checkpoint]
    if missing:
        raise InferenceAcceptanceError(
            f"inference acceptance checkpoint identity incomplete: {', '.join(missing)}"
        )
    parity = report.get("reload_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise InferenceAcceptanceError("reload parity is not PASS")
    claims = report.get("claims")
    if not isinstance(claims, Mapping) or claims.get("promotion_authority") is not False:
        raise InferenceAcceptanceError("promotion authority boundary is missing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m twelve_six.inference.acceptance",
        description=(
            "Collect machine-verifiable first-party checkpoint/reload/generation/HTTP "
            "acceptance evidence without persisting prompt text."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="repeatable raw Base prompt; defaults to '12-6' when omitted",
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--model-name", default="12-6-base")
    parser.add_argument("--output", type=Path, help="write JSON evidence to a new file")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompts = tuple(args.prompt) if args.prompt else ("12-6",)
    try:
        report = collect_checkpoint_acceptance(
            args.checkpoint,
            prompts,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            model_name=args.model_name,
        )
        encoded = json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"
        if args.output is None:
            print(encoded, end="")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            print(
                f"inference-acceptance: PASS evidence={report['evidence_sha256']} "
                f"output={args.output}",
                file=os.sys.stderr,
            )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"inference-acceptance: FAIL error={exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
