"""Nika Core's synchronous local provider protocol over the pinned ModelGateway."""
from __future__ import annotations

import re
from collections.abc import Callable
from http.server import ThreadingHTTPServer
from typing import Any

from tools.inference_runtime import GenerationConfig
from tools.plan4_model_gateway import SCHEMA, ModelGateway
from tools.plan4_model_service import _serve_json_loopback

_ROLES = frozenset({"system", "user", "assistant", "tool"})
_REQUEST_FIELDS = frozenset({"model", "messages", "temperature"})
_IDENTITY_FIELDS = ("model_weights_sha256", "model_spec_sha256", "tokenizer_sha256")


def _gateway_request(op: str, **args: object) -> dict[str, object]:
    return {"schema": SCHEMA, "op": op, "args": args}


def _messages(request: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    messages = request.get("messages")
    if type(messages) is not list or not messages:
        raise ValueError("messages required")
    parsed = []
    for message in messages:
        if type(message) is not dict or set(message) != {"role", "content"}:
            raise ValueError("invalid message fields")
        role, content = message["role"], message["content"]
        if type(role) is not str or role not in _ROLES:
            raise ValueError("invalid message role")
        if type(content) is not str or not content.strip():
            raise ValueError("invalid message content")
        parsed.append((role, content))
    return tuple(parsed)


def serve_nika_loopback(
    gateway: ModelGateway,
    *,
    provider: str,
    model: str,
    expected_weights_sha256: str,
    render_messages: Callable[[tuple[tuple[str, str], ...]], str],
    generation_config: GenerationConfig,
    bearer_token: str,
) -> ThreadingHTTPServer:
    """Bind the peer's existing chat-completions client to one host-selected model."""
    if not isinstance(gateway, ModelGateway) or not callable(render_messages):
        raise TypeError("ModelGateway and the selected model's message renderer are required")
    if not isinstance(generation_config, GenerationConfig) or generation_config.max_new_tokens == 0:
        raise ValueError("a positive, explicit generation profile is required")
    description = gateway.dispatch(_gateway_request("describe", provider=provider, model=model))
    if not description["ok"]:
        raise ValueError("Nika requires a registered ModelGateway route")
    route = description["result"]
    pinned = route["identity"]
    if (
        route["transport"] not in {"local", "server"}
        or "generate" not in route["capabilities"]
        or pinned["model_weights_sha256"] != expected_weights_sha256
    ):
        raise ValueError("Nika route differs from the selected local generation model")
    config = generation_config.to_dict()
    temperature = 0.0 if generation_config.strategy == "greedy" else generation_config.temperature

    def dispatch(request: object) -> tuple[int, object]:
        if type(request) is not dict or set(request) - _REQUEST_FIELDS:
            return 400, {"error": {"code": "INVALID_REQUEST"}}
        if type(request.get("model")) is not str:
            return 400, {"error": {"code": "INVALID_REQUEST"}}
        if request["model"] != model:
            return 404, {"error": {"code": "UNKNOWN_MODEL"}}
        if "temperature" in request:
            requested = request["temperature"]
            if type(requested) not in (int, float) or requested != temperature:
                return 400, {"error": {"code": "UNSUPPORTED_TEMPERATURE"}}
        try:
            messages = _messages(request)
        except ValueError:
            return 400, {"error": {"code": "INVALID_MESSAGES"}}
        try:
            prompt = render_messages(messages)
            if type(prompt) is not str or not prompt:
                raise ValueError("invalid rendered prompt")
        except Exception:  # noqa: BLE001 - renderer errors stay inside the host boundary
            return 503, {"error": {"code": "MESSAGE_RENDERING_FAILED"}}
        response = gateway.dispatch(_gateway_request(
            "generate", provider=provider, model=model, prompt=prompt, config=config,
        ))
        if not response["ok"]:
            return 503, {"error": {"code": response["error"]["code"]}}
        result = response["result"]
        usage = result["usage"]
        if result["identity"] != pinned or usage is None:
            return 503, {"error": {"code": "MODEL_IDENTITY_MISMATCH"}}
        generation = result["data"].get("generation")
        if (
            type(generation) is not dict
            or generation.get("status") != "COMPLETED"
            or type(generation.get("text")) is not str
            or type(generation.get("request_id")) is not str
            or type(generation.get("result_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", generation["result_sha256"]) is None
        ):
            return 503, {"error": {"code": "INVALID_GENERATION_RESULT"}}
        if any(generation.get(field) != pinned[field] for field in _IDENTITY_FIELDS):
            return 503, {"error": {"code": "MODEL_IDENTITY_MISMATCH"}}
        return 200, {
            "id": generation["request_id"],
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": generation["text"]}}],
            "usage": {
                "prompt_tokens": usage["input_tokens"],
                "completion_tokens": usage["output_tokens"],
                "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            },
            "twelve_six": {"identity": pinned, "result_sha256": generation["result_sha256"]},
        }

    return _serve_json_loopback("/v1/chat/completions", dispatch, bearer_token=bearer_token)
