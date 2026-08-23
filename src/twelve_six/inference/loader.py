from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import cast

from .contracts import InferenceBackend

BackendLoader = Callable[[Path], InferenceBackend]


def load_backend(loader_spec: str, checkpoint: Path) -> InferenceBackend:
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    module_name, separator, attribute_name = loader_spec.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("backend loader must use MODULE:CALLABLE syntax")

    module = importlib.import_module(module_name)
    loader = getattr(module, attribute_name, None)
    if loader is None or not callable(loader):
        raise ValueError(f"backend loader is not callable: {loader_spec}")

    backend = cast(BackendLoader, loader)(checkpoint)
    if not isinstance(backend, InferenceBackend):
        raise TypeError(
            "backend must provide eos_token_id, max_context_tokens, encode(), decode(), "
            "and next_token_logits()"
        )
    if not isinstance(backend.max_context_tokens, int) or backend.max_context_tokens < 1:
        raise ValueError("backend max_context_tokens must be a positive integer")
    return backend
