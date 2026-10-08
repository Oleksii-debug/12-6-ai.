"""12-6 owned agent-facing reference runtime (Plan 5; no Nika product state)."""

from .context import ContextEntry, ContextError, build_context, canonical_bytes

__all__ = ["ContextEntry", "ContextError", "build_context", "canonical_bytes"]
