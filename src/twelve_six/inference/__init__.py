from importlib import import_module

from .contracts import GenerationConfig, GenerationResult, InferenceBackend
from .generation import CacheMode, generate, generate_token_ids

_LAZY_API_EXPORTS = frozenset(
    {
        "FirstPartyInference",
        "load_first_party_inference",
        "load_random_init_inference",
    }
)


def __getattr__(name: str) -> object:
    """Load high-level first-party API only when a caller asks for it.

    ``integration.s0_runtime`` imports the low-level ``inference.static_kv`` module.
    Eagerly importing ``inference.api`` while the package initializes would route
    back through ``first_party`` into the still-initializing runtime. Keep the
    public high-level exports intact without making low-level package import cyclic.
    """

    if name in _LAZY_API_EXPORTS:
        module = import_module(".api", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CacheMode",
    "FirstPartyInference",
    "GenerationConfig",
    "GenerationResult",
    "InferenceBackend",
    "generate",
    "generate_token_ids",
    "load_first_party_inference",
    "load_random_init_inference",
]
