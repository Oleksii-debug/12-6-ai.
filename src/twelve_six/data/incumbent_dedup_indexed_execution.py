"""Fail-closed trust boundary for the incumbent indexed-dedup executor.

The execution/science implementation is preserved byte-for-byte in the private core
module.  This facade only closes mutable stdlib transitive dependencies before the
core can reconstruct or execute incumbent matcher authority.
"""
from __future__ import annotations

from typing import Any

from twelve_six.data import _incumbent_dedup_indexed_execution_core as _core

# Preserve the complete existing module surface, including private helpers exercised
# by the exact-lineage qualification suite.  Behavior-bearing functions keep the
# same core globals; the attestation hooks below are installed into those globals.
for _name, _value in vars(_core).items():
    if not _name.startswith("__"):
        globals()[_name] = _value

_CORE_LOADER_ATTEST = _core._attest_loader_frozen_runtime_dependencies
_CORE_RUNTIME_ATTEST = _core.attest_incumbent_runtime

# json.dumps can take the common fast path through this mutable singleton even while
# json.dumps and JSONEncoder themselves remain unchanged.
_FROZEN_JSON_DEFAULT_ENCODER = json._default_encoder
_FROZEN_JSON_DEFAULT_ENCODER_STATE = _freeze_direct_behavior(json._default_encoder)
_FROZEN_TRANSITIVE_BEHAVIOR = (
    *_core._FROZEN_TRANSITIVE_BEHAVIOR,
    (
        "json",
        "_default_encoder",
        json,
        _FROZEN_JSON_DEFAULT_ENCODER_STATE,
    ),
)
_core._FROZEN_TRANSITIVE_BEHAVIOR = _FROZEN_TRANSITIVE_BEHAVIOR

# re._compile delegates to a mutable compiler module and a mutable cache.  Bind the
# concrete compiler behavior and cache/capacity identities, but intentionally do not
# treat cache contents as authority: verified contents are discarded before use.
_FROZEN_RE_COMPILER = re._compiler
_FROZEN_RE_COMPILER_COMPILE = _freeze_direct_behavior(re._compiler.compile)
_FROZEN_RE_COMPILER_ISSTRING = _freeze_direct_behavior(re._compiler.isstring)
_FROZEN_RE_CACHE = re._cache
_FROZEN_RE_MAXCACHE = re._MAXCACHE


def _attest_loader_frozen_runtime_dependencies() -> None:
    """Extend the existing loader attester over json/re transitive mutable state."""
    _CORE_LOADER_ATTEST()

    compiler = getattr(re, "_compiler", None)
    if compiler is not _FROZEN_RE_COMPILER:
        raise IndexedExecutionError("transitive behavior drift: re._compiler")
    if not _direct_behavior_state_matches(
        getattr(compiler, "compile", None), _FROZEN_RE_COMPILER_COMPILE
    ):
        raise IndexedExecutionError("transitive behavior drift: re._compiler.compile")
    if not _direct_behavior_state_matches(
        getattr(compiler, "isstring", None), _FROZEN_RE_COMPILER_ISSTRING
    ):
        raise IndexedExecutionError("transitive behavior drift: re._compiler.isstring")

    cache = getattr(re, "_cache", None)
    if type(cache) is not dict or cache is not _FROZEN_RE_CACHE:
        raise IndexedExecutionError("transitive behavior drift: re._cache")
    maxcache = getattr(re, "_MAXCACHE", None)
    if (
        type(_FROZEN_RE_MAXCACHE) is not int
        or type(maxcache) is not int
        or maxcache != _FROZEN_RE_MAXCACHE
    ):
        raise IndexedExecutionError("transitive behavior drift: re._MAXCACHE")


def _neutralize_verified_re_cache() -> None:
    """Verify regex runtime identity before clearing non-authoritative cache contents."""
    _attest_loader_frozen_runtime_dependencies()
    _FROZEN_RE_CACHE.clear()


def attest_incumbent_runtime(v3: Any) -> None:
    """Attest exact incumbent authority with two-phase regex-cache neutralization."""
    _neutralize_verified_re_cache()
    _CORE_RUNTIME_ATTEST(v3)
    # Source reconstruction can legitimately populate the trusted cache object.
    # Clear it again so neither the reference nor indexed execution inherits state.
    _neutralize_verified_re_cache()


# Install the hardened hooks into the byte-preserved core globals.  Existing core
# functions such as audit_payloads_indexed and _attest_executable_module therefore
# resolve the hardened attesters without duplicating any executor/science logic.
_core._attest_loader_frozen_runtime_dependencies = _attest_loader_frozen_runtime_dependencies
_core.attest_incumbent_runtime = attest_incumbent_runtime
