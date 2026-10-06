"""Fail-closed learned-20M decision authority for the canonical byte tokenizer."""

from __future__ import annotations

import builtins
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import CodeType, FunctionType, MappingProxyType
from typing import Any

from twelve_six.data.balanced_split_application_v1 import (
    APPLICATION_SCHEMA,
    CANONICAL_SPLIT_GIT_BLOB_SHA1,
    CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
    SELECTION_SCHEMA,
    BalancedSplitApplicationError,
    verify_balanced_selection,
)

from . import base as base_module
from . import byte as byte_module
from .base import TokenizerIdentity as _CanonicalTokenizerIdentity
from .byte import ByteTokenizer

SCHEMA = "12-6.d04-learned20m-tokenizer-decision.v1"
DECISION = "RETAIN_BYTE_BASELINE"
STATUS = "TERMINAL_TOKENIZER_DECISION_ZERO_CREDIT"
CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1 = "ee21cc40ba07d6e3bc82b20a8642658284f74959"
_BYTE_TOKENIZER_SOURCE_PATH = Path(__file__).with_name("byte.py")

_UPSTREAM_IDENTITY_FIELDS = (
    "retained_inventory_identity_sha256",
    "decontamination_authority_sha256",
    "dedup_authority_sha256",
    "balance_policy_identity_sha256",
    "balance_result_identity_sha256",
)
_APPLICATION_KEYS = {
    "schema",
    "status",
    "balanced_selection_identity_sha256",
    *_UPSTREAM_IDENTITY_FIELDS,
    "canonical_split_git_blob_sha1",
    "split_spec_identity_sha256",
    "selected_record_count",
    "selected_source_bytes",
    "selected_family_source_bytes",
    "selected_stratum_source_bytes",
    "split_family",
    "claim_boundary",
    "application_identity_sha256",
}
_REPORT_KEYS = {
    "schema",
    "status",
    "decision",
    "balanced_selection_identity_sha256",
    "split_application_identity_sha256",
    *_UPSTREAM_IDENTITY_FIELDS,
    "canonical_split_git_blob_sha1",
    "split_spec_identity_sha256",
    "canonical_byte_tokenizer_git_blob_sha1",
    "tokenizer_version",
    "tokenizer_config_sha256",
    "tokenizer_vocab_sha256",
    "vocab_size",
    "normalization",
    "encoding",
    "tokenizer_fit_executed",
    "training_authorized_by_this_report",
    "compute_authorized_by_this_report",
    "authorized_optimized_target_exposure",
    "decision_identity_sha256",
}
_ZERO_CREDIT_BOUNDARY = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "paid_compute_authorized": False,
    "final_test_outcomes_read": False,
    "authorized_optimized_target_exposure": 0,
}
# These values are intentionally literal and independent of the already-loaded
# byte module. The checked source blob is one authority; mutable Python module
# globals must not be able to redefine the expected runtime baseline before
# this module is imported or reloaded.
_EXPECTED_TOKENIZER_RUNTIME_IDENTITY = {
    "version": "s0-byte-v1",
    "config_sha256": "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1",
    "vocab_sha256": "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571",
    "vocab_size": 256,
    "normalization": "none",
    "encoding": "utf-8",
    "special_tokens": {},
}
_EXPECTED_TOKENIZER_CLASS_STATE = {
    "pad_id": None,
    "bos_id": None,
    "eos_id": None,
    "byte_offset": 0,
    "version": "s0-byte-v1",
    "vocab_size": 256,
    "normalization": "none",
    "encoding": "utf-8",
}
_EXPECTED_TOKENIZER_METHOD_KWDEFAULTS = {
    "__init__": None,
    "identity": None,
    "encode": {"add_bos": False, "add_eos": False},
    "decode": {"skip_special_tokens": True, "errors": "strict"},
    "oov_count": None,
    "fertility": None,
}
_EXPECTED_BYTE_MODULE_CONSTANTS = {
    "BYTE_TOKENIZER_VERSION": "s0-byte-v1",
    "BYTE_TOKENIZER_HASH": "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1",
    "BYTE_VOCAB_HASH": "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571",
}
_EXPECTED_BYTE_MODULE_CONFIG = {
    "schema_version": 1,
    "tokenizer_version": "s0-byte-v1",
    "type": "utf8-byte",
    "normalization": "none",
    "encoding": "utf-8",
    "special_tokens": {},
    "byte_offset": 0,
    "byte_values": 256,
    "vocab_size": 256,
}

# Freeze ambient call targets used by source-pinned byte.py and by this verifier.
# Function.__builtins__ identity alone is insufficient because the shared builtins
# mapping is mutable, including helpers such as compile/vars/type used to prove the
# live runtime still matches the pinned source.
_EXPECTED_BUILTINS_MODULE = builtins
_EXPECTED_BYTE_RUNTIME_BUILTINS = {
    "RuntimeError": builtins.RuntimeError,
    "TypeError": builtins.TypeError,
    "ValueError": builtins.ValueError,
    "all": builtins.all,
    "any": builtins.any,
    "bytearray": builtins.bytearray,
    "bytes": builtins.bytes,
    "compile": builtins.compile,
    "dict": builtins.dict,
    "getattr": builtins.getattr,
    "int": builtins.int,
    "isinstance": builtins.isinstance,
    "len": builtins.len,
    "list": builtins.list,
    "object": builtins.object,
    "open": builtins.open,
    "range": builtins.range,
    "set": builtins.set,
    "str": builtins.str,
    "type": builtins.type,
    "vars": builtins.vars,
}
_EXPECTED_BYTE_SOURCE_PATH = _BYTE_TOKENIZER_SOURCE_PATH
_EXPECTED_BYTE_SOURCE_PATH_TEXT = str(_BYTE_TOKENIZER_SOURCE_PATH)
_EXPECTED_BYTE_SOURCE_PATH_CLASS = type(_BYTE_TOKENIZER_SOURCE_PATH)
_EXPECTED_BYTE_SOURCE_READ_BYTES = _EXPECTED_BYTE_SOURCE_PATH_CLASS.read_bytes
_EXPECTED_TOKENIZER_BASE_MODULE = base_module
_EXPECTED_TOKENIZER_IDENTITY_CLASS = _CanonicalTokenizerIdentity
_EXPECTED_BYTE_JSON_MODULE = json
_EXPECTED_BYTE_HASHLIB_MODULE = hashlib
_EXPECTED_BYTE_JSON_DUMPS = json.dumps
_EXPECTED_BYTE_HASHLIB_SHA1 = hashlib.sha1
_EXPECTED_BYTE_HASHLIB_SHA256 = hashlib.sha256


class TokenizerDecisionError(ValueError):
    """Raised when terminal tokenizer-decision evidence fails closed."""


def _canonical_json(value: Mapping[str, Any]) -> str:
    try:
        rendered = json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        # UTF-8 is the identity encoding; lone surrogates are not valid evidence.
        rendered.encode("utf-8")
        return rendered
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TokenizerDecisionError("authority must be strict UTF-8 JSON") from exc


def authority_sha256(value: Mapping[str, Any]) -> str:
    """Return the SHA-256 identity of a canonical JSON mapping."""

    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _self_hash(value: Mapping[str, Any], identity_field: str) -> str:
    core = dict(value)
    core.pop(identity_field, None)
    return authority_sha256(core)


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    if value != value.lower() or any(ch not in "0123456789abcdef" for ch in value):
        raise TokenizerDecisionError(f"{field} must be a lowercase SHA-256 hex string")
    return value


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _read_canonical_byte_tokenizer_source() -> bytes:
    """Read canonical source without mutable Path instance dispatch."""

    _verify_byte_tokenizer_runtime_dependencies()
    try:
        with _EXPECTED_BYTE_RUNTIME_BUILTINS["open"](
            _EXPECTED_BYTE_SOURCE_PATH_TEXT,
            "rb",
        ) as source_file:
            payload = source_file.read()
    except OSError as exc:
        raise TokenizerDecisionError(
            "cannot read canonical byte tokenizer implementation"
        ) from exc
    _verify_byte_tokenizer_runtime_dependencies()
    if type(payload) is not bytes:
        raise TokenizerDecisionError(
            "canonical byte tokenizer implementation reader returned non-bytes"
        )
    return payload


def _canonical_byte_tokenizer_git_blob_sha1() -> str:
    return _git_blob_sha1(_read_canonical_byte_tokenizer_source())


def _verify_canonical_byte_tokenizer_implementation() -> str:
    observed = _canonical_byte_tokenizer_git_blob_sha1()
    if observed != CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1:
        raise TokenizerDecisionError("canonical byte tokenizer implementation identity drift")
    return observed


def _verified_canonical_byte_tokenizer_method_codes() -> dict[str, CodeType]:
    """Compile the pinned source without executing it and bind live method code."""

    payload = _read_canonical_byte_tokenizer_source()
    if _git_blob_sha1(payload) != CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1:
        raise TokenizerDecisionError(
            "canonical byte tokenizer implementation identity drift"
        )
    try:
        module_code = _EXPECTED_BYTE_RUNTIME_BUILTINS["compile"](
            payload,
            str(_BYTE_TOKENIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        raise TokenizerDecisionError(
            "cannot compile canonical byte tokenizer implementation"
        ) from exc
    class_codes = [
        value
        for value in module_code.co_consts
        if isinstance(value, CodeType) and value.co_name == "ByteTokenizer"
    ]
    if len(class_codes) != 1:
        raise TokenizerDecisionError(
            "canonical byte tokenizer class code identity unavailable"
        )
    expected_names = {
        "__init__",
        "identity",
        "encode",
        "decode",
        "oov_count",
        "fertility",
    }
    methods = {
        value.co_name: value
        for value in class_codes[0].co_consts
        if isinstance(value, CodeType) and value.co_name in expected_names
    }
    if set(methods) != expected_names:
        raise TokenizerDecisionError(
            "canonical byte tokenizer method code identity unavailable"
        )
    return methods


def _verified_canonical_byte_tokenizer_helper_codes() -> dict[str, CodeType]:
    """Compile the pinned source and return behavior-bearing module helper code."""

    payload = _read_canonical_byte_tokenizer_source()
    if _git_blob_sha1(payload) != CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1:
        raise TokenizerDecisionError(
            "canonical byte tokenizer implementation identity drift"
        )
    try:
        module_code = _EXPECTED_BYTE_RUNTIME_BUILTINS["compile"](
            payload,
            str(_BYTE_TOKENIZER_SOURCE_PATH),
            "exec",
            dont_inherit=True,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        raise TokenizerDecisionError(
            "cannot compile canonical byte tokenizer implementation"
        ) from exc
    expected_names = {
        "canonical_config_json",
        "tokenizer_config_hash",
        "canonical_vocab_json",
        "vocab_hash",
    }
    helpers = {
        value.co_name: value
        for value in module_code.co_consts
        if isinstance(value, CodeType) and value.co_name in expected_names
    }
    if set(helpers) != expected_names:
        raise TokenizerDecisionError(
            "canonical byte tokenizer helper code identity unavailable"
        )
    return helpers


def _verify_byte_tokenizer_runtime_dependencies() -> None:
    """Bind mutable interpreter and stdlib dependencies used by D04 verification."""

    if builtins is not _EXPECTED_BUILTINS_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: builtins module"
        )
    builtins_state = _EXPECTED_BUILTINS_MODULE.__dict__
    for name, expected in _EXPECTED_BYTE_RUNTIME_BUILTINS.items():
        if builtins_state.get(name) is not expected:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime dependency drift: builtins.{name}"
            )
    if _BYTE_TOKENIZER_SOURCE_PATH is not _EXPECTED_BYTE_SOURCE_PATH:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: source path"
        )
    if (
        type(_BYTE_TOKENIZER_SOURCE_PATH) is not _EXPECTED_BYTE_SOURCE_PATH_CLASS
        or _EXPECTED_BYTE_SOURCE_PATH_CLASS.read_bytes
        is not _EXPECTED_BYTE_SOURCE_READ_BYTES
    ):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: source reader"
        )
    if base_module is not _EXPECTED_TOKENIZER_BASE_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: base module"
        )
    if json is not _EXPECTED_BYTE_JSON_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: json module"
        )
    if hashlib is not _EXPECTED_BYTE_HASHLIB_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: hashlib module"
        )
    if json.dumps is not _EXPECTED_BYTE_JSON_DUMPS:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: json.dumps"
        )
    if hashlib.sha1 is not _EXPECTED_BYTE_HASHLIB_SHA1:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: hashlib.sha1"
        )
    if hashlib.sha256 is not _EXPECTED_BYTE_HASHLIB_SHA256:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime dependency drift: hashlib.sha256"
        )


def _verify_runtime_byte_tokenizer_module_state() -> None:
    """Bind mutable module state used by the source-pinned tokenizer runtime."""

    module_state = _EXPECTED_BYTE_RUNTIME_BUILTINS["vars"](byte_module)
    if module_state.get("ByteTokenizer") is not ByteTokenizer:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime module drift: ByteTokenizer"
        )
    if (
        base_module.TokenizerIdentity is not _EXPECTED_TOKENIZER_IDENTITY_CLASS
        or _CanonicalTokenizerIdentity is not _EXPECTED_TOKENIZER_IDENTITY_CLASS
        or module_state.get("TokenizerIdentity")
        is not _EXPECTED_TOKENIZER_IDENTITY_CLASS
    ):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime module drift: TokenizerIdentity"
        )
    if module_state.get("hashlib") is not _EXPECTED_BYTE_HASHLIB_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime module drift: hashlib"
        )
    if module_state.get("json") is not _EXPECTED_BYTE_JSON_MODULE:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime module drift: json"
        )

    for field, expected in _EXPECTED_BYTE_MODULE_CONSTANTS.items():
        observed = module_state.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime module drift: {field}"
            )

    observed_config = module_state.get("_CONFIG")
    if type(observed_config) is not dict or set(observed_config) != set(
        _EXPECTED_BYTE_MODULE_CONFIG
    ):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime module drift: _CONFIG"
        )
    for field, expected in _EXPECTED_BYTE_MODULE_CONFIG.items():
        observed = observed_config[field]
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerDecisionError(
                "canonical byte tokenizer runtime module drift: _CONFIG"
            )

    expected_helpers = _verified_canonical_byte_tokenizer_helper_codes()
    for name, expected_code in expected_helpers.items():
        function = module_state.get(name)
        if (
            type(function) is not FunctionType
            or function.__code__ != expected_code
            or function.__globals__ is not module_state
            or function.__defaults__ is not None
            or function.__kwdefaults__ is not None
        ):
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime module drift: {name}"
            )
        if function.__builtins__ is not vars(builtins):
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime module drift: {name} builtins"
            )


def _runtime_byte_tokenizer_method(
    class_state: Mapping[str, Any],
    name: str,
) -> FunctionType:
    """Read one live class method without invoking descriptor binding."""

    raw = class_state.get(name)
    if name == "identity":
        if type(raw) is not property or raw.fget is None:
            raise TokenizerDecisionError(
                "canonical byte tokenizer runtime implementation drift: identity"
            )
        function = raw.fget
    elif name == "oov_count":
        if type(raw) is not staticmethod:
            raise TokenizerDecisionError(
                "canonical byte tokenizer runtime implementation drift: oov_count"
            )
        function = raw.__func__
    else:
        if type(raw) is not FunctionType:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime implementation drift: {name}"
            )
        function = raw
    if type(function) is not FunctionType:
        raise TokenizerDecisionError(
            f"canonical byte tokenizer runtime implementation drift: {name}"
        )
    return function


def _verify_runtime_byte_tokenizer_method_defaults(
    name: str,
    function: FunctionType,
) -> None:
    """Bind behavior-bearing callable defaults omitted from code-object identity."""

    if function.__defaults__ is not None:
        raise TokenizerDecisionError(
            f"canonical byte tokenizer runtime implementation drift: {name} defaults"
        )
    expected_kwdefaults = _EXPECTED_TOKENIZER_METHOD_KWDEFAULTS[name]
    observed_kwdefaults = function.__kwdefaults__
    if expected_kwdefaults is None:
        valid = observed_kwdefaults is None
    else:
        valid = (
            type(observed_kwdefaults) is dict
            and set(observed_kwdefaults) == set(expected_kwdefaults)
            and all(
                type(observed_kwdefaults[key]) is type(expected)
                and observed_kwdefaults[key] == expected
                for key, expected in expected_kwdefaults.items()
            )
        )
    if not valid:
        raise TokenizerDecisionError(
            f"canonical byte tokenizer runtime implementation drift: {name} defaults"
        )


def _verify_runtime_byte_tokenizer_class() -> None:
    """Bind the live ByteTokenizer class to the source-pinned runtime contract."""

    # TokenizerIdentity intentionally omits several class-level protocol fields.
    # Constructor dispatch is part of the tokenizer runtime contract too. Both
    # module aliases are mutable, so comparing them to each other is insufficient:
    # a replacement class could preserve every checked method object while a
    # custom metaclass, local __new__, or inherited constructor returns a forged
    # instance. Pin the canonical construction path before any instantiation.
    if type(ByteTokenizer) is not type:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime implementation drift: metaclass"
        )
    if (
        type.__getattribute__(ByteTokenizer, "__bases__") != (object,)
        or type.__getattribute__(ByteTokenizer, "__mro__") != (ByteTokenizer, object)
    ):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime implementation drift: bases"
        )
    class_state = type.__getattribute__(ByteTokenizer, "__dict__")
    for hook in (
        "__new__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__del__",
    ):
        if hook in class_state:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime implementation drift: {hook}"
            )
    sentinel = object()
    for field, expected in _EXPECTED_TOKENIZER_CLASS_STATE.items():
        observed = class_state.get(field, sentinel)
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime identity drift: {field}"
            )
    class_special_tokens = class_state.get("special_tokens", sentinel)
    if (
        type(class_special_tokens) is not MappingProxyType
        or dict(class_special_tokens)
    ):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime identity drift: special_tokens"
        )

    expected_method_codes = _verified_canonical_byte_tokenizer_method_codes()
    for name, expected_code in expected_method_codes.items():
        runtime_method = _runtime_byte_tokenizer_method(class_state, name)
        if runtime_method.__code__ != expected_code:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime implementation drift: {name}"
            )
        if runtime_method.__globals__ is not vars(byte_module):
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime implementation drift: {name} globals"
            )
        if runtime_method.__builtins__ is not vars(builtins):
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime implementation drift: {name} builtins"
            )
        _verify_runtime_byte_tokenizer_method_defaults(name, runtime_method)


def _snapshot_verified_tokenizer_identity(tokenizer: Any) -> dict[str, object]:
    """Copy canonical identity values so no live object is trusted after reseal."""

    snapshot: dict[str, object] = {}
    for field in (
        "version",
        "config_sha256",
        "vocab_sha256",
        "vocab_size",
        "normalization",
        "encoding",
    ):
        expected = _EXPECTED_TOKENIZER_RUNTIME_IDENTITY[field]
        observed = getattr(tokenizer, field)
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerDecisionError(
                f"canonical byte tokenizer runtime identity drift: {field}"
            )
        snapshot[field] = observed

    special_tokens = getattr(tokenizer, "special_tokens")
    if type(special_tokens) is not MappingProxyType or dict(special_tokens):
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime identity drift: special_tokens"
        )
    snapshot["special_tokens"] = {}
    return snapshot


def _verified_canonical_byte_tokenizer_identity() -> tuple[str, dict[str, object]]:
    """Bind the loaded runtime identity to the source-pinned byte baseline."""

    _verify_byte_tokenizer_runtime_dependencies()
    implementation = _verify_canonical_byte_tokenizer_implementation()
    _verify_runtime_byte_tokenizer_module_state()
    _verify_runtime_byte_tokenizer_class()

    tokenizer = ByteTokenizer().identity
    first_snapshot = _snapshot_verified_tokenizer_identity(tokenizer)

    # Identity construction/observation is effectful Python code. Re-seal after
    # the first observation, then observe once more and perform a final re-seal.
    # Callers consume only the detached snapshot, never the live identity object.
    _verify_byte_tokenizer_runtime_dependencies()
    _verify_runtime_byte_tokenizer_module_state()
    _verify_runtime_byte_tokenizer_class()

    final_snapshot = _snapshot_verified_tokenizer_identity(tokenizer)
    if final_snapshot != first_snapshot:
        raise TokenizerDecisionError(
            "canonical byte tokenizer runtime identity observation drift"
        )

    _verify_byte_tokenizer_runtime_dependencies()
    _verify_runtime_byte_tokenizer_module_state()
    _verify_runtime_byte_tokenizer_class()
    return implementation, final_snapshot


def _verify_selection(
    selection: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> tuple[str, dict[str, Any]]:
    if not isinstance(selection, Mapping):
        raise TokenizerDecisionError("unsupported balanced-selection authority")
    selection = dict(selection)
    if selection.get("schema") != SELECTION_SCHEMA:
        raise TokenizerDecisionError("unsupported balanced-selection authority")
    try:
        _, totals = verify_balanced_selection(
            selection,
            expected_selection_identity_sha256=expected_selection_identity_sha256,
            expected_retained_inventory_identity_sha256=(
                expected_retained_inventory_identity_sha256
            ),
            expected_decontamination_authority_sha256=(
                expected_decontamination_authority_sha256
            ),
            expected_dedup_authority_sha256=expected_dedup_authority_sha256,
            expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
            expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
        )
    except BalancedSplitApplicationError as exc:
        raise TokenizerDecisionError(str(exc)) from exc
    identity = _require_sha256(
        expected_selection_identity_sha256,
        field="expected_selection_identity_sha256",
    )
    return identity, totals


def _verify_split_application(
    application: Mapping[str, Any],
    selection: Mapping[str, Any],
    totals: Mapping[str, Any],
    *,
    expected_application_identity_sha256: str,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> str:
    if not isinstance(application, Mapping):
        raise TokenizerDecisionError("split application fields are not closed-world")
    application = dict(application)
    if set(application) != _APPLICATION_KEYS:
        raise TokenizerDecisionError("split application fields are not closed-world")
    if application.get("schema") != APPLICATION_SCHEMA:
        raise TokenizerDecisionError("unsupported split-application authority")
    if application.get("status") != "PASS_ZERO_CREDIT":
        raise TokenizerDecisionError("split application is not canonical PASS_ZERO_CREDIT")

    expected_application = _require_sha256(
        expected_application_identity_sha256,
        field="expected_application_identity_sha256",
    )
    claimed_application = _require_sha256(
        application.get("application_identity_sha256"),
        field="application_identity_sha256",
    )
    if _self_hash(application, "application_identity_sha256") != claimed_application:
        raise TokenizerDecisionError("split application self-hash mismatch")
    if claimed_application != expected_application:
        raise TokenizerDecisionError("split application identity mismatch")

    expected_selection = _require_sha256(
        expected_selection_identity_sha256,
        field="expected_selection_identity_sha256",
    )
    if application.get("balanced_selection_identity_sha256") != expected_selection:
        raise TokenizerDecisionError("split application is from a different balanced selection")
    if selection.get("balanced_selection_identity_sha256") != expected_selection:
        raise TokenizerDecisionError("selection identity drift")

    expected_upstreams = {
        "retained_inventory_identity_sha256": expected_retained_inventory_identity_sha256,
        "decontamination_authority_sha256": expected_decontamination_authority_sha256,
        "dedup_authority_sha256": expected_dedup_authority_sha256,
        "balance_policy_identity_sha256": expected_balance_policy_identity_sha256,
        "balance_result_identity_sha256": expected_balance_result_identity_sha256,
    }
    for field, expected in expected_upstreams.items():
        expected_sha = _require_sha256(expected, field=f"expected_{field}")
        if selection.get(field) != expected_sha or application.get(field) != expected_sha:
            raise TokenizerDecisionError(f"{field} lineage mismatch")

    if application.get("canonical_split_git_blob_sha1") != CANONICAL_SPLIT_GIT_BLOB_SHA1:
        raise TokenizerDecisionError("split mechanics identity drift")
    split_spec_identity = _require_sha256(
        application.get("split_spec_identity_sha256"),
        field="split_spec_identity_sha256",
    )
    if split_spec_identity != CANONICAL_SPLIT_SPEC_IDENTITY_SHA256:
        raise TokenizerDecisionError(
            "split application does not bind canonical split spec authority"
        )
    boundary = application.get("claim_boundary")
    if (
        type(boundary) is not dict
        or set(boundary) != set(_ZERO_CREDIT_BOUNDARY)
        or any(
            type(boundary[key]) is not type(expected) or boundary[key] != expected
            for key, expected in _ZERO_CREDIT_BOUNDARY.items()
        )
    ):
        raise TokenizerDecisionError("split application truth boundary widened")
    accounting = {
        "selected_record_count": "record_count",
        "selected_source_bytes": "source_bytes",
        "selected_family_source_bytes": "family_source_bytes",
        "selected_stratum_source_bytes": "stratum_source_bytes",
    }
    for application_field, totals_field in accounting.items():
        observed = application.get(application_field)
        expected = totals.get(totals_field)
        if type(expected) is int:
            valid = type(observed) is int and observed == expected
        elif isinstance(expected, Mapping):
            valid = (
                type(observed) is dict
                and set(observed) == set(expected)
                and all(
                    type(observed[key]) is int and observed[key] == expected[key]
                    for key in expected
                )
            )
        else:
            valid = False
        if not valid:
            raise TokenizerDecisionError(f"split application {application_field} drift")
    if _self_hash(application, "application_identity_sha256") != claimed_application:
        raise TokenizerDecisionError("split application changed during verification")
    return expected_application


def _bind_upstreams(
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> tuple[str, str]:
    if not isinstance(selection, Mapping):
        raise TokenizerDecisionError("unsupported balanced-selection authority")
    if not isinstance(application, Mapping):
        raise TokenizerDecisionError("split application fields are not closed-world")
    selection_snapshot = dict(selection)
    application_snapshot = dict(application)

    selection_identity, totals = _verify_selection(
        selection_snapshot,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    application_identity = _verify_split_application(
        application_snapshot,
        selection_snapshot,
        totals,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    return selection_identity, application_identity


def bind_byte_baseline_decision(
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> dict[str, Any]:
    """Bind canonical balanced-selection/split lineage to the frozen byte tokenizer."""

    _verify_byte_tokenizer_runtime_dependencies()
    selection_identity, application_identity = _bind_upstreams(
        selection,
        application,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    _verify_byte_tokenizer_runtime_dependencies()
    validated_upstreams = {
        "retained_inventory_identity_sha256": _require_sha256(
            expected_retained_inventory_identity_sha256,
            field="expected_retained_inventory_identity_sha256",
        ),
        "decontamination_authority_sha256": _require_sha256(
            expected_decontamination_authority_sha256,
            field="expected_decontamination_authority_sha256",
        ),
        "dedup_authority_sha256": _require_sha256(
            expected_dedup_authority_sha256,
            field="expected_dedup_authority_sha256",
        ),
        "balance_policy_identity_sha256": _require_sha256(
            expected_balance_policy_identity_sha256,
            field="expected_balance_policy_identity_sha256",
        ),
        "balance_result_identity_sha256": _require_sha256(
            expected_balance_result_identity_sha256,
            field="expected_balance_result_identity_sha256",
        ),
    }

    (
        tokenizer_implementation_git_blob_sha1,
        tokenizer_identity,
    ) = _verified_canonical_byte_tokenizer_identity()

    core: dict[str, Any] = {
        "schema": SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "balanced_selection_identity_sha256": selection_identity,
        "split_application_identity_sha256": application_identity,
        **validated_upstreams,
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "split_spec_identity_sha256": CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
        "canonical_byte_tokenizer_git_blob_sha1": tokenizer_implementation_git_blob_sha1,
        "tokenizer_version": tokenizer_identity["version"],
        "tokenizer_config_sha256": tokenizer_identity["config_sha256"],
        "tokenizer_vocab_sha256": tokenizer_identity["vocab_sha256"],
        "vocab_size": tokenizer_identity["vocab_size"],
        "normalization": tokenizer_identity["normalization"],
        "encoding": tokenizer_identity["encoding"],
        "tokenizer_fit_executed": False,
        "training_authorized_by_this_report": False,
        "compute_authorized_by_this_report": False,
        "authorized_optimized_target_exposure": 0,
    }
    decision_identity = authority_sha256(core)
    _verify_byte_tokenizer_runtime_dependencies()
    return {**core, "decision_identity_sha256": decision_identity}


def verify_byte_baseline_decision(
    report: Mapping[str, Any],
    selection: Mapping[str, Any],
    application: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_application_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> None:
    """Verify decision identity and rebind the canonical upstream lineage."""

    _verify_byte_tokenizer_runtime_dependencies()
    if not isinstance(report, Mapping):
        raise TokenizerDecisionError("report fields are not closed-world")
    report = dict(report)
    if set(report) != _REPORT_KEYS:
        raise TokenizerDecisionError("report fields are not closed-world")
    if report.get("schema") != SCHEMA or report.get("status") != STATUS:
        raise TokenizerDecisionError("report schema/status mismatch")
    if report.get("decision") != DECISION:
        raise TokenizerDecisionError("unexpected tokenizer decision")

    selection_identity, application_identity = _bind_upstreams(
        selection,
        application,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_application_identity_sha256=expected_application_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    _verify_byte_tokenizer_runtime_dependencies()
    validated_upstreams = {
        "retained_inventory_identity_sha256": _require_sha256(
            expected_retained_inventory_identity_sha256,
            field="expected_retained_inventory_identity_sha256",
        ),
        "decontamination_authority_sha256": _require_sha256(
            expected_decontamination_authority_sha256,
            field="expected_decontamination_authority_sha256",
        ),
        "dedup_authority_sha256": _require_sha256(
            expected_dedup_authority_sha256,
            field="expected_dedup_authority_sha256",
        ),
        "balance_policy_identity_sha256": _require_sha256(
            expected_balance_policy_identity_sha256,
            field="expected_balance_policy_identity_sha256",
        ),
        "balance_result_identity_sha256": _require_sha256(
            expected_balance_result_identity_sha256,
            field="expected_balance_result_identity_sha256",
        ),
    }
    if report.get("balanced_selection_identity_sha256") != selection_identity:
        raise TokenizerDecisionError("report balanced-selection identity mismatch")
    if report.get("split_application_identity_sha256") != application_identity:
        raise TokenizerDecisionError("report split-application identity mismatch")
    report_split_spec_identity = _require_sha256(
        report.get("split_spec_identity_sha256"),
        field="split_spec_identity_sha256",
    )
    if report_split_spec_identity != CANONICAL_SPLIT_SPEC_IDENTITY_SHA256:
        raise TokenizerDecisionError("report split-spec authority identity drift")
    for field, expected in validated_upstreams.items():
        if report.get(field) != expected:
            raise TokenizerDecisionError(f"report {field} drift")
    if report.get("canonical_split_git_blob_sha1") != CANONICAL_SPLIT_GIT_BLOB_SHA1:
        raise TokenizerDecisionError("report split mechanics identity drift")

    (
        tokenizer_implementation_git_blob_sha1,
        tokenizer_identity,
    ) = _verified_canonical_byte_tokenizer_identity()
    expected_tokenizer = {
        "canonical_byte_tokenizer_git_blob_sha1": tokenizer_implementation_git_blob_sha1,
        "tokenizer_version": tokenizer_identity["version"],
        "tokenizer_config_sha256": tokenizer_identity["config_sha256"],
        "tokenizer_vocab_sha256": tokenizer_identity["vocab_sha256"],
        "vocab_size": tokenizer_identity["vocab_size"],
        "normalization": tokenizer_identity["normalization"],
        "encoding": tokenizer_identity["encoding"],
    }
    for key, expected in expected_tokenizer.items():
        observed = report.get(key)
        if type(observed) is not type(expected) or observed != expected:
            raise TokenizerDecisionError(f"report {key} drift")
    if report.get("tokenizer_fit_executed") is not False:
        raise TokenizerDecisionError("byte-baseline decision cannot claim tokenizer fitting")
    if report.get("training_authorized_by_this_report") is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize training")
    if report.get("compute_authorized_by_this_report") is not False:
        raise TokenizerDecisionError("tokenizer decision cannot authorize compute")
    exposure = report.get("authorized_optimized_target_exposure")
    if type(exposure) is not int or exposure != 0:
        raise TokenizerDecisionError("tokenizer decision cannot authorize exposure")

    supplied_identity = _require_sha256(
        report.get("decision_identity_sha256"), field="decision_identity_sha256"
    )
    if _self_hash(report, "decision_identity_sha256") != supplied_identity:
        raise TokenizerDecisionError("decision report identity mismatch")
    _verify_byte_tokenizer_runtime_dependencies()
