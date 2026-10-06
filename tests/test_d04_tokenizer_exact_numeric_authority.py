"""Exact scalar-type regressions for zero-credit tokenizer decision authorities.

Synthetic lineage hashes in these unit tests do not establish real corpus or
training authorization; the upstream binder is stubbed only for local report tests.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.tokenization import decision_authority as authority


SHA = {f"expected_{key}": str(index) * 64 for index, key in enumerate(
    (
        "selection_identity_sha256",
        "application_identity_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "dedup_authority_sha256",
        "balance_policy_identity_sha256",
        "balance_result_identity_sha256",
    ),
    start=1,
)}


def _selection() -> dict:
    return {
        "balanced_selection_identity_sha256": SHA[
            "expected_selection_identity_sha256"
        ],
        **{
            key: SHA[f"expected_{key}"]
            for key in authority._UPSTREAM_IDENTITY_FIELDS
        },
    }


def _application(selection: dict) -> tuple[dict, dict]:
    totals = {
        "record_count": 2,
        "source_bytes": 100,
        "family_source_bytes": {"example": 100},
        "stratum_source_bytes": {"ua": 100},
    }
    application = {
        "schema": authority.APPLICATION_SCHEMA,
        "status": "PASS_ZERO_CREDIT",
        "balanced_selection_identity_sha256": selection[
            "balanced_selection_identity_sha256"
        ],
        **{
            key: selection[key]
            for key in authority._UPSTREAM_IDENTITY_FIELDS
        },
        "canonical_split_git_blob_sha1": authority.CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256,
        "selected_record_count": totals["record_count"],
        "selected_source_bytes": totals["source_bytes"],
        "selected_family_source_bytes": totals["family_source_bytes"],
        "selected_stratum_source_bytes": totals["stratum_source_bytes"],
        "split_family": {},
        "claim_boundary": dict(authority._ZERO_CREDIT_BOUNDARY),
    }
    application["application_identity_sha256"] = authority.authority_sha256(
        application
    )
    return application, totals


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("authorized_optimized_target_exposure", 0.0),
        ("tokenizer_fit_authorized", 0),
        ("training_eligible", 0),
        ("model_training_authorized", 0),
        ("paid_compute_authorized", 0),
        ("final_test_outcomes_read", 0),
    ],
)
def test_split_application_rejects_resealed_numeric_type_aliases(
    field: str, replacement: object,
) -> None:
    selection = _selection()
    application, totals = _application(selection)
    kwargs = {
        **SHA,
        "expected_application_identity_sha256": application[
            "application_identity_sha256"
        ],
    }
    assert authority._verify_split_application(
        application, selection, totals, **kwargs,
    ) == application["application_identity_sha256"]

    forged = deepcopy(application)
    forged["claim_boundary"][field] = replacement
    forged.pop("application_identity_sha256")
    forged["application_identity_sha256"] = authority.authority_sha256(forged)
    kwargs["expected_application_identity_sha256"] = forged[
        "application_identity_sha256"
    ]
    with pytest.raises(authority.TokenizerDecisionError, match="truth boundary"):
        authority._verify_split_application(
            forged, selection, totals, **kwargs,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("authorized_optimized_target_exposure", 0.0),
        ("authorized_optimized_target_exposure", False),
        ("vocab_size", 256.0),
    ],
)
def test_decision_report_rejects_resealed_numeric_type_aliases(
    monkeypatch: pytest.MonkeyPatch, field: str, replacement: object,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority, "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    report = authority.bind_byte_baseline_decision(
        selection, application, **SHA,
    )
    assert report["authorized_optimized_target_exposure"] == 0
    assert type(report["authorized_optimized_target_exposure"]) is int
    authority.verify_byte_baseline_decision(
        report, selection, application, **SHA,
    )

    forged = deepcopy(report)
    forged[field] = replacement
    forged.pop("decision_identity_sha256")
    forged["decision_identity_sha256"] = authority.authority_sha256(forged)
    with pytest.raises(authority.TokenizerDecisionError):
        authority.verify_byte_baseline_decision(
            forged, selection, application, **SHA,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("version", "tampered-byte-v1"),
        ("vocab_size", 257),
        ("normalization", "NFC"),
        ("encoding", "latin-1"),
        ("special_tokens", {"unexpected": 1}),
    ],
)
def test_bind_rejects_process_local_tokenizer_runtime_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    monkeypatch.setattr(authority.ByteTokenizer, field, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime identity drift: {field}",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("version", "tampered-byte-v1"),
        ("vocab_size", 257),
        ("normalization", "NFC"),
        ("encoding", "latin-1"),
        ("special_tokens", {"unexpected": 1}),
    ],
)
def test_verify_rejects_process_local_tokenizer_runtime_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)
    monkeypatch.setattr(authority.ByteTokenizer, field, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime identity drift: {field}",
    ):
        authority.verify_byte_baseline_decision(
            report,
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    ("field", "nested_key", "replacement"),
    [
        ("selected_record_count", None, 2.0),
        ("selected_source_bytes", None, 100.0),
        ("selected_family_source_bytes", "example", 100.0),
        ("selected_stratum_source_bytes", "ua", 100.0),
    ],
)
def test_split_application_rejects_resealed_accounting_type_aliases(
    field: str, nested_key: str | None, replacement: float,
) -> None:
    selection = _selection()
    application, totals = _application(selection)
    if nested_key is None:
        application[field] = replacement
    else:
        application[field][nested_key] = replacement
    application.pop("application_identity_sha256")
    application["application_identity_sha256"] = authority.authority_sha256(
        application
    )
    kwargs = {
        **SHA,
        "expected_application_identity_sha256": application[
            "application_identity_sha256"
        ],
    }
    with pytest.raises(authority.TokenizerDecisionError, match=f"{field} drift"):
        authority._verify_split_application(
            application, selection, totals, **kwargs,
        )


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), chr(0xD800)],
)
def test_canonical_authority_hash_refuses_nonfinite_or_invalid_unicode(
    invalid: object,
) -> None:
    with pytest.raises(authority.TokenizerDecisionError, match="strict UTF-8 JSON"):
        authority.authority_sha256({"untrusted": invalid})


def test_canonical_authority_hash_refuses_recursive_input() -> None:
    recursive: dict[str, object] = {}
    recursive["self"] = recursive
    with pytest.raises(authority.TokenizerDecisionError, match="strict UTF-8 JSON"):
        authority.authority_sha256(recursive)


def test_canonical_authority_hash_preserves_finite_identity() -> None:
    import hashlib

    value = {"b": 2, "a": 1, "fraction": 0.25}
    expected = hashlib.sha256(b'{"a":1,"b":2,"fraction":0.25}').hexdigest()
    assert authority.authority_sha256(value) == expected


@pytest.mark.parametrize(
    "surface",
    ["version", "config_sha256", "vocab_sha256"],
)
def test_reloaded_authority_does_not_trust_mutated_byte_module_baseline(
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
) -> None:
    import importlib

    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    tampered_hash = "a" * 64

    try:
        with monkeypatch.context() as patch:
            if surface == "version":
                patch.setattr(byte_module, "BYTE_TOKENIZER_VERSION", "tampered-byte-v1")
                patch.setattr(byte_module.ByteTokenizer, "version", "tampered-byte-v1")
            elif surface == "config_sha256":
                patch.setattr(byte_module, "BYTE_TOKENIZER_HASH", tampered_hash)
                patch.setattr(byte_module, "tokenizer_config_hash", lambda: tampered_hash)
            else:
                patch.setattr(byte_module, "BYTE_VOCAB_HASH", tampered_hash)
                patch.setattr(byte_module, "vocab_hash", lambda: tampered_hash)

            reloaded = importlib.reload(authority)
            patch.setattr(
                reloaded,
                "_bind_upstreams",
                lambda *_args, **_kwargs: (
                    SHA["expected_selection_identity_sha256"],
                    SHA["expected_application_identity_sha256"],
                ),
            )
            with pytest.raises(
                reloaded.TokenizerDecisionError,
                match=f"runtime identity drift: {surface}",
            ):
                reloaded.bind_byte_baseline_decision(
                    selection,
                    application,
                    **SHA,
                )
    finally:
        importlib.reload(authority)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("pad_id", 0),
        ("bos_id", 1),
        ("eos_id", 2),
        ("byte_offset", 1),
        ("special_tokens", {}),
    ],
)
def test_bind_rejects_byte_class_semantics_missing_from_tokenizer_identity(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    monkeypatch.setattr(authority.ByteTokenizer, field, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime identity drift: {field}",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    "method_name",
    ["__init__", "identity", "encode", "decode", "oov_count", "fertility"],
)
def test_bind_rejects_semantically_equivalent_runtime_method_replacement(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    raw = vars(authority.ByteTokenizer)[method_name]
    if method_name == "identity":
        original = raw.fget
        assert original is not None

        def wrapped_identity(self):
            return original(self)

        replacement = property(wrapped_identity)
    elif method_name == "oov_count":
        original = raw.__func__

        def wrapped_oov_count(text):
            return original(text)

        replacement = staticmethod(wrapped_oov_count)
    else:
        original = raw

        def wrapped_method(*args, **kwargs):
            return original(*args, **kwargs)

        replacement = wrapped_method

    monkeypatch.setattr(authority.ByteTokenizer, method_name, replacement)
    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime implementation drift: {method_name}",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    ("method_name", "attribute", "replacement"),
    [
        ("encode", "__kwdefaults__", {"add_bos": True, "add_eos": False}),
        (
            "decode",
            "__kwdefaults__",
            {"skip_special_tokens": True, "errors": "ignore"},
        ),
        (
            "decode",
            "__kwdefaults__",
            {"skip_special_tokens": 1, "errors": "strict"},
        ),
        ("fertility", "__defaults__", (0,)),
    ],
)
def test_bind_rejects_runtime_method_default_drift(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    attribute: str,
    replacement: object,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    method = vars(authority.ByteTokenizer)[method_name]
    assert type(method) is type(lambda: None)
    monkeypatch.setattr(method, attribute, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime implementation drift: {method_name} defaults",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    "helper_name",
    [
        "canonical_config_json",
        "tokenizer_config_hash",
        "canonical_vocab_json",
        "vocab_hash",
    ],
)
def test_bind_rejects_runtime_byte_helper_replacement(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original = getattr(byte_module, helper_name)

    def wrapped(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(byte_module, helper_name, wrapped)
    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime module drift: {helper_name}",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def test_bind_rejects_runtime_config_drift_even_when_hash_check_is_bypassed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    monkeypatch.setitem(byte_module._CONFIG, "encoding", "latin-1")
    monkeypatch.setattr(
        byte_module,
        "tokenizer_config_hash",
        lambda: authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY["config_sha256"],
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: _CONFIG",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def test_bind_rejects_runtime_tokenizer_identity_factory_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original = byte_module.TokenizerIdentity
    monkeypatch.setattr(
        byte_module,
        "TokenizerIdentity",
        lambda **kwargs: original(**kwargs),
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: TokenizerIdentity",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def test_bind_rejects_runtime_module_tokenizer_export_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    class ReplacementTokenizer(authority.ByteTokenizer):
        pass

    monkeypatch.setattr(byte_module, "ByteTokenizer", ReplacementTokenizer)
    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: ByteTokenizer",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def _clone_function_with_detached_globals(function):
    from types import FunctionType

    cloned = FunctionType(
        function.__code__,
        dict(function.__globals__),
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    cloned.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    return cloned


def test_bind_rejects_tokenizer_method_with_detached_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original = vars(authority.ByteTokenizer)["encode"]
    replacement = _clone_function_with_detached_globals(original)
    monkeypatch.setattr(authority.ByteTokenizer, "encode", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: encode globals",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def test_bind_rejects_tokenizer_helper_with_detached_globals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original = byte_module.tokenizer_config_hash
    replacement = _clone_function_with_detached_globals(original)
    monkeypatch.setattr(byte_module, "tokenizer_config_hash", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: tokenizer_config_hash",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def test_bind_rejects_added_instance_lookup_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    def passthrough_getattribute(self, name):
        return object.__getattribute__(self, name)

    monkeypatch.setattr(
        authority.ByteTokenizer,
        "__getattribute__",
        passthrough_getattribute,
        raising=False,
    )
    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: __getattribute__",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize("module_name", ["json", "hashlib"])
def test_bind_rejects_semantically_equivalent_module_dependency_proxy(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original = getattr(byte_module, module_name)

    class Proxy:
        def __getattr__(self, name):
            return getattr(original, name)

    monkeypatch.setattr(byte_module, module_name, Proxy())
    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime module drift: {module_name}",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("schema_version", True),
        ("byte_offset", False),
        ("byte_values", 256.0),
        ("vocab_size", 256.0),
    ],
)
def test_bind_rejects_runtime_config_numeric_type_alias_with_hash_bypass(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    monkeypatch.setitem(byte_module._CONFIG, field, replacement)
    monkeypatch.setattr(
        byte_module,
        "tokenizer_config_hash",
        lambda: authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY["config_sha256"],
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: _CONFIG",
    ):
        authority.bind_byte_baseline_decision(
            selection,
            application,
            **SHA,
        )


def _replacement_tokenizer_namespace() -> dict[str, object]:
    """Clone the checked class surface without generated storage descriptors."""

    return {
        name: value
        for name, value in vars(authority.ByteTokenizer).items()
        if name not in {"__dict__", "__weakref__"}
    }


def test_bind_rejects_dual_alias_custom_tokenizer_metaclass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    canonical_identity = authority.ByteTokenizer().identity

    class DivergentTokenizer:
        identity = canonical_identity

        def encode(self, _text: str) -> list[int]:
            return [255]

    class ConstructorMeta(type):
        def __call__(cls, *_args, **_kwargs):
            return DivergentTokenizer()

    replacement = ConstructorMeta(
        "ByteTokenizer",
        (object,),
        _replacement_tokenizer_namespace(),
    )
    monkeypatch.setattr(authority, "ByteTokenizer", replacement)
    monkeypatch.setattr(byte_module, "ByteTokenizer", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: metaclass",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_dual_alias_local_new_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    canonical_identity = authority.ByteTokenizer().identity

    class DivergentTokenizer:
        identity = canonical_identity

        def encode(self, _text: str) -> list[int]:
            return [254]

    def divergent_new(_cls):
        return DivergentTokenizer()

    namespace = _replacement_tokenizer_namespace()
    namespace["__new__"] = staticmethod(divergent_new)
    replacement = type("ByteTokenizer", (object,), namespace)
    monkeypatch.setattr(authority, "ByteTokenizer", replacement)
    monkeypatch.setattr(byte_module, "ByteTokenizer", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: __new__",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_dual_alias_inherited_constructor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    canonical_identity = authority.ByteTokenizer().identity

    class DivergentTokenizer:
        identity = canonical_identity

        def encode(self, _text: str) -> list[int]:
            return [253]

    class ConstructorBase:
        def __new__(cls):
            return DivergentTokenizer()

    replacement = type(
        "ByteTokenizer",
        (ConstructorBase,),
        _replacement_tokenizer_namespace(),
    )
    monkeypatch.setattr(authority, "ByteTokenizer", replacement)
    monkeypatch.setattr(byte_module, "ByteTokenizer", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: bases",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_dual_alias_destructor_side_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    destructor_called = False

    def effectful_del(_self):
        nonlocal destructor_called
        destructor_called = True

    namespace = _replacement_tokenizer_namespace()
    namespace["__del__"] = effectful_del
    replacement = type("ByteTokenizer", (object,), namespace)
    monkeypatch.setattr(authority, "ByteTokenizer", replacement)
    monkeypatch.setattr(byte_module, "ByteTokenizer", replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: __del__",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)
    assert destructor_called is False


def _clone_function_with_captured_builtins(function, captured_builtins):
    from types import FunctionType

    globals_state = function.__globals__
    missing = object()
    previous = globals_state.get("__builtins__", missing)
    globals_state["__builtins__"] = captured_builtins
    try:
        cloned = FunctionType(
            function.__code__,
            globals_state,
            function.__name__,
            function.__defaults__,
            function.__closure__,
        )
    finally:
        if previous is missing:
            del globals_state["__builtins__"]
        else:
            globals_state["__builtins__"] = previous
    cloned.__kwdefaults__ = (
        None if function.__kwdefaults__ is None else dict(function.__kwdefaults__)
    )
    return cloned


def test_bind_rejects_dual_alias_encode_with_detached_captured_builtins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    original = vars(authority.ByteTokenizer)["encode"]
    assert type(original) is type(lambda: None)
    canonical_builtins = original.__builtins__
    assert isinstance(canonical_builtins, dict)
    divergent_builtins = dict(canonical_builtins)
    divergent_builtins["list"] = lambda _value: [999]
    replacement_encode = _clone_function_with_captured_builtins(
        original,
        divergent_builtins,
    )
    assert replacement_encode.__code__ is original.__code__
    assert replacement_encode.__globals__ is original.__globals__
    assert replacement_encode.__builtins__ is divergent_builtins

    namespace = _replacement_tokenizer_namespace()
    namespace["encode"] = replacement_encode
    replacement = type("ByteTokenizer", (object,), namespace)
    monkeypatch.setattr(authority, "ByteTokenizer", replacement)
    monkeypatch.setattr(byte_module, "ByteTokenizer", replacement)

    assert replacement().identity == authority._CanonicalTokenizerIdentity(
        version="s0-byte-v1",
        config_sha256=(
            "b04055c1061dd641dcab7cb9d62a931f09b8d1a070140a926ceb4e91d73ca8e1"
        ),
        vocab_sha256=(
            "905ed40bb42cc4d550e228ff5f24158d504b38e8ed5974dfa3077bd5867ad571"
        ),
        vocab_size=256,
        normalization="none",
        encoding="utf-8",
        special_tokens={},
    )
    assert replacement().encode("A") == [999]

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: encode builtins",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


@pytest.mark.parametrize(
    "helper_name",
    [
        "canonical_config_json",
        "tokenizer_config_hash",
        "canonical_vocab_json",
        "vocab_hash",
    ],
)
def test_bind_rejects_byte_helper_with_detached_captured_builtins(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    original = getattr(byte_module, helper_name)
    detached_builtins = dict(original.__builtins__)
    replacement = _clone_function_with_captured_builtins(
        original,
        detached_builtins,
    )
    assert replacement.__code__ is original.__code__
    assert replacement.__globals__ is original.__globals__
    assert replacement.__builtins__ is detached_builtins
    monkeypatch.setattr(byte_module, helper_name, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"runtime module drift: {helper_name} builtins",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)

def test_bind_rejects_shared_builtin_member_drift_with_same_captured_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    encode = vars(authority.ByteTokenizer)["encode"]
    original_list = builtins.list
    observed_error = None
    divergent = None
    same_mapping = None
    builtins.list = lambda _value: [999]
    try:
        divergent = byte_module.ByteTokenizer().encode("A")
        same_mapping = encode.__builtins__ is vars(builtins)
        try:
            authority.bind_byte_baseline_decision(selection, application, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.list = original_list

    assert divergent == [999]
    assert same_mapping is True
    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical byte tokenizer runtime dependency drift: builtins.list"
    )


@pytest.mark.parametrize(
    ("dependency_name", "replacement"),
    [
        ("json.dumps", lambda *_args, **_kwargs: "{}"),
        ("json.loads", lambda *_args, **_kwargs: {}),
        ("hashlib.sha1", lambda *_args, **_kwargs: object()),
        ("hashlib.sha256", lambda *_args, **_kwargs: object()),
    ],
)
def test_bind_rejects_mutated_shared_stdlib_dependency_member(
    monkeypatch: pytest.MonkeyPatch,
    dependency_name: str,
    replacement,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    module_name, member = dependency_name.split(".", 1)
    module = getattr(authority, module_name)
    monkeypatch.setattr(module, member, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=(
            "canonical byte tokenizer runtime dependency drift: "
            + dependency_name.replace(".", r"\.")
        ),
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


@pytest.mark.parametrize(
    ("module_name", "expected_module"),
    [
        ("json", authority._EXPECTED_BYTE_JSON_MODULE),
        ("hashlib", authority._EXPECTED_BYTE_HASHLIB_MODULE),
    ],
)
def test_bind_rejects_dual_alias_stdlib_dependency_proxy(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    expected_module: object,
) -> None:
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    class Proxy:
        def __getattr__(self, name: str):
            return getattr(expected_module, name)

    proxy = Proxy()
    monkeypatch.setattr(authority, module_name, proxy)
    monkeypatch.setattr(byte_module, module_name, proxy)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=(
            "canonical byte tokenizer runtime dependency drift: "
            f"{module_name} module"
        ),
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)

def test_bind_rejects_identity_constructor_postflight_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original_init = authority._CanonicalTokenizerIdentity.__init__

    def effectful_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)

        def divergent_encode(
            _self,
            _text: str,
            *,
            add_bos: bool = False,
            add_eos: bool = False,
        ) -> list[int]:
            del add_bos, add_eos
            return [999]

        monkeypatch.setattr(authority.ByteTokenizer, "encode", divergent_encode)

    monkeypatch.setattr(
        authority._CanonicalTokenizerIdentity,
        "__init__",
        effectful_init,
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: encode",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_identity_observer_postflight_runtime_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original_getattribute = authority._CanonicalTokenizerIdentity.__getattribute__
    armed = True

    def effectful_getattribute(self, name: str):
        nonlocal armed
        value = original_getattribute(self, name)
        if armed and name == "version":
            armed = False

            def divergent_fertility(_self, _text: str) -> float:
                return 999.0

            monkeypatch.setattr(
                authority.ByteTokenizer,
                "fertility",
                divergent_fertility,
            )
        return value

    monkeypatch.setattr(
        authority._CanonicalTokenizerIdentity,
        "__getattribute__",
        effectful_getattribute,
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: fertility",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_dual_alias_tokenizer_identity_class_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import base as base_module
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    class ReplacementIdentity:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    canonical_identity = base_module.TokenizerIdentity
    monkeypatch.setattr(
        authority,
        "_CanonicalTokenizerIdentity",
        ReplacementIdentity,
    )
    monkeypatch.setattr(byte_module, "TokenizerIdentity", ReplacementIdentity)

    forged = byte_module.ByteTokenizer().identity
    assert type(forged) is ReplacementIdentity
    assert forged.version == "s0-byte-v1"
    assert authority._CanonicalTokenizerIdentity is byte_module.TokenizerIdentity
    assert byte_module.TokenizerIdentity is not canonical_identity

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: TokenizerIdentity",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_second_identity_observation_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original_getattribute = authority._CanonicalTokenizerIdentity.__getattribute__
    version_reads = 0

    def staged_getattribute(self, name: str):
        nonlocal version_reads
        value = original_getattribute(self, name)
        if name == "version":
            version_reads += 1
            if version_reads == 2:
                return "forged-byte-v2"
        return value

    monkeypatch.setattr(
        authority._CanonicalTokenizerIdentity,
        "__getattribute__",
        staged_getattribute,
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime identity drift: version",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)
    assert version_reads == 2


def test_bind_reseals_after_second_identity_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    original_getattribute = authority._CanonicalTokenizerIdentity.__getattribute__
    version_reads = 0

    def staged_getattribute(self, name: str):
        nonlocal version_reads
        value = original_getattribute(self, name)
        if name == "version":
            version_reads += 1
            if version_reads == 2:

                def divergent_fertility(_self, _text: str) -> float:
                    return 999.0

                monkeypatch.setattr(
                    authority.ByteTokenizer,
                    "fertility",
                    divergent_fertility,
                )
        return value

    monkeypatch.setattr(
        authority._CanonicalTokenizerIdentity,
        "__getattribute__",
        staged_getattribute,
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime implementation drift: fertility",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)
    assert version_reads == 2


def test_bind_rejects_triple_alias_tokenizer_identity_class_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from twelve_six.tokenization import base as base_module
    from twelve_six.tokenization import byte as byte_module

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    class ReplacementIdentity:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    canonical_identity = authority._EXPECTED_TOKENIZER_IDENTITY_CLASS
    monkeypatch.setattr(
        authority,
        "_CanonicalTokenizerIdentity",
        ReplacementIdentity,
    )
    monkeypatch.setattr(byte_module, "TokenizerIdentity", ReplacementIdentity)
    monkeypatch.setattr(base_module, "TokenizerIdentity", ReplacementIdentity)

    assert authority._CanonicalTokenizerIdentity is byte_module.TokenizerIdentity
    assert byte_module.TokenizerIdentity is base_module.TokenizerIdentity
    assert base_module.TokenizerIdentity is not canonical_identity

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime module drift: TokenizerIdentity",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


def test_bind_rejects_tokenizer_base_module_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    class BaseProxy:
        TokenizerIdentity = authority._EXPECTED_TOKENIZER_IDENTITY_CLASS

    monkeypatch.setattr(authority, "base_module", BaseProxy())

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime dependency drift: base module",
    ):
        authority.bind_byte_baseline_decision(selection, application, **SHA)


@pytest.mark.parametrize(
    "builtin_name",
    ["all", "any", "compile", "dict", "getattr", "object", "open", "set", "type", "vars"],
)
def test_bind_rejects_mutated_verifier_builtin_dependency(builtin_name: str) -> None:
    import builtins

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    original = builtins.__dict__[builtin_name]

    def replacement(*_args, **_kwargs):
        return None

    observed_error = None
    builtins.__dict__[builtin_name] = replacement
    try:
        try:
            authority.bind_byte_baseline_decision(selection, application, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.__dict__[builtin_name] = original

    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical byte tokenizer runtime dependency drift: "
        f"builtins.{builtin_name}"
    )


def test_bind_rejects_builtin_module_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "builtins", object())

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime dependency drift: builtins module",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


def test_bind_rejects_source_path_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authority,
        "_BYTE_TOKENIZER_SOURCE_PATH",
        authority._BYTE_TOKENIZER_SOURCE_PATH.with_name("byte-shadow.py"),
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime dependency drift: source path",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


def test_bind_rejects_source_reader_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_class = authority._EXPECTED_BYTE_SOURCE_PATH_CLASS
    monkeypatch.setattr(source_class, "read_bytes", lambda _self: b"")

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime dependency drift: source reader",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


def test_split_application_snapshots_top_level_mapping_before_verification() -> None:
    selection = _selection()
    application, totals = _application(selection)
    application["selected_source_bytes"] = totals["source_bytes"] + 1
    application.pop("application_identity_sha256")
    application["application_identity_sha256"] = authority.authority_sha256(application)
    expected_application = application["application_identity_sha256"]

    class MutatingApplication(dict):
        def get(self, key, default=None):
            if key == "balanced_selection_identity_sha256":
                self["selected_source_bytes"] = totals["source_bytes"]
            return super().get(key, default)

    staged = MutatingApplication(application)
    kwargs = {
        **SHA,
        "expected_application_identity_sha256": expected_application,
    }
    with pytest.raises(
        authority.TokenizerDecisionError,
        match="selected_source_bytes drift",
    ):
        authority._verify_split_application(staged, selection, totals, **kwargs)
    assert staged["selected_source_bytes"] == totals["source_bytes"] + 1


def test_bind_does_not_reread_upstreams_after_successful_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }

    def bind_then_mutate(*_args, **_kwargs):
        for field in authority._UPSTREAM_IDENTITY_FIELDS:
            selection[field] = "f" * 64
        application["split_spec_identity_sha256"] = "e" * 64
        return (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        )

    monkeypatch.setattr(authority, "_bind_upstreams", bind_then_mutate)
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)

    for field in authority._UPSTREAM_IDENTITY_FIELDS:
        assert report[field] == SHA[f"expected_{field}"]
    assert (
        report["split_spec_identity_sha256"]
        == authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    )


def test_verify_does_not_reread_upstreams_after_successful_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    canonical_result = (
        SHA["expected_selection_identity_sha256"],
        SHA["expected_application_identity_sha256"],
    )
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: canonical_result,
    )
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)

    def bind_then_mutate(*_args, **_kwargs):
        for field in authority._UPSTREAM_IDENTITY_FIELDS:
            selection[field] = "f" * 64
        application["split_spec_identity_sha256"] = "e" * 64
        return canonical_result

    monkeypatch.setattr(authority, "_bind_upstreams", bind_then_mutate)
    authority.verify_byte_baseline_decision(
        report,
        selection,
        application,
        **SHA,
    )


def test_bind_upstreams_reuses_one_detached_selection_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {"sentinel": "application"}
    observed = {}

    def verify_selection(snapshot, **_kwargs):
        observed["selection"] = snapshot
        for field in authority._UPSTREAM_IDENTITY_FIELDS:
            selection[field] = "f" * 64
        return SHA["expected_selection_identity_sha256"], {}

    def verify_application(snapshot, selection_snapshot, _totals, **_kwargs):
        assert snapshot == application
        assert selection_snapshot is observed["selection"]
        for field in authority._UPSTREAM_IDENTITY_FIELDS:
            assert selection_snapshot[field] == SHA[f"expected_{field}"]
        return SHA["expected_application_identity_sha256"]

    monkeypatch.setattr(authority, "_verify_selection", verify_selection)
    monkeypatch.setattr(
        authority,
        "_verify_split_application",
        verify_application,
    )
    result = authority._bind_upstreams(selection, application, **SHA)
    assert result == (
        SHA["expected_selection_identity_sha256"],
        SHA["expected_application_identity_sha256"],
    )


def test_verify_snapshots_stateful_report_before_semantic_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    canonical_result = (
        SHA["expected_selection_identity_sha256"],
        SHA["expected_application_identity_sha256"],
    )
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: canonical_result,
    )
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)
    report["authorized_optimized_target_exposure"] = 1
    report.pop("decision_identity_sha256")
    report["decision_identity_sha256"] = authority.authority_sha256(report)

    class StatefulReport(dict):
        def get(self, key, default=None):
            if key == "decision":
                self["authorized_optimized_target_exposure"] = 0
            elif key == "decision_identity_sha256":
                result = super().get(key, default)
                self["authorized_optimized_target_exposure"] = 1
                return result
            return super().get(key, default)

    staged = StatefulReport(report)
    with pytest.raises(
        authority.TokenizerDecisionError,
        match="cannot authorize exposure",
    ):
        authority.verify_byte_baseline_decision(
            staged,
            selection,
            application,
            **SHA,
        )
    assert staged["authorized_optimized_target_exposure"] == 1


def test_bind_source_read_bypasses_path_instance_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    source_class = authority._EXPECTED_BYTE_SOURCE_PATH_CLASS
    original_getattribute = source_class.__getattribute__

    def guarded_getattribute(self, name: str):
        if self is authority._EXPECTED_BYTE_SOURCE_PATH and name in {
            "open",
            "read_bytes",
        }:
            raise AssertionError("canonical source read used mutable Path dispatch")
        return original_getattribute(self, name)

    monkeypatch.setattr(source_class, "__getattribute__", guarded_getattribute)
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)
    assert (
        report["canonical_byte_tokenizer_git_blob_sha1"]
        == authority.CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1
    )


def test_bind_upstreams_deep_detaches_nested_selection_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = {
        "nested": {
            "family_source_bytes": {"ua": 100},
        },
    }
    application = {"sentinel": "application"}

    def verify_selection(snapshot, **_kwargs):
        selection["nested"]["family_source_bytes"]["ua"] = 999
        assert snapshot["nested"]["family_source_bytes"]["ua"] == 100
        return SHA["expected_selection_identity_sha256"], {}

    def verify_application(_snapshot, selection_snapshot, _totals, **_kwargs):
        assert selection_snapshot["nested"]["family_source_bytes"]["ua"] == 100
        return SHA["expected_application_identity_sha256"]

    monkeypatch.setattr(authority, "_verify_selection", verify_selection)
    monkeypatch.setattr(
        authority,
        "_verify_split_application",
        verify_application,
    )
    assert authority._bind_upstreams(selection, application, **SHA) == (
        SHA["expected_selection_identity_sha256"],
        SHA["expected_application_identity_sha256"],
    )
    assert selection["nested"]["family_source_bytes"]["ua"] == 999


@pytest.mark.parametrize(
    ("mapping_name", "key"),
    [
        ("_ZERO_CREDIT_BOUNDARY", "training_eligible"),
        ("_EXPECTED_TOKENIZER_RUNTIME_IDENTITY", "version"),
        ("_EXPECTED_TOKENIZER_CLASS_STATE", "version"),
        ("_EXPECTED_BYTE_MODULE_CONSTANTS", "BYTE_TOKENIZER_VERSION"),
        ("_EXPECTED_BYTE_RUNTIME_BUILTINS", "list"),
    ],
)
def test_verifier_expected_mapping_is_immutable(
    mapping_name: str,
    key: str,
) -> None:
    expected = getattr(authority, mapping_name)
    original = expected[key]
    with pytest.raises(TypeError):
        expected[key] = object()
    assert expected[key] is original or expected[key] == original


@pytest.mark.parametrize(
    ("container_name", "field", "nested_key"),
    [
        ("_EXPECTED_TOKENIZER_METHOD_KWDEFAULTS", "encode", "add_bos"),
        ("_EXPECTED_TOKENIZER_METHOD_KWDEFAULTS", "decode", "errors"),
        ("_EXPECTED_BYTE_MODULE_CONFIG", "special_tokens", "forged"),
        ("_EXPECTED_TOKENIZER_RUNTIME_IDENTITY", "special_tokens", "forged"),
    ],
)
def test_verifier_nested_expected_mapping_is_immutable(
    container_name: str,
    field: str,
    nested_key: str,
) -> None:
    nested = getattr(authority, container_name)[field]
    with pytest.raises(TypeError):
        nested[nested_key] = object()


def test_builtin_expectation_table_cannot_be_retargeted_with_builtin() -> None:
    import builtins

    replacement = lambda _value: [999]
    with pytest.raises(TypeError):
        authority._EXPECTED_BYTE_RUNTIME_BUILTINS["list"] = replacement

    original = builtins.list
    observed_error = None
    builtins.list = replacement
    try:
        try:
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.list = original

    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical byte tokenizer runtime dependency drift: builtins.list"
    )

def test_expected_builtin_root_cannot_be_retargeted_with_live_builtin() -> None:
    import builtins
    from types import MappingProxyType

    original_root = authority._EXPECTED_BYTE_RUNTIME_BUILTINS
    original_list = builtins.list
    replacement = lambda _value: [999]
    replacement_root = MappingProxyType({
        **original_root,
        "list": replacement,
    })
    observed_error = None
    authority._EXPECTED_BYTE_RUNTIME_BUILTINS = replacement_root
    builtins.list = replacement
    try:
        try:
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.list = original_list
        authority._EXPECTED_BYTE_RUNTIME_BUILTINS = original_root

    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical tokenizer decision verifier root drift: "
        "_EXPECTED_BYTE_RUNTIME_BUILTINS"
    )


def test_identity_observation_cannot_retarget_expected_root_transiently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import MappingProxyType

    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )

    original_root = authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY
    forged_values = {
        "config_sha256": "f" * 64,
        "vocab_sha256": "e" * 64,
        "vocab_size": 999,
        "normalization": "forged",
        "encoding": "latin-1",
    }
    forged_root = MappingProxyType({
        **original_root,
        **forged_values,
    })
    original_getattribute = authority._CanonicalTokenizerIdentity.__getattribute__

    def staged_getattribute(self, name: str):
        value = original_getattribute(self, name)
        if name == "version":
            authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY = forged_root
        if name in forged_values:
            value = forged_values[name]
            if name == "encoding":
                authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY = original_root
        return value

    monkeypatch.setattr(
        authority._CanonicalTokenizerIdentity,
        "__getattribute__",
        staged_getattribute,
    )

    observed_error = None
    try:
        try:
            authority.bind_byte_baseline_decision(selection, application, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        authority._EXPECTED_TOKENIZER_RUNTIME_IDENTITY = original_root

    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical byte tokenizer runtime identity drift: config_sha256"
    )


def test_bind_rejects_byte_module_alias_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(authority, "byte_module", object())

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="runtime dependency drift: byte module",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


def test_bind_source_compile_bypasses_path_string_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = _selection()
    application = {
        "split_spec_identity_sha256": authority.CANONICAL_SPLIT_SPEC_IDENTITY_SHA256
    }
    monkeypatch.setattr(
        authority,
        "_bind_upstreams",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            SHA["expected_application_identity_sha256"],
        ),
    )
    source_class = authority._EXPECTED_BYTE_SOURCE_PATH_CLASS

    def forbidden_str(_self):
        raise AssertionError("source path string dispatch must not run")

    monkeypatch.setattr(source_class, "__str__", forbidden_str)
    report = authority.bind_byte_baseline_decision(selection, application, **SHA)
    assert (
        report["canonical_byte_tokenizer_git_blob_sha1"]
        == authority.CANONICAL_BYTE_TOKENIZER_GIT_BLOB_SHA1
    )

def test_dual_byte_tokenizer_alias_rebind_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForgedTokenizer:
        pass

    monkeypatch.setattr(authority, "ByteTokenizer", ForgedTokenizer)
    monkeypatch.setattr(authority.byte_module, "ByteTokenizer", ForgedTokenizer)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="verifier root drift: ByteTokenizer",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


def test_upstream_selection_verifier_rebind_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authority,
        "verify_balanced_selection",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            {},
        ),
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="verifier root drift: verify_balanced_selection",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


@pytest.mark.parametrize(
    ("root_name", "replacement"),
    [
        ("STATUS", "FORGED"),
        ("CANONICAL_SPLIT_SPEC_IDENTITY_SHA256", "f" * 64),
        ("_EXPECTED_BYTE_SOURCE_PATH_TEXT", "forged-byte.py"),
    ],
)
def test_scalar_authority_root_rebind_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    root_name: str,
    replacement: str,
) -> None:
    monkeypatch.setattr(authority, root_name, replacement)

    with pytest.raises(
        authority.TokenizerDecisionError,
        match=f"verifier root drift: {root_name}",
    ):
        authority.bind_byte_baseline_decision(_selection(), {}, **SHA)


@pytest.mark.parametrize("builtin_name", ["property", "staticmethod"])
def test_descriptor_builtin_rebind_fails_closed(builtin_name: str) -> None:
    import builtins

    original = getattr(builtins, builtin_name)
    observed_error = None
    setattr(builtins, builtin_name, object())
    try:
        try:
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        setattr(builtins, builtin_name, original)

    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical byte tokenizer runtime dependency drift: "
        f"builtins.{builtin_name}"
    )


def test_bind_upstreams_reseals_after_selection_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def verify_selection(*_args, **_kwargs):
        authority.STATUS = "FORGED"
        return SHA["expected_selection_identity_sha256"], {}

    monkeypatch.setattr(authority, "_verify_selection", verify_selection)
    monkeypatch.setattr(
        authority,
        "_verify_split_application",
        lambda *_args, **_kwargs: pytest.fail(
            "application verifier must not run after root drift"
        ),
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="verifier root drift: STATUS",
    ):
        authority._bind_upstreams(
            {"sentinel": "selection"},
            {"sentinel": "application"},
            **SHA,
        )


def test_bind_upstreams_reseals_after_application_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authority,
        "_verify_selection",
        lambda *_args, **_kwargs: (
            SHA["expected_selection_identity_sha256"],
            {},
        ),
    )

    def verify_application(*_args, **_kwargs):
        authority.STATUS = "FORGED"
        return SHA["expected_application_identity_sha256"]

    monkeypatch.setattr(
        authority,
        "_verify_split_application",
        verify_application,
    )

    with pytest.raises(
        authority.TokenizerDecisionError,
        match="verifier root drift: STATUS",
    ):
        authority._bind_upstreams(
            {"sentinel": "selection"},
            {"sentinel": "application"},
            **SHA,
        )

def test_verifier_kwdefault_retarget_cannot_legitimize_builtin_root() -> None:
    import builtins
    from types import MappingProxyType

    verifier = authority._verify_expected_root_integrity
    original_root = authority._EXPECTED_BYTE_RUNTIME_BUILTINS
    original_bytearray = builtins.bytearray
    missing = object()
    original_kwdefaults = getattr(verifier, "__kwdefaults__", missing)

    def hostile_bytearray():
        raise RuntimeError("hostile bytearray")

    replacement_root = MappingProxyType({
        **original_root,
        "bytearray": hostile_bytearray,
    })
    decode_error = None
    observed_error = None
    authority._EXPECTED_BYTE_RUNTIME_BUILTINS = replacement_root
    verifier.__kwdefaults__ = {"_runtime_builtins": replacement_root}
    builtins.bytearray = hostile_bytearray
    try:
        try:
            authority._EXPECTED_BYTE_TOKENIZER_CLASS().decode([65])
        except BaseException as exc:
            decode_error = exc
        try:
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.bytearray = original_bytearray
        authority._EXPECTED_BYTE_RUNTIME_BUILTINS = original_root
        if original_kwdefaults is missing:
            del verifier.__kwdefaults__
        else:
            verifier.__kwdefaults__ = original_kwdefaults

    assert type(decode_error) is RuntimeError
    assert str(decode_error) == "hostile bytearray"
    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical tokenizer decision verifier root drift: "
        "_EXPECTED_BYTE_RUNTIME_BUILTINS"
    )


def test_verifier_root_anchor_has_no_mutable_python_code() -> None:
    verifier = authority._verify_expected_root_integrity
    assert not hasattr(verifier.func, "__code__")
    assert not hasattr(verifier.func, "__defaults__")
    assert not hasattr(verifier.func, "__kwdefaults__")
    original_args = verifier.args
    original_func = verifier.func

    with pytest.raises(AttributeError, match="readonly"):
        verifier.args = original_args
    with pytest.raises(AttributeError, match="readonly"):
        verifier.func = original_func

    assert verifier.args is original_args
    assert verifier.func is original_func


def test_verifier_code_retarget_cannot_legitimize_builtin_root() -> None:
    import builtins
    from types import MappingProxyType

    verifier = authority._verify_expected_root_integrity
    original_root = authority._EXPECTED_BYTE_RUNTIME_BUILTINS
    original_bytearray = builtins.bytearray

    def verifier_noop(_observed, _expected):
        return True

    code_mutation_error = None
    try:
        verifier.func.__code__ = verifier_noop.__code__
    except BaseException as exc:
        code_mutation_error = exc

    def hostile_bytearray():
        raise RuntimeError("hostile bytearray")

    replacement_root = MappingProxyType({
        **original_root,
        "bytearray": hostile_bytearray,
    })
    decode_error = None
    observed_error = None
    authority._EXPECTED_BYTE_RUNTIME_BUILTINS = replacement_root
    builtins.bytearray = hostile_bytearray
    try:
        try:
            authority._EXPECTED_BYTE_TOKENIZER_CLASS().decode([65])
        except BaseException as exc:
            decode_error = exc
        try:
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
        except BaseException as exc:
            observed_error = exc
    finally:
        builtins.bytearray = original_bytearray
        authority._EXPECTED_BYTE_RUNTIME_BUILTINS = original_root

    assert type(code_mutation_error) is AttributeError
    assert type(decode_error) is RuntimeError
    assert str(decode_error) == "hostile bytearray"
    assert type(observed_error) is authority.TokenizerDecisionError
    assert str(observed_error) == (
        "canonical tokenizer decision verifier root drift: "
        "_EXPECTED_BYTE_RUNTIME_BUILTINS"
    )

@pytest.mark.parametrize(
    "target_name",
    [
        "_EXPECTED_BYTE_SOURCE_READ_BYTES",
        "_EXPECTED_BYTE_JSON_DUMPS",
        "_EXPECTED_BYTE_JSON_LOADS",
        "_EXPECTED_VERIFY_BALANCED_SELECTION",
    ],
)
def test_trusted_callable_code_retarget_fails_closed(target_name: str) -> None:
    target = getattr(authority, target_name)
    original_code = target.__code__

    def forged_callable(*_args, **_kwargs):
        return None

    target.__code__ = forged_callable.__code__
    try:
        with pytest.raises(
            authority.TokenizerDecisionError,
            match="canonical tokenizer decision verifier executable drift",
        ):
            authority.bind_byte_baseline_decision(_selection(), {}, **SHA)
    finally:
        target.__code__ = original_code

