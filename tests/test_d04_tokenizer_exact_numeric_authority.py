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
        "claim_boundary": deepcopy(authority._ZERO_CREDIT_BOUNDARY),
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

