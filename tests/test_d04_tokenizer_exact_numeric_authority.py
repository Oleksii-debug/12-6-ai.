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
