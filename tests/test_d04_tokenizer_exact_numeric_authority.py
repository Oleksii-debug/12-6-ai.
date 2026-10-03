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
