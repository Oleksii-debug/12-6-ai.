from __future__ import annotations

import hashlib

import pytest

from twelve_six.data.rights_firewall import (
    ALLOW,
    DENY,
    PURPOSE_ANALYSIS,
    PURPOSE_FINAL_TEST,
    PURPOSE_REDISTRIBUTION,
    PURPOSE_SELECTION_VALIDATION,
    PURPOSE_TOKENIZER_FITTING,
    PURPOSE_TRAINING,
    EvaluationReservation,
    PurposeDecisions,
    PurposeRightsAuthority,
    PurposeRightsError,
    PurposeRightsFirewall,
    SourceObjectIdentity,
    purpose_decisions_from_mapping,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _object(name: str) -> SourceObjectIdentity:
    return SourceObjectIdentity(
        source_id=name,
        source_family=f"fixture:{name}",
        upstream_revision=_sha(f"revision:{name}")[:40],
        object_path=f"objects/{name}.txt",
        content_sha256=_sha(f"content:{name}"),
        git_blob_sha1=_sha(f"blob:{name}")[:40],
    )


def _decisions(
    *,
    training: str = DENY,
    tokenizer: str = DENY,
    selection: str = DENY,
    final: str = DENY,
    redistribution: str = DENY,
    analysis: str = ALLOW,
) -> PurposeDecisions:
    return PurposeDecisions(
        training=training,
        tokenizer_fitting=tokenizer,
        selection_validation=selection,
        final_test=final,
        redistribution=redistribution,
        analysis=analysis,
    )


def _authority(
    authority_id: str,
    source_object: SourceObjectIdentity,
    decisions: PurposeDecisions,
    *,
    issued_at: str = "2026-08-26T12:00:00Z",
    commit: str | None = None,
    reservation: EvaluationReservation | None = None,
    successor_of: str | None = None,
    license_id: str = "BROAD-UPSTREAM-LICENSE",
    evidence: str | None = None,
) -> PurposeRightsAuthority:
    return PurposeRightsAuthority(
        authority_id=authority_id,
        authority_commit_sha=commit or _sha(f"commit:{authority_id}")[:40],
        issued_at_utc=issued_at,
        source_object=source_object,
        upstream_license_id=license_id,
        rights_evidence_sha256=evidence or _sha(f"rights:{authority_id}"),
        project_decision_ref=f"authority://{authority_id}",
        decisions=decisions,
        reservation=reservation,
        successor_of_authority_id=successor_of,
    )


def _reservation(purpose: str, *, at: str = "2026-08-26T11:00:00Z") -> EvaluationReservation:
    return EvaluationReservation(
        reserved_purposes=(purpose,),
        reserved_at_utc=at,
        reservation_commit_sha=_sha(f"reservation:{purpose}:{at}")[:40],
    )


def test_training_only_object_cannot_enter_evaluation() -> None:
    obj = _object("training-only")
    firewall = PurposeRightsFirewall()
    firewall.register(
        _authority(
            "TRAIN-AUTH",
            obj,
            _decisions(
                training=ALLOW,
                tokenizer=ALLOW,
                redistribution=ALLOW,
                analysis=ALLOW,
            ),
        )
    )

    firewall.require_use(obj, PURPOSE_TRAINING)
    firewall.require_use(obj, PURPOSE_TOKENIZER_FITTING)
    with pytest.raises(PurposeRightsError, match="selection-validation denied"):
        firewall.require_use(obj, PURPOSE_SELECTION_VALIDATION)
    with pytest.raises(PurposeRightsError, match="final-test denied"):
        firewall.require_use(obj, PURPOSE_FINAL_TEST)


def test_selection_reserved_object_cannot_later_enter_training() -> None:
    obj = _object("selection")
    firewall = PurposeRightsFirewall()
    firewall.register(
        _authority(
            "SELECT-V1",
            obj,
            _decisions(selection=ALLOW),
            reservation=_reservation(PURPOSE_SELECTION_VALIDATION),
        )
    )

    successor = _authority(
        "SELECT-V2",
        obj,
        _decisions(training=ALLOW, tokenizer=ALLOW),
        issued_at="2026-08-26T13:00:00Z",
        commit=_sha("commit:SELECT-V2")[:40],
        successor_of="SELECT-V1",
    )
    with pytest.raises(PurposeRightsError, match="selection-reserved object can never"):
        firewall.register(successor)


def test_final_test_object_cannot_influence_selection() -> None:
    obj = _object("final-test")
    firewall = PurposeRightsFirewall()
    firewall.register(
        _authority(
            "FINAL-AUTH",
            obj,
            _decisions(final=ALLOW),
            reservation=_reservation(PURPOSE_FINAL_TEST),
        )
    )

    firewall.require_use(obj, PURPOSE_FINAL_TEST)
    with pytest.raises(PurposeRightsError, match="may not influence"):
        firewall.assert_observation_may_influence(
            obj,
            observed_under=PURPOSE_FINAL_TEST,
            target_purpose=PURPOSE_SELECTION_VALIDATION,
        )


def test_rights_changes_require_successor_authority() -> None:
    obj = _object("mutable-rights")
    firewall = PurposeRightsFirewall()
    firewall.register(
        _authority(
            "RIGHTS-V1",
            obj,
            _decisions(training=ALLOW, tokenizer=ALLOW),
        )
    )

    changed_without_successor = _authority(
        "RIGHTS-V2-BAD",
        obj,
        _decisions(training=ALLOW, tokenizer=ALLOW, redistribution=ALLOW),
        issued_at="2026-08-26T13:00:00Z",
        commit=_sha("commit:RIGHTS-V2-BAD")[:40],
    )
    with pytest.raises(PurposeRightsError, match="rights changes require a successor"):
        firewall.register(changed_without_successor)

    changed_with_successor = _authority(
        "RIGHTS-V2",
        obj,
        _decisions(training=ALLOW, tokenizer=ALLOW, redistribution=ALLOW),
        issued_at="2026-08-26T13:00:00Z",
        commit=_sha("commit:RIGHTS-V2")[:40],
        successor_of="RIGHTS-V1",
    )
    firewall.register(changed_with_successor)
    assert firewall.current(obj).authority_id == "RIGHTS-V2"


def test_broad_license_never_implies_omitted_project_purposes() -> None:
    broad = {
        PURPOSE_TRAINING: ALLOW,
        PURPOSE_REDISTRIBUTION: ALLOW,
        PURPOSE_ANALYSIS: ALLOW,
    }
    with pytest.raises(PurposeRightsError, match="exactly all six"):
        purpose_decisions_from_mapping(broad)

    obj = _object("broad-license")
    explicit = _authority(
        "BROAD-BUT-EXPLICIT",
        obj,
        _decisions(training=ALLOW, redistribution=ALLOW),
        license_id="CC0-1.0",
    )
    assert explicit.decisions.selection_validation == DENY
    assert explicit.decisions.final_test == DENY


def test_evaluation_permission_requires_exact_reservation_timestamp_and_commit() -> None:
    obj = _object("unreserved-selection")
    with pytest.raises(PurposeRightsError, match="requires an exact pre-use reservation"):
        _authority("NO-RESERVATION", obj, _decisions(selection=ALLOW))

    with pytest.raises(PurposeRightsError, match="ending in Z"):
        EvaluationReservation(
            reserved_purposes=(PURPOSE_SELECTION_VALIDATION,),
            reserved_at_utc="2026-08-26T14:00:00+03:00",
            reservation_commit_sha="a" * 40,
        )

    with pytest.raises(PurposeRightsError, match="lowercase 40-hex"):
        EvaluationReservation(
            reserved_purposes=(PURPOSE_SELECTION_VALIDATION,),
            reserved_at_utc="2026-08-26T11:00:00Z",
            reservation_commit_sha="not-a-commit",
        )


def test_final_test_authority_structurally_denies_adaptive_purposes() -> None:
    with pytest.raises(
        PurposeRightsError, match="selection-validation and final-test|final-test object must deny"
    ):
        _decisions(final=ALLOW, selection=ALLOW)

    final = _decisions(final=ALLOW, analysis=ALLOW)
    assert final.training == DENY
    assert final.tokenizer_fitting == DENY
    assert final.selection_validation == DENY


def test_purpose_dimensions_are_independent_including_analysis_and_redistribution() -> None:
    obj = _object("independent")
    firewall = PurposeRightsFirewall()
    firewall.register(
        _authority(
            "INDEPENDENT",
            obj,
            _decisions(analysis=ALLOW, redistribution=DENY),
        )
    )
    firewall.require_use(obj, PURPOSE_ANALYSIS)
    with pytest.raises(PurposeRightsError, match="redistribution denied"):
        firewall.require_use(obj, PURPOSE_REDISTRIBUTION)
