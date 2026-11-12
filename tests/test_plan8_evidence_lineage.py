import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict, replace
from hashlib import sha256

import pytest
from test_ai_qa_control_section4 import _candidate, _failure, _pass_receipt, _policy
from test_plan8_evidence_policy import _attestation, _authority, _registry_policy

from twelve_six.ai_qa_control import GateKind, GateVerdict, _canonical_json_bytes
from twelve_six.evidence_lineage import EvidenceArtifact, EvidenceLineage, defect_identity
from twelve_six.evidence_policy import EvidenceTarget, admit_signed_evidence


def _sha(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _other_failure(failure):
    summary = "AssertionError: a distinct contract failed"
    return replace(
        failure, failure_summary=summary, failure_summary_sha256=_sha(summary.encode()),
        source_evidence_identity_sha256="6" * 64,
    )


def test_retry_workers_deduplicate_exact_observations_and_group_recurrences(tmp_path):
    root = tmp_path / "lineage"
    failure = _failure()
    initial = EvidenceLineage(root).record_failure(failure)
    with ThreadPoolExecutor(max_workers=4) as pool:
        retried = tuple(pool.map(
            lambda index: EvidenceLineage(root).record_failure(
                replace(failure, defect_id=f"worker-{index}")
            ),
            range(8),
        ))
    assert set(retried) == {initial}
    other_observation = replace(failure, source_evidence_identity_sha256="5" * 64)
    different = _other_failure(failure)
    journal = EvidenceLineage(root)
    second = journal.record_failure(other_observation)
    separate = journal.record_failure(different)
    assert defect_identity(failure) == defect_identity(other_observation)
    assert defect_identity(failure) != defect_identity(different)
    snapshot = journal.readback(defect_identity(failure))
    assert set(snapshot["records"]) == {initial, second}
    assert set(journal.readback(defect_identity(different))["records"]) == {separate}
    assert snapshot == EvidenceLineage(root).readback(defect_identity(failure))
    assert snapshot == journal.readback(
        defect_identity(failure), expected_snapshot_sha256=snapshot["snapshot_sha256"],
    )
    with pytest.raises(ValueError, match="trusted snapshot"):
        journal.readback(defect_identity(failure), expected_snapshot_sha256="0" * 64)


def test_failure_repair_retest_and_promotion_read_back_one_exact_chain(tmp_path):
    failure = _failure()
    candidate = _candidate(failure)
    package = "b" * 64
    journal = EvidenceLineage(tmp_path / "lineage")
    observed = journal.record_failure(failure)
    frozen = journal.freeze_candidate(observed, candidate, package_sha256=package)
    receipts = (
        _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation"),
        _pass_receipt(GateKind.ADVERSARIAL, actor_id="regression-automation"),
        _pass_receipt(GateKind.SIL, actor_id="independent-sil"),
        _pass_receipt(
            GateKind.PHYSICAL, actor_id="scope-auditor",
            verdict=GateVerdict.NOT_APPLICABLE, reason="software-only defect",
        ),
    )
    retests = tuple(journal.record_retest(frozen, item) for item in receipts)
    registry, policy = _registry_policy()
    subject = "windows-nvda-final-product"
    target = EvidenceTarget(candidate.candidate_git_sha, package)
    key, verify = _authority("independent")
    signed = admit_signed_evidence(
        _attestation(key, target, "Q3", subject, registry_sha256=registry.identity_sha256()),
        target=target, signature_verifier=verify,
        trusted_keys={("Q3", subject): frozenset({"independent"})},
    )
    args = {
        "failure": failure, "candidate": candidate, "repair_policy": _policy(),
        "evidence_policy": policy, "capability_id": subject,
        "certifier_actor_id": "independent-certifier",
    }
    blocked = journal.record_promotion_decision(frozen, retests, evidence=(), **args)
    ready = journal.record_promotion_decision(frozen, retests, evidence=(signed,), **args)
    assert ready == journal.record_promotion_decision(
        frozen, tuple(reversed(retests)), evidence=(signed,), **args,
    )
    records = EvidenceLineage(journal.root).readback(defect_identity(failure))["records"]
    assert records[frozen]["parents"] == [observed]
    assert records[blocked]["payload"]["decision"]["decision"] == "BLOCK"
    assert records[ready]["parents"] == [frozen, *sorted(retests)]
    assert records[ready]["payload"]["decision"]["decision"] == "READY_FOR_INDEPENDENT_PROMOTION"
    assert records[ready]["payload"]["evidence_policy_sha256"] == policy.policy_sha256
    assert records[ready]["payload"]["qualification_evidence"][0]["artifact_sha256"] == signed.artifact_sha256


def test_repair_retries_keep_parents_and_refuse_conflicting_freezes_or_retests(tmp_path):
    journal = EvidenceLineage(tmp_path / "lineage")
    failure = _failure()
    different = _other_failure(failure)
    observed = journal.record_failure(failure)
    second_observed = journal.record_failure(different)
    candidate = _candidate(failure)
    frozen = journal.freeze_candidate(observed, candidate, package_sha256="b" * 64)
    second = journal.freeze_candidate(second_observed, _candidate(different), package_sha256="c" * 64)
    retry = replace(candidate, candidate_git_sha="d" * 40)
    retried = journal.freeze_candidate(
        observed, retry, package_sha256="d" * 64, repair_parent=frozen,
    )
    assert journal.readback(defect_identity(failure))["records"][retried]["parents"] == [observed, frozen]
    with pytest.raises(ValueError, match="different defect"):
        journal.freeze_candidate(observed, retry, package_sha256="d" * 64, repair_parent=second)
    with pytest.raises(ValueError, match="does not bind"):
        journal.freeze_candidate(second_observed, candidate, package_sha256="b" * 64)
    with pytest.raises(ValueError, match="conflicting"):
        journal.freeze_candidate(observed, replace(candidate, proposer_actor_id="another-worker"),
                                 package_sha256="f" * 64)
    receipt = _pass_receipt(GateKind.COMPONENT, actor_id="regression-automation")
    journal.record_retest(frozen, receipt)
    with pytest.raises(ValueError, match="different candidate"):
        journal.record_retest(retried, receipt)
    with pytest.raises(ValueError, match="conflicting"):
        journal.record_retest(frozen, replace(receipt, verdict=GateVerdict.FAIL))


def test_artifact_retention_preserves_provenance_and_detects_corruption(tmp_path):
    root = tmp_path / "lineage"
    journal = EvidenceLineage(root)
    failure = _failure()
    actual = b"local evidence bytes"
    pinned = EvidenceArtifact(
        failure.source_evidence_identity_sha256, _sha(actual), len(actual), "local-fixture", actual,
    )
    remote = EvidenceArtifact("d" * 64, "e" * 64, 42, "source-retained-log")
    observed = journal.record_failure(failure, artifacts=(pinned, remote))
    assert observed == journal.record_failure(failure, artifacts=(remote, pinned))
    with pytest.raises(ValueError, match="primary evidence"):
        journal.record_failure(failure, artifacts=(remote,))
    with pytest.raises(ValueError, match="conflicting"):
        journal.record_failure(failure, artifacts=(replace(pinned, provenance="other"), remote))
    reopened = EvidenceLineage(root)
    assert reopened.retained_artifact(observed, pinned.content_sha256) == actual
    assert reopened.retained_artifact(observed, remote.content_sha256) is None
    record = reopened.readback(defect_identity(failure))["records"][observed]
    assert {item["retention"] for item in record["artifacts"]} == {"LOCAL_PINNED", "SOURCE_REFERENCE"}
    with closing(sqlite3.connect(root / "evidence-lineage.sqlite3")) as db, db:
        db.execute("UPDATE retained SET content = ?", (b"tampered evidence",))
    with pytest.raises(ValueError, match="content does not match"):
        reopened.readback(defect_identity(failure))


def test_transaction_rolls_back_failure_and_retained_bytes_together(tmp_path, monkeypatch):
    journal = EvidenceLineage(tmp_path / "lineage")
    failure = _failure()
    actual = b"rollback evidence"
    artifact = EvidenceArtifact(
        failure.source_evidence_identity_sha256, _sha(actual), len(actual), "local-fixture", actual,
    )
    append = EvidenceLineage._append

    def interrupted(*args, **kwargs):
        append(*args, **kwargs)
        raise RuntimeError("interrupted before commit")

    with monkeypatch.context() as patch:
        patch.setattr(EvidenceLineage, "_append", staticmethod(interrupted))
        with pytest.raises(RuntimeError, match="interrupted"):
            journal.record_failure(failure, artifacts=(artifact,))
    assert journal.readback(defect_identity(failure))["records"] == {}
    with closing(sqlite3.connect(journal.root / "evidence-lineage.sqlite3")) as db:
        assert db.execute("SELECT count(*) FROM retained").fetchone() == (0,)
    assert journal.record_failure(failure, artifacts=(artifact,))


def test_missing_causal_parent_and_unsafe_sqlite_sidecars_fail_closed(tmp_path):
    root = tmp_path / "lineage"
    journal = EvidenceLineage(root)
    failure = _failure()
    observed = journal.record_failure(failure)
    journal.freeze_candidate(observed, _candidate(failure), package_sha256="b" * 64)
    sidecar = root / "evidence-lineage.sqlite3-wal"
    sidecar.mkdir()
    with pytest.raises(ValueError, match="unsafe evidence journal file"):
        journal.readback(defect_identity(failure))
    sidecar.rmdir()
    with closing(sqlite3.connect(root / "evidence-lineage.sqlite3")) as db, db:
        db.execute("DELETE FROM records WHERE identity = ?", (observed,))
    with pytest.raises(ValueError, match="missing or out-of-order parent"):
        journal.readback(defect_identity(failure))


def test_self_consistent_record_still_needs_the_correct_causal_parent(tmp_path):
    journal = EvidenceLineage(tmp_path / "lineage")
    failure = _failure()
    observed = journal.record_failure(failure)
    defect = defect_identity(failure)
    forged = _canonical_json_bytes({
        "schema_version": "12-6.plan8-evidence-lineage-record.v1",
        "defect_sha256": defect,
        "kind": "RETEST",
        "parents": [observed],
        "payload": {"receipt": {"forged": True}},
        "artifacts": [],
    })
    with closing(sqlite3.connect(journal.root / "evidence-lineage.sqlite3")) as db, db:
        db.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (_sha(forged), defect, None, forged))
    with pytest.raises(ValueError, match="retest does not reference a candidate freeze"):
        journal.readback(defect)


def test_self_consistent_promotion_cannot_invent_retests_or_qualification(tmp_path):
    journal = EvidenceLineage(tmp_path / "lineage")
    failure = _failure()
    candidate = _candidate(failure)
    observed = journal.record_failure(failure)
    frozen = journal.freeze_candidate(observed, candidate, package_sha256="b" * 64)
    defect = defect_identity(failure)
    forged = _canonical_json_bytes({
        "schema_version": "12-6.plan8-evidence-lineage-record.v1",
        "defect_sha256": defect,
        "kind": "PROMOTION_DECISION",
        "parents": [frozen],
        "payload": {
            "decision": {
                "decision": "READY_FOR_INDEPENDENT_PROMOTION",
                "repair_gate": {
                    "decision": "READY_FOR_INDEPENDENT_PROMOTION",
                    "candidate_identity_sha256": candidate.identity_sha256(),
                    "certifier_actor_id": "independent-certifier",
                    "reasons": [],
                },
                "capability_evidence": {
                    "capability_id": "package-runtime", "required_level": "Q2",
                    "admitted_levels": ["Q2"], "package_bound_levels": ["Q2"], "evidence_complete": True,
                },
            },
            "evidence_policy_sha256": "1" * 64,
            "registry_sha256": "2" * 64,
            "repair_policy": asdict(_policy()),
            "qualification_evidence": [],
        },
        "artifacts": [],
    })
    with closing(sqlite3.connect(journal.root / "evidence-lineage.sqlite3")) as db, db:
        db.execute("INSERT INTO records VALUES (?, ?, ?, ?)", (_sha(forged), defect, None, forged))
    with pytest.raises(ValueError, match="does not match the recorded candidate or gates"):
        journal.readback(defect)
