import json
from dataclasses import replace

import pytest

from twelve_six_agent_runtime.world import WorldError, WorldFact, WorldModel


def fact(fid, subject="machine", attribute="available", state="observed", **kw):
    return WorldFact(
        fid, subject, kw.pop("subject_kind", "capability"), attribute, state,
        kw.pop("value", "yes"), kw.pop("source_id", "source:live"),
        kw.pop("evidence_id", "receipt:live"), kw.pop("origin_kind", "tool"),
        kw.pop("recorded_at", 10), kw.pop("expires_at", None),
        kw.pop("depends_on", ()),
    )


def test_passport_observation_and_unknown_have_distinct_source_authority():
    model = WorldModel()
    assert model.lookup("unknown", "status", now=20).state == "unknown"
    model.apply(fact("o1"), expected_revision=0)
    view = model.lookup("machine", "available", now=12)
    assert (view.state, view.value, view.source_refs, view.evidence_refs) == (
        "observed", "yes", ("source:live",), ("receipt:live",))
    passport = model.passport("machine", now=12)
    assert passport["subject_kind"] == "capability"
    assert passport["attributes"]["available"]["state"] == "observed"
    assert model.passport("unseen", now=12)["attributes"] == {}


def test_derived_facts_remain_derived_and_stale_on_source_change():
    model = WorldModel()
    model.apply(fact("o1"), expected_revision=0)
    model.apply(fact("d1", attribute="eligible", state="derived", value="probably",
                     origin_kind="model", source_id="model-output",
                     evidence_id="model-trace", depends_on=("o1",)), expected_revision=1)
    model.apply(fact("d2", attribute="actionable", state="derived", value="maybe",
                     origin_kind="model", source_id="model-output",
                     evidence_id="model-trace", depends_on=("d1",)), expected_revision=2)
    assert model.lookup("machine", "eligible", now=12).state == "derived"
    assert model.lookup("machine", "actionable", now=12).state == "derived"
    model.apply(fact("o2", value="no", recorded_at=13), expected_revision=3)
    assert model.lookup("machine", "available", now=15).value == "no"
    for attr in ("eligible", "actionable"):
        view = model.lookup("machine", attr, now=15)
        assert view.state == "stale" and view.value is None


def test_ttl_invalidates_observation_and_dependent_derivation():
    model = WorldModel()
    model.apply(fact("o1", expires_at=15), expected_revision=0)
    model.apply(fact("d1", attribute="derived", state="derived", origin_kind="model",
                     depends_on=("o1",)), expected_revision=1)
    assert model.lookup("machine", "derived", now=14).state == "derived"
    assert model.lookup("machine", "available", now=15).state == "stale"
    assert model.lookup("machine", "derived", now=15).state == "stale"


def test_model_memory_cannot_promote_assumption_to_observed():
    model = WorldModel()
    for origin in ("model", "memory"):
        with pytest.raises(WorldError, match="cannot assert observed"):
            model.apply(fact("o1", origin_kind=origin), expected_revision=0)
    with pytest.raises(WorldError, match="needs dependency"):
        model.apply(fact("d1", state="derived", origin_kind="model"), expected_revision=0)
    with pytest.raises(WorldError, match="unknown or future dependency"):
        model.apply(fact("d1", state="derived", origin_kind="model",
                         depends_on=("missing",)), expected_revision=0)
    assert model.revision == 0


def test_unknown_facts_have_no_evidence_and_no_assumed_value():
    model = WorldModel()
    unknown = fact("u1", attribute="network", state="unknown", value=None,
                   source_id=None, evidence_id=None, origin_kind=None)
    model.apply(unknown, expected_revision=0)
    assert model.lookup("machine", "network", now=20).state == "unknown"
    assert model.lookup("machine", "network", now=20).source_refs == ()
    with pytest.raises(WorldError):
        model.apply(replace(unknown, fact_id="u2", value="yes"), expected_revision=1)


def test_revision_fence_duplicate_id_and_timestamp_regression():
    model = WorldModel()
    model.apply(fact("o1"), expected_revision=0)
    with pytest.raises(WorldError, match="stale"):
        model.apply(fact("o2"), expected_revision=0)
    with pytest.raises(WorldError, match="duplicate"):
        model.apply(fact("o1"), expected_revision=1)
    with pytest.raises(WorldError, match="regression"):
        model.apply(fact("o2", recorded_at=9), expected_revision=1)
    assert model.revision == 1


def test_export_restore_restart_integrity_and_corruption():
    model = WorldModel()
    model.apply(fact("o1"), expected_revision=0)
    model.apply(fact("d1", state="derived", attribute="eligible", origin_kind="model",
                     depends_on=("o1",)), expected_revision=1)
    snapshot = model.export()
    restored = WorldModel.restore(snapshot)
    assert restored.export() == snapshot
    assert restored.passport("machine", now=20) == model.passport("machine", now=20)
    damaged = json.loads(snapshot)
    damaged["events"][0]["value"] = "tampered"
    with pytest.raises(WorldError):
        WorldModel.restore(json.dumps(damaged))
    with pytest.raises(WorldError):
        WorldModel.restore(snapshot.replace('"sha256"', '"sha256"') + " ")


def test_type_boundaries_are_fail_closed():
    model = WorldModel()
    for invalid in (fact("x", recorded_at=True), fact("x", expires_at=10),
                    fact("x", source_id=""), fact("x", evidence_id=""),
                    fact("x", depends_on=("missing",)),
                    fact("x", subject_kind="invalid")):
        with pytest.raises(WorldError):
            model.apply(invalid, expected_revision=0)
    with pytest.raises(WorldError):
        model.lookup("machine", "available", now=True)
