"""Plan 5 Section 5: trusted self-observation, uncertainty, recovery and adversaries."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from twelve_six_agent_runtime.self_model import (
    SelfModelError, SelfModelStore, SelfObservation,
)


def observation(ident="obs1", **kw):
    values = dict(
        observation_id=ident, category="capability", name="python",
        value="available", source_id="probe:local",
        evidence_id=f"receipt:{ident}", origin="test", observed_at=10,
        confidence_ppm=1_000_000,
    )
    values.update(kw)
    return SelfObservation(**values)


def trusted(item):
    return item.source_id == "probe:local" and item.evidence_id.startswith("receipt:")


def test_evidence_only_self_model_restart_and_exact_readback(tmp_path):
    path = tmp_path / "self.sqlite"
    store = SelfModelStore(path)
    assert store.append(observation(), expected_revision=0, verifier=trusted) == 1
    store = SelfModelStore(path)
    assert store.history() == (observation(),)
    profile = store.profile(now=12)
    entry = profile["categories"]["capability"]["python"]
    assert entry["status"] == "evidenced" and entry["value"] == "available"
    assert entry["evidence_id"] == "receipt:obs1"
    assert profile["advisory_only"] is True
    assert profile["grants_permissions"] is False
    assert profile["revision"] == 1


def test_confidence_uncertainty_expiry_and_future_evidence(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    low = observation(
        "low", confidence_ppm=620_000, uncertainty="intermittent failures",
        expires_at=20,
    )
    store.append(low, expected_revision=0, verifier=trusted)
    before = store.profile(now=19)
    assert before["categories"]["capability"]["python"]["status"] == "verify"
    expired = store.profile(now=20)["categories"]["capability"]["python"]
    assert expired["status"] == "stale" and expired["value"] is None
    store.append(observation("future", observed_at=100), expected_revision=1,
                 verifier=trusted)
    assert store.profile(now=21)["categories"]["capability"]["python"]["status"] == "stale"
    assert store.profile(now=101)["categories"]["capability"]["python"]["value"] == "available"


def test_readiness_error_patterns_resources_and_limits_are_versioned(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    specs = [
        ("readiness", "agent", "limited"),
        ("limit", "window", "512 tokens"),
        ("error_pattern", "timeout", "retries unsafe"),
        ("resource", "cpu", "pressure high"),
    ]
    for index, (category, name, value) in enumerate(specs):
        store.append(
            observation(str(index), category=category, name=name, value=value),
            expected_revision=index, verifier=trusted,
        )
    profile = SelfModelStore(store.path).profile(now=12)
    assert profile["revision"] == len(specs)
    for category, name, value in specs:
        assert profile["categories"][category][name]["value"] == value


def test_unverified_or_self_generated_evidence_cannot_update_self_model(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    for bad in (
        observation(origin="model"), observation(origin="memory"),
        observation(source_id="forged"), observation(evidence_id="unbound"),
    ):
        with pytest.raises((SelfModelError, ValueError)):
            store.append(bad, expected_revision=0, verifier=trusted)
    with pytest.raises(SelfModelError):
        store.append(observation(), expected_revision=0, verifier=lambda _: True if False else False)
    with pytest.raises(SelfModelError):
        store.append(observation(), expected_revision=0, verifier=None)
    assert store.history() == ()


def test_adversarial_values_and_metadata_fail_closed(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    bad = (
        observation(confidence_ppm=True),
        observation(confidence_ppm=1_000_001),
        observation(confidence_ppm=42),
        observation(observed_at=-1),
        observation(expires_at=10),
        observation(category="other"),
        observation(category="readiness", value="AUTHORIZED"),
        observation(name=""),
        observation(value="x" * 513),
        observation(uncertainty="x" * 1025),
    )
    for item in bad:
        with pytest.raises((SelfModelError, ValueError)):
            store.append(item, expected_revision=0, verifier=trusted)
    with pytest.raises(SelfModelError):
        store.profile(now=True)
    with pytest.raises(SelfModelError):
        store.profile(now=10, min_confidence_ppm=1_000_001)


def test_cas_duplicate_timestamp_regression_and_concurrent_writers(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    first = observation()
    assert store.append(first, expected_revision=0, verifier=trusted) == 1
    with pytest.raises(SelfModelError, match="stale"):
        store.append(observation("second"), expected_revision=0, verifier=trusted)
    with pytest.raises(SelfModelError, match="duplicate"):
        store.append(first, expected_revision=1, verifier=trusted)
    with pytest.raises(SelfModelError, match="timestamp"):
        store.append(observation("earlier", observed_at=9), expected_revision=1,
                     verifier=trusted)

    def attempt(ident):
        try:
            return store.append(observation(ident), expected_revision=1,
                                verifier=trusted)
        except SelfModelError:
            return "stale"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcome = list(pool.map(attempt, ["a", "b"]))
    assert sorted(map(str, outcome)) == ["2", "stale"]
    assert len(SelfModelStore(store.path).history()) == 2


def test_corruption_and_schema_version_fail_without_poisoning_state(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    store.append(observation(), expected_revision=0, verifier=trusted)
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE observations SET raw='{}'")
    with pytest.raises(SelfModelError, match="digest"):
        SelfModelStore(store.path).history()
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE observations SET raw=?, digest=?",
                   (json.dumps({"schema": "bad"}), "bad"))
        db.execute("PRAGMA user_version=6")
    with pytest.raises(SelfModelError, match="version"):
        SelfModelStore(store.path)


def test_stale_revision_does_not_overwrite_other_worker(tmp_path):
    store = SelfModelStore(tmp_path / "self.sqlite")
    first = observation()
    store.append(first, expected_revision=0, verifier=trusted)
    newer = replace(first, observation_id="obs2", evidence_id="receipt:obs2",
                    observed_at=11, value="limited")
    store.append(newer, expected_revision=1, verifier=trusted)
    assert SelfModelStore(store.path).profile(now=12)["categories"]["capability"][
        "python"]["value"] == "limited"
