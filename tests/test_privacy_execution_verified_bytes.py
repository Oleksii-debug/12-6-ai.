from __future__ import annotations

import importlib
import sys

import pytest

from twelve_six.data import privacy_execution_authority as g06

_PROBE = "Contact alice@real-domain.dev with token=AbCd1234!fixture"


def _bound_probe() -> tuple[dict[str, object], dict[str, str]]:
    scan, binding = g06._privacy_binding()
    return scan(_PROBE).evidence(), binding


def test_forged_live_scanner_module_name_cannot_redirect_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _bound_probe()
    live = importlib.import_module(g06.PRIVACY_MODULE)

    def poisoned_scan(text: str):
        raise AssertionError(f"live forged scanner executed: {text!r}")

    poisoned_scan.__module__ = g06.PRIVACY_MODULE
    monkeypatch.setattr(live, "hash_safe_scan", poisoned_scan)

    assert _bound_probe() == baseline


def test_forged_live_policy_module_name_cannot_redirect_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _bound_probe()
    live = importlib.import_module(g06.PRIVACY_MODULE)

    def poisoned_policy() -> dict[str, object]:
        raise AssertionError("live forged policy provider executed")

    poisoned_policy.__module__ = g06.PRIVACY_MODULE
    monkeypatch.setattr(live, "policy_manifest", poisoned_policy)

    assert _bound_probe() == baseline


def test_live_helper_and_action_table_mutation_do_not_affect_verified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _bound_probe()
    live = importlib.import_module(g06.PRIVACY_MODULE)

    def poisoned_detect(text: str):
        raise AssertionError(f"live poisoned helper executed: {text!r}")

    monkeypatch.setattr(live, "detect", poisoned_detect)
    monkeypatch.setitem(live.DETECTOR_ACTIONS, "email", "ALLOW")
    monkeypatch.setitem(live.DETECTOR_ACTIONS, "environment_secret_assignment", "ALLOW")

    assert _bound_probe() == baseline


def test_verified_private_namespace_is_removed_after_binding() -> None:
    _bound_probe()
    private_name = (
        "_twelve_six_g06_privacy_"
        + g06.EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1
    )
    assert private_name not in sys.modules
