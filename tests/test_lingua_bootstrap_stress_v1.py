from __future__ import annotations

from tools.verify_lingua_bootstrap_stress_v1 import UPSTREAM, canonical_bytes, classify, run


def test_exact_lingua_pin_is_immutable() -> None:
    assert UPSTREAM["release"] == "v2.1.1"
    assert UPSTREAM["tag_sha"] == "7ce57e41af5ca9ce4630dac3d8e446dffe40513a"
    assert UPSTREAM["commit_sha"] == "31572a7b1957714364a8fafd24ab248c9ed15d68"
    assert UPSTREAM["wheel"] == (
        "lingua_language_detector-2.1.1-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
    )
    assert UPSTREAM["wheel_sha256"] == "2a468c3fc9eaa6db733a347fee768fe171e76fac2c4bc49951e26bc79aec6a2a"


def test_selected_version_python_constraint_is_project_compatible() -> None:
    assert UPSTREAM["python_requires"] == ">=3.10,<3.14"
    assert UPSTREAM["license"] == "Apache-2.0"


def test_adoptable_requires_real_runtime() -> None:
    assert classify(True, "PASS", "PASS") == "ADOPTABLE_COMPONENT"
    assert classify(False, "PASS", "PASS") == "BLOCKED_ENVIRONMENT"
    assert classify(True, "FAIL", "NOT_EXECUTED") == "RETEST_RUNTIME_REQUIRED"


def test_missing_command_is_detected_without_exception() -> None:
    assert run(["__swarm773_command_that_does_not_exist__"])[0] == 127


def test_canonical_json_is_stable() -> None:
    payload = {"z": 1, "a": [True, "x"]}
    assert canonical_bytes(payload) == canonical_bytes(payload)
    assert canonical_bytes(payload) == b'{"a":[true,"x"],"z":1}'
