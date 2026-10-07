from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.capability_map import (
    CapabilityRegistry,
    CapabilityStatus,
    TestLevel,
    load_capability_registry,
)
from twelve_six.sil_qualification import (
    CommandExecution,
    GitState,
    SILScenario,
    build_package_manifest_bytes,
    build_sil_plan,
    load_sil_scenario,
    parse_vector_command,
    probe_git_state,
    qualify_sil,
    require_exact_clean_git_state,
    verify_sil_evidence,
)


_ROOT = Path(__file__).parents[1]
_REGISTRY = _ROOT / "configs" / "control" / "product_capabilities_v1.json"
_SCENARIO = _ROOT / "configs" / "control" / "sil_scenario_v1.json"
_GIT_SHA = "a" * 40


def _registry() -> CapabilityRegistry:
    return load_capability_registry(_REGISTRY)


def _scenario() -> SILScenario:
    return load_sil_scenario(_SCENARIO)


def _package_bytes() -> bytes:
    return build_package_manifest_bytes(_ROOT)


def _pass_runner(
    argv: tuple[str, ...],
    cwd: Path,
    timeout_seconds: int,
    input_envelope_bytes: bytes,
    expected_input_identity_sha256: str,
) -> CommandExecution:
    assert cwd
    assert timeout_seconds == 300
    assert hashlib.sha256(input_envelope_bytes).hexdigest() == (
        expected_input_identity_sha256
    )
    return CommandExecution(
        return_code=0,
        stdout="PASS " + " ".join(argv),
        stderr="",
        duration_ms=1,
        consumed_input_identity_sha256=expected_input_identity_sha256,
    )


def _git_probe(_: str | Path) -> GitState:
    return GitState(sha=_GIT_SHA, tracked_clean=True)


def _write_evidence(path: Path, evidence: dict[str, object]) -> None:
    path.write_text(
        json.dumps(
            evidence,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )


def _canonical_hash(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_evidence(evidence_path: Path, log_path: Path) -> dict[str, object]:
    return verify_sil_evidence(
        evidence_path,
        log_path,
        expected_package_bytes=_package_bytes(),
        expected_registry=_registry(),
        expected_scenario=_scenario(),
        expected_git_sha=_GIT_SHA,
    )


def test_independent_verifier_requires_exact_clean_checkout() -> None:
    assert require_exact_clean_git_state(
        _ROOT,
        _GIT_SHA,
        git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=True),
    ) == GitState(sha=_GIT_SHA, tracked_clean=True)

    with pytest.raises(ValueError, match="exact-head mismatch"):
        require_exact_clean_git_state(
            _ROOT,
            _GIT_SHA,
            git_probe=lambda _: GitState(sha="b" * 40, tracked_clean=True),
        )

    with pytest.raises(ValueError, match="dirty"):
        require_exact_clean_git_state(
            _ROOT,
            _GIT_SHA,
            git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=False),
        )


def test_sil_plan_executes_every_current_available_journey_with_integration_vectors() -> None:
    registry = _registry()
    plan = build_sil_plan(registry, _scenario())

    expected_available = tuple(
        journey.journey_id
        for journey in registry.journeys
        if registry.journey_available(journey.journey_id)
    )
    assert plan.available_journey_ids == expected_available
    assert plan.available_journey_ids

    planned_keys = {
        (vector.journey_id, vector.capability_id, vector.vector_id)
        for vector in plan.vectors
    }
    for journey in registry.journeys:
        if not registry.journey_available(journey.journey_id):
            continue
        for capability_id in journey.capability_ids:
            capability = registry.capability(capability_id)
            integration_ids = {
                vector.vector_id
                for vector in capability.test_vectors
                if vector.level is TestLevel.INTEGRATION
            }
            assert integration_ids
            assert integration_ids <= {
                vector_id
                for journey_id, planned_capability_id, vector_id in planned_keys
                if journey_id == journey.journey_id
                and planned_capability_id == capability_id
            }


def test_component_only_green_cannot_satisfy_available_capability_contract() -> None:
    registry = _registry()
    capabilities = list(registry.capabilities)
    index = next(
        index
        for index, capability in enumerate(capabilities)
        if capability.status is CapabilityStatus.AVAILABLE
    )
    capability = capabilities[index]
    component_only = tuple(
        vector
        for vector in capability.test_vectors
        if vector.level is TestLevel.COMPONENT
    )
    assert component_only

    with pytest.raises(ValueError, match="component and integration"):
        replace(capability, test_vectors=component_only)


@pytest.mark.parametrize(
    "command",
    (
        "python -m pytest -q tests/test_model.py",
        "pytest tests/test_model.py",
        "pytest -q ../tests/test_model.py",
        "pytest -q /tmp/test_model.py",
        "pytest -q tests/test_model.py::test_one",
        "pytest -q tests/test_model.py; echo forged",
        "pytest -q -k forged tests/test_model.py",
    ),
)
def test_sil_vector_parser_rejects_noncanonical_or_shell_like_commands(command: str) -> None:
    with pytest.raises(ValueError):
        parse_vector_command(command)


def test_sil_vector_parser_translates_checked_in_pytest_vector_without_shell() -> None:
    argv = parse_vector_command(
        "pytest -q tests/test_checkpointing.py tests/test_checkpoint_corruption_fail_closed.py"
    )

    assert argv[1:4] == ("-m", "pytest", "-q")
    assert argv[-2:] == (
        "tests/test_checkpointing.py",
        "tests/test_checkpoint_corruption_fail_closed.py",
    )


def test_qualify_sil_binds_exact_sha_identities_journeys_outputs_logs_and_verdict() -> None:
    registry = _registry()
    scenario = _scenario()

    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )

    assert evidence["git_sha"] == _GIT_SHA
    assert evidence["verdict"] == "PASS"
    assert evidence["available_journey_ids"]
    assert evidence["executions"]
    assert all(item["return_code"] == 0 for item in evidence["executions"])
    log_records = [json.loads(line) for line in log_text.splitlines()]
    assert len(log_records) == len(evidence["executions"])
    assert all(record["journey_id"] for record in log_records)
    for field in (
        "package_identity_sha256",
        "capability_registry_identity_sha256",
        "model_spec_identity_sha256",
        "init_spec_identity_sha256",
        "data_identity_sha256",
        "scenario_identity_sha256",
        "input_identity_sha256",
        "output_identity_sha256",
        "log_sha256",
        "evidence_identity_sha256",
    ):
        value = evidence[field]
        assert isinstance(value, str)
        assert len(value) == 64
    assert evidence["scientific_boundary"] == {
        "corpus_admission_authorized": False,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights_used": False,
    }


def test_sil_fail_execution_cannot_become_pass() -> None:
    calls = 0

    def fail_once(
        argv: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int,
        input_envelope_bytes: bytes,
        expected_input_identity_sha256: str,
    ) -> CommandExecution:
        nonlocal calls
        calls += 1
        result = _pass_runner(
            argv,
            cwd,
            timeout_seconds,
            input_envelope_bytes,
            expected_input_identity_sha256,
        )
        if calls == 1:
            return replace(result, return_code=7, stderr="integration failure")
        return result

    evidence, _ = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=fail_once,
        git_probe=_git_probe,
    )

    assert evidence["verdict"] == "FAIL"
    assert any(item["return_code"] != 0 for item in evidence["executions"])


def test_sil_mismatched_consumed_input_identity_cannot_become_pass() -> None:
    def consume_wrong_identity(
        argv: tuple[str, ...],
        cwd: Path,
        timeout_seconds: int,
        input_envelope_bytes: bytes,
        expected_input_identity_sha256: str,
    ) -> CommandExecution:
        result = _pass_runner(
            argv,
            cwd,
            timeout_seconds,
            input_envelope_bytes,
            expected_input_identity_sha256,
        )
        return replace(
            result,
            consumed_input_identity_sha256="b" * 64,
        )

    evidence, _ = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=consume_wrong_identity,
        git_probe=_git_probe,
    )

    assert evidence["verdict"] == "FAIL"
    assert any(
        item["consumed_input_identity_sha256"]
        != item["expected_input_identity_sha256"]
        for item in evidence["executions"]
    )


def test_sil_rejects_tracked_checkout_mutation_during_vector_execution() -> None:
    states = iter(
        (
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=False),
        )
    )

    with pytest.raises(ValueError, match="became dirty during SIL vector execution"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=_package_bytes(),
            command_runner=_pass_runner,
            git_probe=lambda _: next(states),
        )


def test_probe_git_state_rejects_untracked_nonignored_checkout_drift(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("tracked\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=12-6 SIL test",
            "-c",
            "user.email=sil-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = probe_git_state(tmp_path)
    assert len(clean.sha) == 40
    assert clean.tracked_clean is True

    (tmp_path / "untracked-influence.py").write_text("FORGED = True\n", encoding="utf-8")

    drifted = probe_git_state(tmp_path)
    assert drifted.sha == clean.sha
    assert drifted.tracked_clean is False


def test_sil_rejects_git_head_mismatch_and_dirty_tracked_checkout() -> None:
    registry = _registry()
    scenario = _scenario()

    with pytest.raises(ValueError, match="exact-head mismatch"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha="b" * 40, tracked_clean=True),
        )

    with pytest.raises(ValueError, match="dirty"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=False),
        )


def test_unavailable_journeys_are_evidenced_as_blocked_not_simulated() -> None:
    registry = _registry()
    plan = build_sil_plan(registry, _scenario())

    expected_unavailable = {
        journey.journey_id
        for journey in registry.journeys
        if not registry.journey_available(journey.journey_id)
    }
    observed = {item.journey_id for item in plan.unavailable_journeys}
    assert observed == expected_unavailable
    for item in plan.unavailable_journeys:
        assert item.blocking_capability_ids
        assert all(item.reasons)


def test_strict_sil_scenario_rejects_duplicate_unknown_and_bool_timeout(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(
        b'{"schema_version":1,"schema_version":1,"scenario_id":"x",'
        b'"journey_selector":"ALL_AVAILABLE","fixture_policy":"DETERMINISTIC_SYNTHETIC",'
        b'"synthetic_data_utf8":"x","timeout_seconds_per_vector":1}'
    )
    with pytest.raises(ValueError, match="strict unambiguous"):
        load_sil_scenario(duplicate)

    payload = json.loads(_SCENARIO.read_text(encoding="utf-8"))
    payload["forged"] = True
    unknown = tmp_path / "unknown.json"
    unknown.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields"):
        load_sil_scenario(unknown)

    del payload["forged"]
    payload["timeout_seconds_per_vector"] = True
    bool_timeout = tmp_path / "bool-timeout.json"
    bool_timeout.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="timeout_seconds_per_vector"):
        load_sil_scenario(bool_timeout)


def test_evidence_verifier_rejects_log_and_evidence_resealing(tmp_path: Path) -> None:
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "sil.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    verified = _verify_evidence(evidence_path, log_path)
    assert verified["verdict"] == "PASS"

    log_path.write_text(log_text + "forged", encoding="utf-8")
    with pytest.raises(ValueError, match="log identity"):
        _verify_evidence(evidence_path, log_path)

    log_path.write_text(log_text, encoding="utf-8")
    resealed = dict(evidence)
    resealed["scenario_id"] = "forged-scenario"
    _write_evidence(evidence_path, resealed)
    with pytest.raises(ValueError, match="evidence identity"):
        _verify_evidence(evidence_path, log_path)


def test_verifier_rejects_resealed_invalid_timings(
    tmp_path: Path,
) -> None:
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    evidence["timings"] = {
        "started_unix_ns": 2,
        "finished_unix_ns": 1,
        "duration_ms": -1,
    }
    unsigned = dict(evidence)
    unsigned.pop("evidence_identity_sha256")
    evidence["evidence_identity_sha256"] = _canonical_hash(unsigned)

    evidence_path = tmp_path / "resealed-invalid-timings.json"
    log_path = tmp_path / "sil.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(ValueError, match="timing"):
        _verify_evidence(evidence_path, log_path)


def test_verifier_rejects_total_duration_shorter_than_execution_sum(
    tmp_path: Path,
) -> None:
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    assert sum(item["duration_ms"] for item in evidence["executions"]) > 0
    evidence["timings"]["duration_ms"] = 0
    unsigned = dict(evidence)
    unsigned.pop("evidence_identity_sha256")
    evidence["evidence_identity_sha256"] = _canonical_hash(unsigned)

    evidence_path = tmp_path / "resealed-short-total-duration.json"
    log_path = tmp_path / "sil.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(ValueError, match="shorter than sequential"):
        _verify_evidence(evidence_path, log_path)


def test_verifier_rejects_self_consistent_execution_hash_reseal_against_log(
    tmp_path: Path,
) -> None:
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    evidence["executions"][0]["stdout_sha256"] = "b" * 64
    evidence["output_identity_sha256"] = _canonical_hash(
        {
            "available_journey_ids": evidence["available_journey_ids"],
            "unavailable_journeys": evidence["unavailable_journeys"],
            "executions": evidence["executions"],
        }
    )
    unsigned = dict(evidence)
    unsigned.pop("evidence_identity_sha256")
    evidence["evidence_identity_sha256"] = _canonical_hash(unsigned)

    evidence_path = tmp_path / "resealed-execution-evidence.json"
    log_path = tmp_path / "sil.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(ValueError, match="log stdout identity"):
        _verify_evidence(evidence_path, log_path)


def test_verifier_rejects_self_consistent_package_authority_reseal(
    tmp_path: Path,
) -> None:
    registry = _registry()
    scenario = _scenario()
    evidence, log_text = qualify_sil(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    plan = build_sil_plan(registry, scenario)
    evidence["package_identity_sha256"] = "b" * 64
    evidence["input_identity_sha256"] = _canonical_hash(
        {
            "git_sha": evidence["git_sha"],
            "package_identity_sha256": evidence["package_identity_sha256"],
            "capability_registry_identity_sha256": evidence[
                "capability_registry_identity_sha256"
            ],
            "model_spec_identity_sha256": evidence["model_spec_identity_sha256"],
            "init_spec_identity_sha256": evidence["init_spec_identity_sha256"],
            "data_identity_sha256": evidence["data_identity_sha256"],
            "scenario_identity_sha256": evidence["scenario_identity_sha256"],
            "plan": plan.to_dict(),
        }
    )
    unsigned = dict(evidence)
    unsigned.pop("evidence_identity_sha256")
    evidence["evidence_identity_sha256"] = _canonical_hash(unsigned)

    evidence_path = tmp_path / "resealed-evidence.json"
    log_path = tmp_path / "sil.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    with pytest.raises(ValueError, match="package_identity_sha256"):
        _verify_evidence(evidence_path, log_path)


def test_sil_uses_single_shared_workflow_and_exact_head_checkout() -> None:
    workflow_dir = _ROOT / ".github" / "workflows"
    workflows = sorted(
        path.name
        for path in workflow_dir.iterdir()
        if path.suffix in {".yml", ".yaml"}
    )
    assert workflows == ["ci.yml"]

    workflow = (workflow_dir / "ci.yml").read_text(encoding="utf-8")
    assert "  sil-current-capability-journeys:" in workflow
    assert "needs: bootstrap" in workflow
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in workflow
    assert 'test "$(git rev-parse HEAD)" = "$SIL_EXPECTED_SHA"' in workflow
    assert "python -m twelve_six.sil_qualification run" in workflow
    assert "python -m twelve_six.sil_qualification verify" in workflow
    assert "continue-on-error: true" in workflow
    assert "if: always()" in workflow

def test_package_identity_binds_tracked_package_source_manifest() -> None:
    raw = _package_bytes()
    manifest = json.loads(raw.decode("utf-8"))

    assert manifest["schema_version"] == "12-6.package-source-manifest.v1"
    paths = [item["path"] for item in manifest["files"]]
    assert paths == sorted(paths)
    assert "pyproject.toml" in paths
    assert "src/twelve_six/sil_qualification.py" in paths
    assert any(path.startswith("configs/research/") for path in paths)
    for item in manifest["files"]:
        source = (_ROOT / item["path"]).read_bytes()
        assert item["bytes"] == len(source)
        assert item["sha256"] == hashlib.sha256(source).hexdigest()


def test_qualify_sil_rejects_opaque_package_bytes_not_bound_to_checkout() -> None:
    with pytest.raises(ValueError, match="exact tracked package source manifest"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=b"opaque-unbound-package",
            command_runner=_pass_runner,
            git_probe=_git_probe,
        )

