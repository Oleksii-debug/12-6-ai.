from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six import sil_qualification
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
    canonical_sil_environment_receipt_v1,
    load_sil_environment_receipt,
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


def _environment_receipt() -> dict[str, object]:
    return canonical_sil_environment_receipt_v1()


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
        expected_environment_receipt=_environment_receipt(),
        expected_registry=_registry(),
        expected_scenario=_scenario(),
        expected_git_sha=_GIT_SHA,
    )


def test_public_verifier_requires_exact_git_sha_authority(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="expected_git_sha"):
        verify_sil_evidence(
            tmp_path / "missing-evidence.json",
            tmp_path / "missing-log.jsonl",
            expected_package_bytes=b"candidate",
            expected_environment_receipt={},
            expected_registry=_registry(),
            expected_scenario=_scenario(),
        )


def test_environment_receipt_authority_ignores_module_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    canonical_authority = canonical_sil_environment_receipt_v1
    expected = canonical_authority()

    monkeypatch.setattr(
        sil_qualification,
        "_SIL_ENVIRONMENT_LOCK_SOURCE_COMMIT",
        "f" * 40,
    )
    monkeypatch.setattr(
        sil_qualification,
        "_SIL_ENVIRONMENT_LOCKS",
        (("forged", "requirements/forged.lock.txt", "f" * 64),),
    )
    monkeypatch.setattr(
        sil_qualification,
        "_SIL_ENVIRONMENT_PACKAGES",
        (("forged-package", "999"),),
    )
    forged = dict(expected)
    forged["lock_source_commit"] = "f" * 40
    forged["locks"] = [
        {
            "role": "forged",
            "path": "requirements/forged.lock.txt",
            "sha256": "f" * 64,
        }
    ]
    forged["packages"] = [{"name": "forged-package", "version": "999"}]
    unsigned = dict(forged)
    unsigned.pop("identity_sha256")
    forged["identity_sha256"] = _canonical_hash(unsigned)

    assert canonical_authority() == expected

    path = tmp_path / "forged-environment.json"
    path.write_text(
        json.dumps(
            forged,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        load_sil_environment_receipt(path)


def test_canonical_environment_receipt_binds_pinned_historical_lock_source(
    tmp_path: Path,
) -> None:
    receipt = _environment_receipt()

    assert receipt["lock_source_commit"] == "029514654829cebc149cff6fc1fea2a8ba4fa566"
    assert receipt["python"] == {
        "implementation": "cpython",
        "version": "3.11.16",
    }
    assert [item["role"] for item in receipt["locks"]] == [
        "toolchain",
        "cpu_runtime",
        "dev",
    ]
    assert len(receipt["packages"]) == 21

    path = tmp_path / "environment.json"
    path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    assert load_sil_environment_receipt(path) == receipt


def test_environment_receipt_rejects_byte_or_lock_source_reseal(tmp_path: Path) -> None:
    receipt = _environment_receipt()
    path = tmp_path / "environment.json"

    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bytes are non-canonical"):
        load_sil_environment_receipt(path)

    forged = dict(receipt)
    forged["lock_source_commit"] = "b" * 40
    unsigned = dict(forged)
    unsigned.pop("identity_sha256")
    forged["identity_sha256"] = _canonical_hash(unsigned)
    path.write_text(
        json.dumps(forged, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        load_sil_environment_receipt(path)


def test_independent_verifier_requires_exact_clean_checkout() -> None:
    assert sil_qualification._require_exact_clean_git_state_with_probe(
        _ROOT,
        _GIT_SHA,
        git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=True),
    ) == GitState(sha=_GIT_SHA, tracked_clean=True)

    with pytest.raises(ValueError, match="exact-head mismatch"):
        sil_qualification._require_exact_clean_git_state_with_probe(
            _ROOT,
            _GIT_SHA,
            git_probe=lambda _: GitState(sha="b" * 40, tracked_clean=True),
        )

    with pytest.raises(ValueError, match="dirty"):
        sil_qualification._require_exact_clean_git_state_with_probe(
            _ROOT,
            _GIT_SHA,
            git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=False),
        )


def test_public_qualify_sil_ignores_rebound_package_manifest_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged_package = b"forged-package-manifest"
    monkeypatch.setattr(
        sil_qualification,
        "build_package_manifest_bytes",
        lambda _root: forged_package,
    )

    with pytest.raises(
        ValueError,
        match="package_bytes do not match exact tracked package source manifest",
    ):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=forged_package,
            environment_receipt=_environment_receipt(),
        )


def test_public_sil_authorities_reject_caller_supplied_execution_backends() -> None:
    registry = _registry()
    scenario = _scenario()

    with pytest.raises(TypeError):
        qualify_sil(  # type: ignore[call-arg]
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=_git_probe,
        )

    with pytest.raises(TypeError):
        require_exact_clean_git_state(  # type: ignore[call-arg]
            _ROOT,
            _GIT_SHA,
            git_probe=_git_probe,
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


def test_sil_policy_predeclares_closed_predecessor_journey_contract() -> None:
    policy = dict(sil_qualification._CANONICAL_JOURNEY_E2E_VECTOR_POLICY)

    assert policy["developer-replace-cognitive-core"] == (
        "section0-product-stack",
        "section1-stack",
    )
    assert policy["maintainer-capability-qualification"] == ("section2-stack",)


def test_sil_plan_binds_explicit_end_to_end_contracts_to_exact_journey_steps() -> None:
    registry = _registry()
    plan = build_sil_plan(registry, _scenario())

    contract_ids = tuple(
        journey_id for journey_id, _vector_ids in plan.journey_end_to_end_contracts
    )
    assert contract_ids == plan.available_journey_ids

    serialized = plan.to_dict()["journey_end_to_end_contracts"]
    assert [item["journey_id"] for item in serialized] == list(
        plan.available_journey_ids
    )
    assert all(
        item["execution_mode"] == "SEQUENTIAL_SHARED_INPUT_ENVELOPE"
        and item["completion_rule"] == "ALL_DECLARED_STEPS_PASS_IN_ORDER"
        for item in serialized
    )

    for journey_id, declared_vector_ids in plan.journey_end_to_end_contracts:
        observed = tuple(
            vector.vector_id
            for vector in plan.vectors
            if vector.journey_id == journey_id
        )
        assert observed == declared_vector_ids


def test_sil_plan_rejects_missing_or_resealed_end_to_end_policy() -> None:
    registry = _registry()
    scenario = _scenario()
    plan = build_sil_plan(registry, scenario)

    with pytest.raises(ValueError, match="lacks explicit end-to-end contract"):
        sil_qualification._build_sil_plan_with_policy(
            registry,
            scenario,
            e2e_policy=plan.journey_end_to_end_contracts[:-1],
        )

    forged = list(plan.journey_end_to_end_contracts)
    journey_id, _vector_ids = forged[0]
    forged[0] = (journey_id, ("model-resource-envelope",))
    with pytest.raises(ValueError, match="do not match"):
        sil_qualification._build_sil_plan_with_policy(
            registry,
            scenario,
            e2e_policy=tuple(forged),
        )


def test_public_sil_plan_rejects_caller_supplied_policy_authority() -> None:
    plan = build_sil_plan(_registry(), _scenario())

    with pytest.raises(TypeError, match="_sealed_e2e_policy"):
        build_sil_plan(
            _registry(),
            _scenario(),
            _sealed_e2e_policy=plan.journey_end_to_end_contracts,
        )


def test_sil_plan_policy_global_rebind_cannot_reseal_loaded_validator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = build_sil_plan(_registry(), _scenario()).journey_end_to_end_contracts
    monkeypatch.setattr(
        sil_qualification,
        "_CANONICAL_JOURNEY_E2E_VECTOR_POLICY",
        (("developer-model-contract", ("forged-vector",)),),
    )

    rebound = build_sil_plan(_registry(), _scenario())

    assert rebound.journey_end_to_end_contracts == original


def test_sil_plan_enum_policy_global_rebind_cannot_reseal_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = build_sil_plan(_registry(), _scenario()).to_dict()

    class ForgedStatus:
        AVAILABLE = object()
        UNAVAILABLE = object()

    class ForgedLevel:
        INTEGRATION = object()

    monkeypatch.setattr(sil_qualification, "CapabilityStatus", ForgedStatus)
    monkeypatch.setattr(sil_qualification, "TestLevel", ForgedLevel)

    assert build_sil_plan(_registry(), _scenario()).to_dict() == baseline


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

    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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

    evidence, _ = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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

    evidence, _ = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=lambda _: next(states),
        )


def test_sil_rejects_checkout_mutation_before_evidence_sealing() -> None:
    registry = _registry()
    scenario = _scenario()
    plan = build_sil_plan(registry, scenario)
    final_probe_call = 2 * len(plan.vectors) + 2
    calls = 0

    def mutate_only_at_final_seal(_: str | Path) -> GitState:
        nonlocal calls
        calls += 1
        return GitState(
            sha=_GIT_SHA,
            tracked_clean=calls != final_probe_call,
        )

    with pytest.raises(ValueError, match="dirty before SIL evidence sealing"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=mutate_only_at_final_seal,
        )

    assert calls == final_probe_call


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


def test_sil_subprocess_environment_rejects_host_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_must_execute.py").write_text(
        "def test_must_execute() -> None:\n    assert False\n",
        encoding="utf-8",
    )
    # A tracked repo-root pytest.py would shadow the real test runner unless the
    # child interpreter is started with Python safe-path semantics.
    (tmp_path / "pytest.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=12-6 SIL env test",
            "-c",
            "user.email=sil-env-test@example.invalid",
            "commit",
            "-q",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
    )

    monkeypatch.setenv("GIT_DIR", str(tmp_path / "redirected-git-dir"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "redirected-work-tree"))
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "redirected-pythonpath"))

    child_env = sil_qualification._qualification_subprocess_env()
    assert "GIT_DIR" not in child_env
    assert "GIT_WORK_TREE" not in child_env
    assert "PYTEST_ADDOPTS" not in child_env
    assert "PYTHONPATH" not in child_env
    assert child_env["GIT_OPTIONAL_LOCKS"] == "0"
    assert child_env["PYTHONHASHSEED"] == "0"
    assert child_env["PYTHONNOUSERSITE"] == "1"
    assert child_env["PYTHONSAFEPATH"] == "1"
    assert child_env["PYTHONUTF8"] == "1"
    assert child_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"

    state = probe_git_state(tmp_path)
    assert state.tracked_clean is True

    input_bytes = b'{"probe":"must-execute"}'
    input_identity = hashlib.sha256(input_bytes).hexdigest()
    execution = sil_qualification.run_command(
        (
            sil_qualification.sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_must_execute.py",
        ),
        tmp_path,
        30,
        input_bytes,
        input_identity,
    )
    assert execution.consumed_input_identity_sha256 == input_identity
    assert execution.return_code != 0


def test_sil_command_runner_never_inherits_stdin_or_enables_shell(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_bytes = b'{"stdin":"sealed"}'
    input_identity = hashlib.sha256(input_bytes).hexdigest()
    observed: dict[str, object] = {}

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=f"SIL_INPUT_VERIFIED_SHA256={input_identity}\n",
            stderr="",
        )

    monkeypatch.setattr(sil_qualification.subprocess, "run", fake_run)

    execution = sil_qualification.run_command(
        (
            sil_qualification.sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_must_execute.py",
        ),
        tmp_path,
        30,
        input_bytes,
        input_identity,
    )

    assert observed["stdin"] is subprocess.DEVNULL
    assert observed["shell"] is False
    assert execution.return_code == 0
    assert execution.consumed_input_identity_sha256 == input_identity


def test_sil_rejects_git_head_mismatch_and_dirty_tracked_checkout() -> None:
    registry = _registry()
    scenario = _scenario()

    with pytest.raises(ValueError, match="exact-head mismatch"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=lambda _: GitState(sha="b" * 40, tracked_clean=True),
        )

    with pytest.raises(ValueError, match="dirty"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
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


def test_sealed_sil_policy_keeps_known_unavailable_journey_contracts_dormant() -> None:
    registry = _registry()
    plan = build_sil_plan(registry, _scenario())

    unavailable_ids = {item.journey_id for item in plan.unavailable_journeys}
    contract_ids = {journey_id for journey_id, _ in plan.journey_end_to_end_contracts}

    assert "maintainer-project-control" in unavailable_ids
    assert "maintainer-sil-qualification" in unavailable_ids
    assert "maintainer-project-control" not in plan.available_journey_ids
    assert "maintainer-sil-qualification" not in plan.available_journey_ids
    assert "maintainer-project-control" not in contract_ids
    assert "maintainer-sil-qualification" not in contract_ids


def test_sil_policy_rejects_unknown_dormant_journey_contract() -> None:
    registry = _registry()
    plan = build_sil_plan(registry, _scenario())
    forged_policy = (
        *plan.journey_end_to_end_contracts,
        ("forged-dormant-journey", ("section3-sil-stack",)),
    )

    with pytest.raises(ValueError, match="references unknown journey"):
        sil_qualification._build_sil_plan_with_policy(
            registry,
            _scenario(),
            e2e_policy=forged_policy,
        )


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
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=_registry(),
        scenario=_scenario(),
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
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
    marker = "  sil-current-capability-journeys:"
    assert marker in workflow
    sil_job = marker + workflow.split(marker, 1)[1]
    assert "needs: bootstrap" in sil_job
    assert "ref: ${{ github.event.pull_request.head.sha || github.sha }}" in sil_job
    assert 'test "$(git rev-parse HEAD)" = "$SIL_EXPECTED_SHA"' in sil_job
    assert (
        'SIL_ENV_LOCK_SOURCE_COMMIT: "029514654829cebc149cff6fc1fea2a8ba4fa566"'
        in sil_job
    )
    assert "git fetch --no-tags origin refs/pull/402/head" in sil_job
    assert 'test "$(git rev-parse FETCH_HEAD)" = "$SIL_ENV_LOCK_SOURCE_COMMIT"' in sil_job
    assert "requirements/locks/linux-x86_64/toolchain.lock.txt" in sil_job
    assert "requirements/execution/linux-x86_64/cpu-runtime.lock.txt" in sil_job
    assert "requirements/locks/linux-x86_64/dev.lock.txt" in sil_job
    assert "--require-hashes --no-deps" in sil_job
    assert "--no-deps --no-build-isolation -e ." in sil_job
    assert "python -m pip install --upgrade pip" not in sil_job
    assert "pip install -e .[dev]" not in sil_job
    assert '"$sil_python" -m twelve_six.sil_qualification environment-receipt' in sil_job
    assert sil_job.count(
        '--environment-receipt "$RUNNER_TEMP/sil-environment.json"'
    ) == 2
    assert '"$RUNNER_TEMP/sil-venv/bin/python" -m twelve_six.sil_qualification run' in sil_job
    assert '"$RUNNER_TEMP/sil-venv/bin/python" -m twelve_six.sil_qualification verify' in sil_job
    assert "continue-on-error: true" in sil_job
    assert "if: always()" in sil_job


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
        assert item["git_mode"] in {"100644", "100755"}
        assert len(item["git_blob_sha"]) == 40
        assert item["bytes"] == len(source)
        assert item["sha256"] == hashlib.sha256(source).hexdigest()


def test_qualify_sil_rejects_opaque_package_bytes_not_bound_to_checkout() -> None:
    with pytest.raises(ValueError, match="exact tracked package source manifest"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=b"opaque-unbound-package",
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=_git_probe,
        )


def test_package_manifest_identity_binds_git_mode_and_blob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "twelve_six").mkdir(parents=True)
    pyproject = b"[project]\nname='fixture'\n"
    module = b"VALUE = 1\n"
    (tmp_path / "pyproject.toml").write_bytes(pyproject)
    (tmp_path / "src" / "twelve_six" / "module.py").write_bytes(module)

    def git_blob_sha(raw: bytes) -> str:
        header = f"blob {len(raw)}\0".encode("ascii")
        return hashlib.sha1(
            header + raw,
            usedforsecurity=False,
        ).hexdigest()

    mode = {"module": "100644"}

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        stdout = (
            "100644 "
            + git_blob_sha(pyproject)
            + " 0\tpyproject.toml\0"
            + mode["module"]
            + " "
            + git_blob_sha(module)
            + " 0\tsrc/twelve_six/module.py\0"
        )
        return subprocess.CompletedProcess(
            args=["git", "ls-files", "--stage", "-z"],
            returncode=0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(sil_qualification.subprocess, "run", fake_run)
    mode_644 = build_package_manifest_bytes(tmp_path)
    mode["module"] = "100755"
    mode_755 = build_package_manifest_bytes(tmp_path)

    assert mode_644 != mode_755
    manifest = json.loads(mode_755.decode("utf-8"))
    module_entry = next(
        item for item in manifest["files"] if item["path"].endswith("module.py")
    )
    assert module_entry["git_mode"] == "100755"
    assert module_entry["git_blob_sha"] == git_blob_sha(module)


def test_package_manifest_preserves_raw_unicode_git_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "twelve_six").mkdir(parents=True)
    pyproject = b"[project]\nname='fixture'\n"
    module = b"VALUE = 2\n"
    path = "src/twelve_six/перевірка.py"
    (tmp_path / "pyproject.toml").write_bytes(pyproject)
    (tmp_path / path).write_bytes(module)

    def git_blob_sha(raw: bytes) -> str:
        header = f"blob {len(raw)}\0".encode("ascii")
        return hashlib.sha1(
            header + raw,
            usedforsecurity=False,
        ).hexdigest()

    stdout = (
        "100644 "
        + git_blob_sha(pyproject)
        + " 0\tpyproject.toml\0"
        + "100644 "
        + git_blob_sha(module)
        + f" 0\t{path}\0"
    )
    result = subprocess.CompletedProcess(
        args=["git", "ls-files", "--stage", "-z"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    monkeypatch.setattr(
        sil_qualification.subprocess,
        "run",
        lambda *_args, **_kwargs: result,
    )

    manifest = json.loads(build_package_manifest_bytes(tmp_path).decode("utf-8"))

    assert path in [item["path"] for item in manifest["files"]]


def test_package_manifest_rejects_worktree_bytes_not_matching_index_blob(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src" / "twelve_six").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_bytes(b"[project]\n")
    (tmp_path / "src" / "twelve_six" / "module.py").write_bytes(b"VALUE = 3\n")
    stdout = (
        "100644 "
        + ("a" * 40)
        + " 0\tpyproject.toml\0"
        + "100644 "
        + ("b" * 40)
        + " 0\tsrc/twelve_six/module.py\0"
    )
    result = subprocess.CompletedProcess(
        args=["git", "ls-files", "--stage", "-z"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    monkeypatch.setattr(
        sil_qualification.subprocess,
        "run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(ValueError, match="do not match Git blob"):
        build_package_manifest_bytes(tmp_path)


def test_package_manifest_rejects_symlink_git_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stdout = (
        "100644 " + ("a" * 40) + " 0\tpyproject.toml\0"
        "120000 " + ("b" * 40) + " 0\tsrc/twelve_six/forged.py\0"
    )
    result = subprocess.CompletedProcess(
        args=["git", "ls-files", "--stage"],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    monkeypatch.setattr(
        sil_qualification.subprocess,
        "run",
        lambda *_args, **_kwargs: result,
    )

    with pytest.raises(ValueError, match="regular Git file"):
        build_package_manifest_bytes(tmp_path)


def test_sil_plan_rejects_actual_vector_resealing() -> None:
    plan = build_sil_plan(_registry(), _scenario())
    first = plan.vectors[0]
    forged = sil_qualification.PlannedVector(
        first.journey_id,
        first.capability_id,
        "forged-vector",
        first.argv,
    )

    with pytest.raises(ValueError, match="actual vectors do not match declared"):
        sil_qualification.SILPlan(
            plan.available_journey_ids,
            plan.unavailable_journeys,
            plan.journey_end_to_end_contracts,
            (forged, *plan.vectors[1:]),
        )


def test_sil_builder_rejects_scenario_behavioral_subclass_resealing() -> None:
    scenario = _scenario()

    class ForgedScenario(SILScenario):
        def identity_sha256(self) -> str:
            return "f" * 64

    forged = ForgedScenario(
        scenario.schema_version,
        scenario.scenario_id,
        scenario.journey_selector,
        scenario.fixture_policy,
        scenario.synthetic_data_utf8,
        scenario.timeout_seconds_per_vector,
    )

    with pytest.raises(ValueError, match="scenario must be a SILScenario"):
        build_sil_plan(_registry(), forged)


def test_exact_git_state_rejects_behavioral_subclass() -> None:
    class ForgedGitState(GitState):
        pass

    with pytest.raises(ValueError, match="git probe must return GitState"):
        sil_qualification._require_exact_clean_git_state_with_probe(
            _ROOT,
            _GIT_SHA,
            git_probe=lambda _: ForgedGitState(sha=_GIT_SHA, tracked_clean=True),
        )


def test_command_execution_contract_rejects_bool_return_code_and_bad_consumed_hash() -> None:
    with pytest.raises(ValueError, match="return_code"):
        CommandExecution(True, "", "", 0, None)

    with pytest.raises(ValueError, match="consumed_input_identity_sha256"):
        CommandExecution(0, "", "", 0, "not-a-sha")


def test_closed_sil_scalar_and_container_boundaries_reject_behavioral_subclasses() -> None:
    class ForgedStr(str):
        def strip(self) -> str:
            return "forged-valid"

    class ForgedBytes(bytes):
        def decode(self, *args: object, **kwargs: object) -> str:
            raise AssertionError("behavioral bytes subclass must not be decoded")

    class ForgedTuple(tuple):
        pass

    with pytest.raises(ValueError, match="SIL scenario input must be bytes"):
        sil_qualification._strict_json_object(
            ForgedBytes(b"{}"),
            maximum_bytes=1024,
            label="SIL scenario",
        )

    scenario = _scenario()
    with pytest.raises(ValueError, match="scenario_id must be a canonical identifier"):
        replace(scenario, scenario_id=ForgedStr(scenario.scenario_id))

    plan = build_sil_plan(_registry(), scenario)
    vector = plan.vectors[0]
    with pytest.raises(ValueError, match="planned vector argv must be canonical"):
        sil_qualification.PlannedVector(
            vector.journey_id,
            vector.capability_id,
            vector.vector_id,
            ForgedTuple(vector.argv),
        )

    with pytest.raises(ValueError, match="stdout/stderr must be text"):
        CommandExecution(0, ForgedStr("stdout"), "", 0, None)

    with pytest.raises(ValueError, match="SIL log must be non-empty bytes"):
        sil_qualification._load_sil_log_records(ForgedBytes(b"{}\n"))


def test_sil_objects_revalidate_after_post_construction_mutation() -> None:
    scenario = _scenario()
    object.__setattr__(scenario, "timeout_seconds_per_vector", 0)

    with pytest.raises(ValueError, match="timeout_seconds_per_vector"):
        scenario.identity_sha256()

    plan = build_sil_plan(_registry(), _scenario())
    vector = plan.vectors[0]
    object.__setattr__(vector, "argv", ("python", "-c", "forged"))

    with pytest.raises(ValueError, match="planned vector argv"):
        plan.to_dict()


def test_sil_stored_state_authority_ignores_class_method_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _registry()
    scenario = _scenario()
    baseline_evidence, baseline_log = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )

    for cls in (
        sil_qualification.SILScenario,
        sil_qualification.PlannedVector,
        sil_qualification.UnavailableJourney,
        sil_qualification.SILPlan,
    ):
        monkeypatch.setattr(cls, "__post_init__", lambda _self: None)
    monkeypatch.setattr(
        sil_qualification.SILScenario,
        "to_dict",
        lambda _self: {"forged": True},
    )
    monkeypatch.setattr(
        sil_qualification.SILScenario,
        "identity_sha256",
        lambda _self: "f" * 64,
    )
    monkeypatch.setattr(
        sil_qualification.PlannedVector,
        "to_dict",
        lambda _self: {"forged": True},
    )
    monkeypatch.setattr(
        sil_qualification.UnavailableJourney,
        "to_dict",
        lambda _self: {"forged": True},
    )
    monkeypatch.setattr(
        sil_qualification.SILPlan,
        "to_dict",
        lambda _self: {"forged": True},
    )

    invalid_scenario = _scenario()
    object.__setattr__(invalid_scenario, "timeout_seconds_per_vector", 0)
    with pytest.raises(ValueError, match="timeout_seconds_per_vector"):
        build_sil_plan(registry, invalid_scenario)

    rebound_evidence, rebound_log = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=_environment_receipt(),
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )

    for field in (
        "capability_registry_identity_sha256",
        "scenario_identity_sha256",
        "input_identity_sha256",
        "output_identity_sha256",
    ):
        assert rebound_evidence[field] == baseline_evidence[field]
    assert rebound_log == baseline_log

    evidence_path = tmp_path / "stored-state-evidence.json"
    log_path = tmp_path / "stored-state.log"
    _write_evidence(evidence_path, rebound_evidence)
    log_path.write_text(rebound_log, encoding="utf-8")

    verified = _verify_evidence(evidence_path, log_path)
    assert verified["input_identity_sha256"] == baseline_evidence[
        "input_identity_sha256"
    ]


def test_sil_execution_state_validators_ignore_helper_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sil_qualification,
        "_require_git_sha",
        lambda _name, value: value,
    )
    monkeypatch.setattr(
        sil_qualification,
        "_require_sha256",
        lambda _name, value: value,
    )
    monkeypatch.setattr(
        sil_qualification,
        "_is_exact_type",
        lambda _value, _expected: True,
    )

    state = GitState(sha=_GIT_SHA, tracked_clean=True)
    object.__setattr__(state, "tracked_clean", "yes")
    with pytest.raises(ValueError, match="tracked_clean must be boolean"):
        GitState.__post_init__(state)

    execution = CommandExecution(
        return_code=0,
        stdout="ok",
        stderr="",
        duration_ms=1,
        consumed_input_identity_sha256="a" * 64,
    )
    object.__setattr__(execution, "consumed_input_identity_sha256", "forged")
    with pytest.raises(
        ValueError,
        match="consumed_input_identity_sha256",
    ):
        CommandExecution.__post_init__(execution)


def test_sil_revalidates_mutated_exact_git_probe_result() -> None:
    state = GitState(sha=_GIT_SHA, tracked_clean=True)
    object.__setattr__(state, "tracked_clean", "yes")

    with pytest.raises(ValueError, match="tracked_clean must be boolean"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=_pass_runner,
            git_probe=lambda _: state,
        )


def test_sil_revalidates_mutated_exact_command_result() -> None:
    def stale_runner(
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
        object.__setattr__(result, "return_code", True)
        return result

    with pytest.raises(ValueError, match="return_code must be an integer"):
        sil_qualification._qualify_sil_with_backends(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=_registry(),
            scenario=_scenario(),
            package_bytes=_package_bytes(),
            environment_receipt=_environment_receipt(),
            command_runner=stale_runner,
            git_probe=_git_probe,
        )

def test_public_sil_environment_validation_ignores_module_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = _registry()
    scenario = _scenario()
    canonical_environment = _environment_receipt()
    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=_package_bytes(),
        environment_receipt=canonical_environment,
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    evidence_path = tmp_path / "environment-authority-evidence.json"
    log_path = tmp_path / "environment-authority.log"
    _write_evidence(evidence_path, evidence)
    log_path.write_text(log_text, encoding="utf-8")

    monkeypatch.setattr(
        sil_qualification,
        "_validate_sil_environment_receipt",
        lambda _payload: canonical_environment,
    )
    forged_environment = {"forged": True}

    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=_package_bytes(),
            environment_receipt=forged_environment,
        )

    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        verify_sil_evidence(
            evidence_path,
            log_path,
            expected_package_bytes=_package_bytes(),
            expected_environment_receipt=forged_environment,
            expected_registry=registry,
            expected_scenario=scenario,
            expected_git_sha=_GIT_SHA,
        )


def test_parse_vector_command_ignores_pureposixpath_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForgedPath:
        def __init__(self, value: str) -> None:
            self.parts = ("tests", "forged.py")
            self.suffix = ".py"
            self._value = value

        def is_absolute(self) -> bool:
            return False

        def as_posix(self) -> str:
            return self._value

    monkeypatch.setattr(
        sil_qualification,
        "PurePosixPath",
        ForgedPath,
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="SIL integration test path must stay inside the repository",
    ):
        parse_vector_command("pytest -q tests/../forged.py")

