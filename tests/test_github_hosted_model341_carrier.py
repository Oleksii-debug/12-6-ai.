from __future__ import annotations

from pathlib import Path

import pytest

from tools import verify_github_hosted_model341_carrier as carrier


def _source_sha() -> str:
    return carrier._observed_checkout_sha(carrier.ROOT)


def _different_sha(value: str) -> str:
    replacement = "0" if value[0] != "0" else "1"
    return replacement + value[1:]


def _hosted_env() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_REPOSITORY": carrier.EXPECTED_REPOSITORY,
        "GITHUB_REPOSITORY_VISIBILITY": "public",
        "RUNNER_ENVIRONMENT": carrier.EXPECTED_RUNNER_ENVIRONMENT,
        "RUNNER_OS": "Linux",
        "RUNNER_ARCH": "X64",
    }


def test_preflight_binds_exact_worker_checkout_and_zero_effect_boundary() -> None:
    source_sha = _source_sha()
    evidence = carrier.build_preflight_evidence(
        source_sha=source_sha,
        mode="preflight",
        env=_hosted_env(),
    )

    assert evidence["result"] == "PASS"
    assert evidence["source_sha"] == source_sha
    assert evidence["expected_source_sha"] == source_sha
    assert evidence["observed_checkout_sha"] == source_sha
    assert evidence["runner"] == {
        "provider": "github-hosted",
        "environment": "github-hosted",
        "label": "ubuntu-24.04",
        "os": "Linux",
        "arch": "X64",
    }
    assert evidence["worker_identity"]["MODE"] == "synthetic-mechanics"
    assert evidence["worker_identity"]["EXPECTED_PARAMETERS"] == 20_613_440
    assert evidence["scientific_effects"] == {
        "worker_invoked": False,
        "real_target_execution_supported": False,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
    }
    assert len(evidence["evidence_sha256"]) == 64


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GITHUB_ACTIONS", "false"),
        ("GITHUB_REPOSITORY", "example/not-canonical"),
        ("GITHUB_REPOSITORY_VISIBILITY", "private"),
        ("RUNNER_ENVIRONMENT", "self-hosted"),
        ("RUNNER_OS", "Windows"),
        ("RUNNER_ARCH", "ARM64"),
    ],
)
def test_preflight_rejects_wrong_execution_authority(name: str, value: str) -> None:
    env = _hosted_env()
    env[name] = value

    with pytest.raises(carrier.HostedCarrierPreflightError, match=name):
        carrier.build_preflight_evidence(
            source_sha=_source_sha(),
            mode="preflight",
            env=env,
        )


def test_preflight_rejects_source_sha_not_matching_checkout() -> None:
    source_sha = _source_sha()

    with pytest.raises(
        carrier.HostedCarrierPreflightError,
        match="source_sha does not match checked-out Git HEAD",
    ):
        carrier.build_preflight_evidence(
            source_sha=_different_sha(source_sha),
            mode="preflight",
            env=_hosted_env(),
        )


def test_checkout_sha_authentication_fails_outside_git_worktree(tmp_path: Path) -> None:
    with pytest.raises(
        carrier.HostedCarrierPreflightError,
        match="unable to resolve checked-out Git HEAD",
    ):
        carrier._observed_checkout_sha(tmp_path)


@pytest.mark.parametrize("name", carrier.FORBIDDEN_WORKER_AUTHORITY_ENV)
def test_preflight_rejects_worker_authority_environment(name: str) -> None:
    env = _hosted_env()
    env[name] = "present"

    with pytest.raises(carrier.HostedCarrierPreflightError, match=name):
        carrier.build_preflight_evidence(
            source_sha=_source_sha(),
            mode="preflight",
            env=env,
        )


def test_real_target_or_training_modes_do_not_exist() -> None:
    for mode in ("real-target", "train", "synthetic-mechanics"):
        with pytest.raises(carrier.HostedCarrierPreflightError, match="unsupported carrier mode"):
            carrier.build_preflight_evidence(
                source_sha=_source_sha(),
                mode=mode,
                env=_hosted_env(),
            )


def test_preflight_rejects_external_worker_identity_drift(tmp_path: Path) -> None:
    worker = tmp_path / "src" / "twelve_six" / "training" / "external_worker.py"
    launcher = tmp_path / "tools" / "run_external_training_worker.py"
    worker.parent.mkdir(parents=True)
    launcher.parent.mkdir(parents=True)
    worker.write_text("PROTOCOL_VERSION = 999\n", encoding="utf-8")
    launcher.write_text("pass\n", encoding="utf-8")

    with pytest.raises(carrier.HostedCarrierPreflightError, match="Git blob identity drifted"):
        carrier.build_preflight_evidence(
            source_sha=_source_sha(),
            mode="preflight",
            env=_hosted_env(),
            root=tmp_path,
        )


def test_shared_ci_carrier_has_no_worker_or_real_target_invocation() -> None:
    workflow = (carrier.ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "learned20m_hosted_carrier_mode:" in workflow
    assert "options:\n          - preflight" in workflow
    assert "  learned20m-hosted-carrier:" in workflow
    carrier_job = workflow.split("  learned20m-hosted-carrier:", 1)[1]
    assert "runs-on: ubuntu-24.04" in carrier_job
    assert "tools/verify_github_hosted_model341_carrier.py" in carrier_job
    assert "run_external_training_worker.py" not in carrier_job
    assert "TWELVE_SIX_EXTERNAL_TRAINING_ROOT" not in carrier_job
    assert "TWELVE_SIX_TRAINER_AUTHORITY_SHA256" not in carrier_job
    assert "self-hosted" not in carrier_job
