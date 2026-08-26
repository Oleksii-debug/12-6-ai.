from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu200_runner", ROOT / "tools" / "self_hosted_gpu_runner.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_exact_sha_requires_full_immutable_commit_identity() -> None:
    assert runner.validate_exact_sha("a" * 40) == "a" * 40
    with pytest.raises(runner.RunnerContractError, match="40 hexadecimal") as exc:
        runner.validate_exact_sha("main")
    assert exc.value.code == "INVALID_TARGET_SHA"


def test_scheduler_contract_requires_explicit_cuda_capability_labels() -> None:
    labels = ["self-hosted", "linux", "x64", "gpu", "cuda", "twelve-six-ai"]
    assert runner.validate_scheduler_labels(labels) == sorted(labels)
    with pytest.raises(runner.RunnerContractError) as exc:
        runner.validate_scheduler_labels(["self-hosted", "linux", "x64"])
    assert exc.value.code == "RUNNER_LABEL_CONTRACT"


def test_nvidia_query_parser_keeps_exact_gpu_identity() -> None:
    rows = runner.parse_nvidia_rows(
        "0, NVIDIA RTX 6000 Ada Generation, GPU-abc, 590.10, 49140, 48000\n"
    )
    assert rows == [
        {
            "physical_index": 0,
            "name": "NVIDIA RTX 6000 Ada Generation",
            "uuid": "GPU-abc",
            "driver_version": "590.10",
            "memory_total_mib": 49140,
            "memory_free_mib": 48000,
        }
    ]


def test_vram_gate_refuses_requirement_plus_reserve_above_free_memory() -> None:
    gpu = {"memory_free_mib": 2048}
    with pytest.raises(runner.RunnerContractError) as exc:
        runner.require_vram_headroom(gpu, required_gib=1.5, reserve_gib=1.0)
    assert exc.value.code == "INSUFFICIENT_VRAM_HEADROOM"


def test_vram_gate_records_remaining_headroom() -> None:
    gpu = {"memory_free_mib": 8192}
    report = runner.require_vram_headroom(gpu, required_gib=4.0, reserve_gib=1.0)
    assert report["required_mib"] == 4096
    assert report["reserve_mib"] == 1024
    assert report["headroom_after_request_mib"] == 4096


def test_durable_root_refuses_workspace_and_runner_temp(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(runner.RunnerContractError) as exc:
        runner.require_durable_root(workspace, workspace, None)
    assert exc.value.code == "DURABLE_PATH_EPHEMERAL"


def test_evidence_is_self_hashed_and_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "evidence.json"
    written = runner._write_evidence(
        path,
        {"schema": runner.HOST_SCHEMA, "phase": "host_preflight", "status": "PASS"},
    )
    assert runner._read_evidence(path, runner.HOST_SCHEMA) == written
    text = path.read_text(encoding="utf-8").replace('"status": "PASS"', '"status": "REFUSED"')
    path.write_text(text, encoding="utf-8")
    with pytest.raises(runner.RunnerContractError) as exc:
        runner._read_evidence(path, runner.HOST_SCHEMA)
    assert exc.value.code == "EVIDENCE_IDENTITY_MISMATCH"


def test_workflow_is_manual_for_gpu_and_cpu_path_is_fail_closed() -> None:
    workflow = (ROOT / ".github/workflows/gpu200-self-hosted-smoke.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "default: false" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "inputs.authorize_self_hosted_gpu == true" in workflow
    assert "runs-on: [self-hosted, linux, x64, gpu, cuda, twelve-six-ai]" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "assert p[\"reason_code\"] == \"RUNNER_LABEL_CONTRACT\"" in workflow


def test_workflow_never_embeds_or_prints_repository_secrets() -> None:
    workflow = (ROOT / ".github/workflows/gpu200-self-hosted-smoke.yml").read_text(
        encoding="utf-8"
    )
    lowered = workflow.lower()
    assert "secrets." not in lowered
    assert "github.token" not in lowered
    assert "actions_runtime_token" not in lowered
    assert "persist-credentials: false" in workflow
    assert "printenv" not in lowered
    assert "set -x" not in lowered


def test_composite_action_reuses_universal_execution_bootstrap() -> None:
    action = (ROOT / ".github/actions/self-hosted-gpu-preflight/action.yml").read_text(
        encoding="utf-8"
    )
    assert "uses: ./.github/actions/execution-bootstrap" in action
    assert "host-preflight" in action
    assert "environment-preflight" in action
    assert 'default: "runtime,cuda"' in action
