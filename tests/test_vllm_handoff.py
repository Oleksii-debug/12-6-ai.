from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.inference.vllm_handoff import (
    EXPORT_ATTESTATION_NAME,
    PARITY_REQUEST_NAME,
    VllmHandoffError,
    bind_parity_report,
    inspect_vllm_handoff,
    main,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _make_export(
    root: Path,
    *,
    checkpoint_id: str = "ckpt-test",
    transformers_status: str = "VERIFIED",
    runtime_status: str = "VERIFIED",
    model_type: str = "twelve_six",
    architecture: str = "TwelveSixForCausalLM",
) -> Path:
    export = root / "export"
    export.mkdir()
    weights = export / "model.safetensors"
    weights.write_bytes(b"safe-test-weights")
    config = export / "config.json"
    _write_json(
        config,
        {
            "architectures": [architecture],
            "model_type": model_type,
        },
    )
    source_manifest = export / "12-6-checkpoint-manifest.json"
    _write_json(source_manifest, {"checkpoint_id": checkpoint_id})

    weights_sha = _sha256(weights)
    config_sha = _sha256(config)
    _write_json(
        export / EXPORT_ATTESTATION_NAME,
        {
            "schema": "12-6.hf-style-export.v1",
            "checkpoint_id": checkpoint_id,
            "source_manifest_sha256": _sha256(source_manifest),
            "model_safetensors_sha256": weights_sha,
            "config_sha256": config_sha,
            "compatibility": {
                "layout": "HF_STYLE_SAFETENSORS_DIRECTORY",
                "weights": "EXACT_CANONICAL_BYTE_COPY",
                "transformers_architecture": transformers_status,
                "runtime_logit_generation_parity": runtime_status,
            },
        },
    )
    _write_json(
        export / PARITY_REQUEST_NAME,
        {
            "schema": "12-6.export-parity-request.v1",
            "status": "NOT_TESTED",
            "checkpoint_id": checkpoint_id,
            "reference_weights_sha256": weights_sha,
            "candidate_weights_sha256": weights_sha,
            "candidate_config_sha256": config_sha,
            "required_checks": [
                "prompt_token_identity",
                "next_token_logit_parity",
                "greedy_generation_parity",
            ],
            "authority": "D07_or_independent_parity_harness",
            "hook_result": None,
        },
    )
    return export


def _make_parity_report(root: Path, *, passed: bool = True) -> Path:
    report = root / "parity.json"
    _write_json(
        report,
        {
            "schema": "12-6.inference-parity.v1",
            "passed": passed,
            "prompts_compared": 2,
            "steps_compared": 5,
            "max_new_tokens": 3,
            "atol": 0.0,
            "rtol": 0.0,
            "max_abs_error": 0.0,
            "max_rel_error": 0.0,
            "failures": [] if passed else [{"kind": "logit_mismatch"}],
        },
    )
    return report


def test_bind_parity_then_preflight_ready(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    parity = _make_parity_report(tmp_path)
    binding = tmp_path / "binding.json"

    bind_parity_report(export, parity, binding)
    result = inspect_vllm_handoff(export, binding)

    assert result.ready_for_plugin_implementation is True
    assert result.checkpoint_id == "ckpt-test"
    assert result.vllm_runtime_status == "NOT_TESTED"
    assert result.blockers == ()
    assert result.parity_report_sha256 == _sha256(parity)


def test_conservative_current_export_status_remains_blocked(tmp_path: Path) -> None:
    export = _make_export(
        tmp_path,
        transformers_status="NOT_CLAIMED",
        runtime_status="NOT_TESTED",
    )
    parity = _make_parity_report(tmp_path)
    binding = tmp_path / "binding.json"
    bind_parity_report(export, parity, binding)

    result = inspect_vllm_handoff(export, binding)

    assert result.ready_for_plugin_implementation is False
    assert any("Transformers architecture compatibility" in item for item in result.blockers)
    assert any("runtime parity" in item for item in result.blockers)
    assert result.vllm_runtime_status == "NOT_TESTED"


def test_preflight_requires_artifact_bound_parity(tmp_path: Path) -> None:
    export = _make_export(tmp_path)

    result = inspect_vllm_handoff(export)

    assert result.ready_for_plugin_implementation is False
    assert "artifact-bound D07 parity evidence is missing" in result.blockers


def test_tampered_export_weights_fail_closed(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    (export / "model.safetensors").write_bytes(b"tampered")

    result = inspect_vllm_handoff(export)

    assert result.ready_for_plugin_implementation is False
    assert result.blockers[0].startswith("export_integrity:")
    assert "model.safetensors hash" in result.blockers[0]


def test_binding_cannot_be_reused_for_different_export(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first_export = _make_export(first_root, checkpoint_id="ckpt-first")
    first_parity = _make_parity_report(first_root)
    binding = first_root / "binding.json"
    bind_parity_report(first_export, first_parity, binding)

    second_root = tmp_path / "second"
    second_root.mkdir()
    second_export = _make_export(second_root, checkpoint_id="ckpt-second")
    result = inspect_vllm_handoff(second_export, binding)

    assert result.ready_for_plugin_implementation is False
    assert any("checkpoint_id does not match export" in item for item in result.blockers)


def test_failed_parity_report_cannot_be_bound(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    parity = _make_parity_report(tmp_path, passed=False)

    with pytest.raises(VllmHandoffError, match="did not pass"):
        bind_parity_report(export, parity, tmp_path / "binding.json")


def test_wrong_transformers_identity_blocks_plugin_handoff(tmp_path: Path) -> None:
    export = _make_export(
        tmp_path,
        model_type="gpt2",
        architecture="GPT2LMHeadModel",
    )
    parity = _make_parity_report(tmp_path)
    binding = tmp_path / "binding.json"
    bind_parity_report(export, parity, binding)

    result = inspect_vllm_handoff(export, binding)

    assert result.ready_for_plugin_implementation is False
    assert any("config.model_type" in item for item in result.blockers)
    assert any("config.architectures" in item for item in result.blockers)


def test_duplicate_json_key_fails_closed(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    (export / PARITY_REQUEST_NAME).write_text(
        '{"schema":"12-6.export-parity-request.v1","schema":"duplicate"}',
        encoding="utf-8",
    )

    result = inspect_vllm_handoff(export)

    assert result.ready_for_plugin_implementation is False
    assert "duplicate JSON key" in result.blockers[0]


def test_tampered_embedded_parity_report_is_detected(tmp_path: Path) -> None:
    export = _make_export(tmp_path)
    parity = _make_parity_report(tmp_path)
    binding_path = tmp_path / "binding.json"
    bind_parity_report(export, parity, binding_path)

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["parity_report"]["steps_compared"] = 999
    _write_json(binding_path, binding)

    result = inspect_vllm_handoff(export, binding_path)

    assert result.ready_for_plugin_implementation is False
    assert "embedded parity report canonical hash mismatch" in result.blockers


def test_cli_json_reports_blocked_with_exit_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    export = _make_export(
        tmp_path,
        transformers_status="NOT_CLAIMED",
        runtime_status="NOT_TESTED",
    )

    return_code = main(["preflight", "--export-dir", str(export), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 2
    assert payload["schema"] == "12-6.vllm-handoff-preflight.v1"
    assert payload["ready_for_plugin_implementation"] is False
    assert payload["vllm_runtime_status"] == "NOT_TESTED"
