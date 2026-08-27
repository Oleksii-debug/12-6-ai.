from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import runtime208_vllm_learned_10m as runtime208
from tools.validate_vllm_learned_parity import collect
from twelve_six.inference.transformers_llama import llama_config_dict


def _source_run(*, status: str, conclusion: str | None) -> dict[str, object]:
    return {
        "id": runtime208.SOURCE_WORKFLOW_RUN_ID,
        "workflow_id": runtime208.SOURCE_WORKFLOW_ID,
        "name": runtime208.SOURCE_WORKFLOW_NAME,
        "head_sha": runtime208.SOURCE_HEAD_SHA,
        "status": status,
        "conclusion": conclusion,
    }


def test_exact_learned_10m_modelspec_is_maintained_llama() -> None:
    spec = runtime208.expected_model_spec()
    config = llama_config_dict(spec)

    assert spec.parameter_count() == 10_000_640
    assert spec.identity_sha256() == runtime208.EXPECTED_MODEL_SPEC_SHA256
    assert config["architectures"] == ["LlamaForCausalLM"]
    assert config["model_type"] == "llama"
    assert config["hidden_size"] == 256
    assert config["num_hidden_layers"] == 12
    assert config["num_attention_heads"] == 8
    assert config["num_key_value_heads"] == 2
    assert config["head_dim"] == 32
    assert config["intermediate_size"] == 864
    assert config["max_position_embeddings"] == 1024
    assert config["tie_word_embeddings"] is True


def test_source_gate_blocks_queued_exact_producer_without_relabeling() -> None:
    result = runtime208.validate_source_run(_source_run(status="queued", conclusion=None))

    assert result["state"] == "BLOCKED_SOURCE_NOT_TERMINAL"
    assert result["ready_for_preparation"] is False


def test_source_gate_accepts_only_exact_terminal_success() -> None:
    result = runtime208.validate_source_run(_source_run(status="completed", conclusion="success"))
    assert result["state"] == "SOURCE_TERMINAL_SUCCESS"
    assert result["ready_for_preparation"] is True

    wrong = _source_run(status="completed", conclusion="success")
    wrong["head_sha"] = "0" * 40
    with pytest.raises(runtime208.Runtime208Error, match="head_sha mismatch"):
        runtime208.validate_source_run(wrong)


def test_artifact_metadata_binds_dynamic_id_digest_to_exact_producer() -> None:
    artifact = {
        "id": 12345,
        "name": runtime208.SOURCE_ARTIFACT_NAME,
        "digest": "sha256:" + "a" * 64,
        "expired": False,
        "workflow_run": {
            "id": runtime208.SOURCE_WORKFLOW_RUN_ID,
            "head_sha": runtime208.SOURCE_HEAD_SHA,
        },
    }
    bound = runtime208.validate_artifact_metadata(artifact)
    assert bound["artifact_id"] == 12345
    assert bound["artifact_digest"] == artifact["digest"]

    artifact["workflow_run"]["head_sha"] = "f" * 40
    with pytest.raises(runtime208.Runtime208Error, match="head SHA mismatch"):
        runtime208.validate_artifact_metadata(artifact)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_retained_best_requires_fresh_pass_and_common_eval_identity(tmp_path: Path) -> None:
    checkpoint_id = "c" * 64
    target = 1_500_000
    _write_json(
        tmp_path / "fresh-verification.json",
        {
            "source_sha": runtime208.SOURCE_HEAD_SHA,
            "identity_sha256": "d" * 64,
            "fresh_verification": {"status": "PASS"},
            "ladder_common_evaluation": {
                "identity": {"identity_sha256": runtime208.EXPECTED_COMMON_EVAL_ID}
            },
            "evidence": {
                "best": {
                    "checkpoint_id": checkpoint_id,
                    "target_optimized_tokens": target,
                }
            },
        },
    )
    _write_json(
        tmp_path / "retained" / "index.json",
        {
            "source_sha": runtime208.SOURCE_HEAD_SHA,
            "roles": {
                "best": {
                    "checkpoint_id": checkpoint_id,
                    "target_optimized_tokens": target,
                    "fresh_verification": "PASS",
                }
            },
        },
    )
    (tmp_path / "retained" / "best").mkdir()

    result = runtime208.validate_retained_checkpoint(tmp_path)
    assert result["checkpoint_id"] == checkpoint_id
    assert result["target_optimized_tokens"] == target

    fresh = json.loads((tmp_path / "fresh-verification.json").read_text(encoding="utf-8"))
    fresh["ladder_common_evaluation"]["identity"]["identity_sha256"] = "e" * 64
    _write_json(tmp_path / "fresh-verification.json", fresh)
    with pytest.raises(runtime208.Runtime208Error, match="common evaluation identity mismatch"):
        runtime208.validate_retained_checkpoint(tmp_path)


def test_cpu_runtime_identity_gate_is_exact() -> None:
    identity = {
        "python": runtime208.EXPECTED_PYTHON,
        "vllm_import_version": runtime208.EXPECTED_VLLM_IMPORT_VERSION,
        "vllm_distribution_version": runtime208.EXPECTED_VLLM_CPU_DIST_VERSION,
        "torch": runtime208.EXPECTED_TORCH_CPU_VERSION,
        "transformers": runtime208.EXPECTED_TRANSFORMERS_VERSION,
        "safetensors": runtime208.EXPECTED_SAFETENSORS_VERSION,
        "cuda_available": False,
        "torch_cuda_version": None,
    }
    runtime208.validate_cpu_runtime_identity(
        identity, runtime208.EXPECTED_VLLM_CPU_WHEEL_SHA256
    )

    identity["torch"] = "different"
    with pytest.raises(runtime208.Runtime208Error, match="torch"):
        runtime208.validate_cpu_runtime_identity(
            identity, runtime208.EXPECTED_VLLM_CPU_WHEEL_SHA256
        )


def test_learned_parity_refuses_cpu_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tools.validate_vllm_learned_parity.torch.cuda.is_available", lambda: False)
    with pytest.raises(ValueError, match="requires compatible CUDA GPU"):
        collect(object())
