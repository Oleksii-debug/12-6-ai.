from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint
from twelve_six.inference.contracts import GenerationConfig
from twelve_six.model import ModelSpec, StageConfig, TwelveSixDecoder, load_stage_config
from twelve_six.postbase import (
    ControllerGenerationPort,
    ControllerGenerationRequest,
    PostBaseCompatibilityError,
    PostBaseModelAdapter,
    validate_postbase_compatible_spec,
)
from twelve_six.tokenization import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def _identity(
    stage: StageConfig,
    tokenizer: ByteTokenizer,
    *,
    step: int = 3,
    tokens_seen: int = 96,
) -> CheckpointIdentity:
    return CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=stage.model.to_dict(),
        parameter_count=stage.expected_parameters,
        tokenizer_hash=tokenizer.identity.config_sha256,
        tokenizer_vocab_hash=tokenizer.identity.vocab_sha256,
        dataset_manifest_hash="b" * 64,
        run_manifest_hash="c" * 64,
        training_config={
            "data": {"tokenizer_version": tokenizer.identity.version},
            "training": {"context_length": stage.model.max_seq_len},
        },
        seed=17,
        precision="fp32",
        step=step,
        tokens_seen=tokens_seen,
        optimizer={"name": "adamw"},
        scheduler=None,
        environment_lock_hash="d" * 64,
    )


def _checkpoint(
    tmp_path: Path,
    *,
    step: int = 3,
    tokens_seen: int = 96,
) -> tuple[Path, StageConfig]:
    stage = load_stage_config(ROOT / "configs/stages/s0_10k.json")
    tokenizer = ByteTokenizer()
    checkpoint = tmp_path / "base-checkpoint"
    model = TwelveSixDecoder(stage.model, stage.init)
    save_checkpoint(
        checkpoint,
        model=model,
        identity=_identity(stage, tokenizer, step=step, tokens_seen=tokens_seen),
    )
    return checkpoint, stage


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_adapter_consumes_verified_checkpoint_without_modifying_it(tmp_path: Path) -> None:
    checkpoint, stage = _checkpoint(tmp_path)
    before = _snapshot_files(checkpoint)

    adapter = PostBaseModelAdapter.from_checkpoint(
        checkpoint,
        expected_model_spec_sha256=stage.model.identity_sha256(),
    )
    assert isinstance(adapter, ControllerGenerationPort)

    deliberation = adapter.generate(
        ControllerGenerationRequest(
            controller="deliberation",
            prompt="local controller probe",
            config=GenerationConfig(max_new_tokens=2),
        )
    )
    tool = adapter.generate(
        ControllerGenerationRequest(
            controller="tool",
            prompt="tool planning probe",
            config=GenerationConfig(max_new_tokens=2),
        )
    )

    after = _snapshot_files(checkpoint)
    assert after == before
    assert deliberation.base_evidence is adapter.base_evidence
    assert tool.base_evidence is adapter.base_evidence
    assert deliberation.post_base_evidence.controller == "deliberation"
    assert tool.post_base_evidence.controller == "tool"
    assert deliberation.post_base_evidence.runtime_policy == "LOCAL_FREE"


def test_base_and_post_base_evidence_namespaces_do_not_mix(tmp_path: Path) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    adapter = PostBaseModelAdapter.from_checkpoint(checkpoint)
    response = adapter.generate(
        ControllerGenerationRequest(
            controller="deliberation",
            prompt="evidence firewall",
            config=GenerationConfig(max_new_tokens=1),
        )
    )

    base = asdict(response.base_evidence)
    post_base = asdict(response.post_base_evidence)

    assert base["evidence_namespace"] == "base"
    assert post_base["evidence_namespace"] == "post_base"
    assert set(base).intersection(post_base) == {"evidence_namespace"}
    assert "model_spec_sha256" in base
    assert "dataset_manifest_sha256" in base
    assert "tokens_seen" in base
    assert "prompt_utf8_sha256" not in base
    assert "generated_token_count" not in base
    assert "model_spec_sha256" not in post_base
    assert "dataset_manifest_sha256" not in post_base
    assert "tokens_seen" not in post_base


def test_adapter_rejects_non_learned_checkpoint_counters(tmp_path: Path) -> None:
    checkpoint, _ = _checkpoint(tmp_path, step=0, tokens_seen=0)
    with pytest.raises(PostBaseCompatibilityError, match="learned Base checkpoints only"):
        PostBaseModelAdapter.from_checkpoint(checkpoint)


def test_adapter_rejects_external_network_or_wrapper_backend() -> None:
    class ExternalNetworkBackend:
        pass

    with pytest.raises(PostBaseCompatibilityError, match="external, network, wrapper"):
        PostBaseModelAdapter(ExternalNetworkBackend())  # type: ignore[arg-type]


def test_adapter_public_surface_has_no_training_or_checkpoint_writer_entrypoint(
    tmp_path: Path,
) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    adapter = PostBaseModelAdapter.from_checkpoint(checkpoint)

    forbidden = {
        "backward",
        "optimizer",
        "optimizer_step",
        "save_checkpoint",
        "train",
        "zero_grad",
    }
    assert forbidden.isdisjoint(name for name in dir(adapter) if not name.startswith("_"))


def test_current_learned_10m_modelspec_is_supported_without_stage_allowlist() -> None:
    spec = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=1024,
        d_model=256,
        n_layers=12,
        n_heads=8,
        n_kv_heads=2,
        head_dim=32,
        d_ff=864,
        rope_rotary_dim=32,
    )
    assert spec.parameter_count() == 10_000_640
    assert spec.identity_sha256() == (
        "61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998"
    )
    validate_postbase_compatible_spec(spec)


def test_primary_model341_20m_modelspec_is_supported_without_size_gate() -> None:
    # MODEL-341 exact head e4ff486fd90802fc123bebf60eed4e59196a98df.
    primary_20m = ModelSpec(
        schema_version=1,
        vocab_size=256,
        max_seq_len=1024,
        d_model=320,
        n_layers=16,
        n_heads=10,
        n_kv_heads=2,
        head_dim=32,
        d_ff=1080,
        rope_rotary_dim=32,
    )
    assert primary_20m.parameter_count() == 20_613_440
    assert primary_20m.identity_sha256() == (
        "fbff24d561a2818453554d58ca23fc6ace3303b078f1935a8576c4565bd92441"
    )
    validate_postbase_compatible_spec(primary_20m)


def test_expected_modelspec_binding_fails_closed(tmp_path: Path) -> None:
    checkpoint, _ = _checkpoint(tmp_path)
    with pytest.raises(PostBaseCompatibilityError, match="controller expectation"):
        PostBaseModelAdapter.from_checkpoint(
            checkpoint,
            expected_model_spec_sha256="0" * 64,
        )
