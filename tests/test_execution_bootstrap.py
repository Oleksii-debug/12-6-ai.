from __future__ import annotations

import subprocess
import venv
from pathlib import Path

import pytest

import tools.execution_bootstrap as eb

ROOT = Path(__file__).resolve().parents[1]


def test_pytest_command_requires_tests_capability() -> None:
    with pytest.raises(eb.ExecutionBootstrapError, match="undeclared capability tests"):
        eb.resolve_plan(ROOT, ["runtime"], ["python -m pytest -q tests"])


def test_ruff_command_requires_lint_capability() -> None:
    with pytest.raises(eb.ExecutionBootstrapError, match="undeclared capability lint"):
        eb.resolve_plan(ROOT, ["runtime"], ["ruff check src"])


@pytest.mark.parametrize("capability", ["datatrove", "pyarrow", "vllm"])
def test_unlocked_optional_capabilities_fail_before_install(capability: str) -> None:
    with pytest.raises(eb.ExecutionBootstrapError, match="unavailable_no_exact_lock"):
        eb.resolve_plan(ROOT, [capability], [])


def test_cpu_training_test_plan_has_dev_but_no_cuda_packages() -> None:
    plan = eb.resolve_plan(ROOT, ["runtime", "tests"], ["python -m pytest -q tests"])
    assert [r["role"] for r in plan["locks"]] == ["toolchain", "cpu_runtime", "dev"]
    assert plan["cuda_packages_present"] is False


def test_tokenizer_plan_is_purpose_specific_and_non_cuda() -> None:
    plan = eb.resolve_plan(ROOT, ["tokenizer"], [])
    assert [r["role"] for r in plan["locks"]] == ["toolchain", "tokenizer_support", "tokenizer_overlay"]
    assert plan["cuda_packages_present"] is False


def test_cuda_profile_uses_declared_cuda_runtime() -> None:
    plan = eb.resolve_plan(ROOT, ["runtime", "cuda"], [])
    assert [r["role"] for r in plan["locks"]] == ["toolchain", "cuda_runtime"]
    assert plan["cuda_packages_present"] is True


def _empty_venv(tmp_path: Path) -> Path:
    target = tmp_path / "empty"
    venv.EnvBuilder(with_pip=False).create(target)
    return eb._venv_python(target)


@pytest.mark.parametrize("module", ["pytest", "ruff", "tokenizers"])
def test_missing_import_detected_before_experiment(tmp_path: Path, module: str) -> None:
    python = _empty_venv(tmp_path / module)
    with pytest.raises(subprocess.CalledProcessError):
        eb._probe_imports(python, [module], ROOT)


@pytest.mark.parametrize("executable", ["pytest", "ruff"])
def test_missing_executable_detected_in_target_venv(tmp_path: Path, executable: str) -> None:
    python = _empty_venv(tmp_path / executable)
    with pytest.raises(eb.ExecutionBootstrapError, match="missing declared executables"):
        eb._probe_executables(python, [executable])


def test_cpu_runtime_executes_tiny_training_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    import torch
    import torch.nn.functional as F

    from twelve_six import ModelSpec, TwelveSixDecoder
    from twelve_six.checkpoint import CheckpointIdentity, load_checkpoint, save_checkpoint

    torch.manual_seed(151)
    spec = ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        d_ff=32,
        rope_rotary_dim=8,
    )
    model = TwelveSixDecoder(spec)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    logits = model(input_ids).logits
    assert logits.shape == (1, 5, spec.vocab_size)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, spec.vocab_size),
        input_ids[:, 1:].reshape(-1),
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    optimizer.step()
    assert any(
        not torch.equal(before[name], value)
        for name, value in model.state_dict().items()
    )

    identity = CheckpointIdentity(
        git_sha="a" * 40,
        model_spec=spec.to_dict(),
        parameter_count=spec.parameter_count(),
        tokenizer_hash="b" * 64,
        tokenizer_vocab_hash="c" * 64,
        dataset_manifest_hash="d" * 64,
        run_manifest_hash="e" * 64,
        training_config={"purpose": "env151-tiny-cpu-smoke"},
        seed=151,
        precision="fp32",
        step=1,
        tokens_seen=input_ids.numel(),
        optimizer={"name": "SGD", "lr": 0.05},
        scheduler=None,
        environment_lock_hash="f" * 64,
    )
    checkpoint = tmp_path / "checkpoint"
    manifest = save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        trainer_state={"loss": float(loss.detach())},
        identity=identity,
    )

    restored = TwelveSixDecoder(spec)
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.05)
    result = load_checkpoint(
        checkpoint,
        model=restored,
        optimizer=restored_optimizer,
        restore_rng=False,
        expected_model_spec_hash=manifest["identity"]["model_spec_hash"],
        expected_tokenizer_vocab_hash=identity.tokenizer_vocab_hash,
        expected_run_manifest_hash=identity.run_manifest_hash,
    )
    for name, expected in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    assert result.trainer_state["loss"] == pytest.approx(float(loss.detach()))
