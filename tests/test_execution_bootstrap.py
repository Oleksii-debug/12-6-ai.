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
