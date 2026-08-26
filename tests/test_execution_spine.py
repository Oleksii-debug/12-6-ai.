from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpu_execution_plan_excludes_cuda_payloads() -> None:
    bootstrap = _load_tool("execution_bootstrap.py")

    plan = bootstrap.resolve_plan(ROOT, ["runtime", "tests", "lint"], [])

    assert [lock["role"] for lock in plan["locks"]] == ["toolchain", "cpu_runtime", "dev"]
    assert plan["cuda_packages_present"] is False
    assert plan["package_count"] > 0


def test_cuda_execution_plan_is_explicit_and_separate() -> None:
    bootstrap = _load_tool("execution_bootstrap.py")

    plan = bootstrap.resolve_plan(ROOT, ["runtime", "cuda"], [])

    assert [lock["role"] for lock in plan["locks"]] == ["toolchain", "cuda_runtime"]
    assert plan["cuda_packages_present"] is True


def test_tokenizer_purpose_plan_does_not_pull_runtime_or_cuda() -> None:
    bootstrap = _load_tool("execution_bootstrap.py")

    plan = bootstrap.resolve_plan(ROOT, ["tokenizer"], [])

    assert [lock["role"] for lock in plan["locks"]] == [
        "toolchain",
        "tokenizer_support",
        "tokenizer_overlay",
    ]
    assert plan["cuda_packages_present"] is False


def test_command_audit_rejects_undeclared_pytest_capability() -> None:
    bootstrap = _load_tool("execution_bootstrap.py")

    with pytest.raises(bootstrap.ExecutionBootstrapError, match="undeclared capability tests"):
        bootstrap.resolve_plan(ROOT, ["runtime"], ["python -m pytest -q"])


def test_workflow_audit_accepts_central_scientific_bootstrap(tmp_path: Path) -> None:
    audit = _load_tool("audit_execution_workflows.py")
    workflow = tmp_path / "workflow.yml"
    workflow.write_text(
        """
name: test
jobs:
  check:
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97
        with:
          python-version: "3.11.16"
      - uses: ./.github/actions/execution-bootstrap
        with:
          capabilities: runtime,tests,lint
          venv: .research-env
          manifest: evidence/environment.json
      - run: .research-env/bin/python -m pytest -q tests/test_execution_spine.py
""".lstrip(),
        encoding="utf-8",
    )

    result = audit.audit_workflow(workflow)

    assert result["status"] == "PASS"
    assert result["central_dev_bootstrap"] is True
    assert result["direct_lock_install"] is False


def test_workflow_audit_rejects_ad_hoc_lock_install(tmp_path: Path) -> None:
    audit = _load_tool("audit_execution_workflows.py")
    workflow = tmp_path / "legacy.yml"
    workflow.write_text(
        """
name: legacy
jobs:
  check:
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
      - uses: actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97
        with:
          python-version: "3.11.16"
      - run: |
          python -m pip install \
            --require-hashes -r requirements/locks/linux-x86_64/dev.lock.txt
          python -m pytest -q
""".lstrip(),
        encoding="utf-8",
    )

    result = audit.audit_workflow(workflow)

    assert result["status"] == "FAIL"
    assert "direct_lock_install_deprecated" in result["findings"]
    assert "scientific_tools_without_central_dev_bootstrap" in result["findings"]
