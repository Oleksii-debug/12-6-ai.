from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_tool(name: str):
    path = ROOT / "tools" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_execution_spine_resolves_canonical_and_tokenizer_purpose() -> None:
    bootstrap = _load_tool("bootstrap_execution_spine.py")

    base = bootstrap._resolve(ROOT, "linux-x86_64")
    assert base["kind"] == "canonical"
    assert base["base_profile"]["profile_id"] == "linux-x86_64"
    assert base["overlay_paths"] == []

    tokenizer = bootstrap._resolve(ROOT, "linux-x86_64-tokenizer-experiment")
    assert tokenizer["kind"] == "purpose"
    assert tokenizer["base_profile"]["profile_id"] == "linux-x86_64"
    assert [path.name for path in tokenizer["overlay_paths"]] == ["overlay.lock.txt"]


def test_execution_spine_lock_selection_includes_dev_only_when_requested() -> None:
    bootstrap = _load_tool("bootstrap_execution_spine.py")
    resolved = bootstrap._resolve(ROOT, "linux-x86_64")

    runtime = bootstrap._lock_paths(ROOT, resolved["base_profile"], False)
    dev = bootstrap._lock_paths(ROOT, resolved["base_profile"], True)

    assert [path.name for path in runtime] == ["toolchain.lock.txt", "runtime.lock.txt"]
    assert [path.name for path in dev] == [
        "toolchain.lock.txt",
        "runtime.lock.txt",
        "dev.lock.txt",
    ]


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
      - run: python tools/bootstrap_execution_spine.py --purpose linux-x86_64 --with-dev
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
