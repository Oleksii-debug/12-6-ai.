from __future__ import annotations

import json
from pathlib import Path

from tools.ci153_workflow_dependency_auditor import audit_repository


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    _write(
        repo / "requirements/profiles/index.json",
        json.dumps(
            {
                "profiles": {
                    "linux-x86_64-cuda-training": {
                        "path": "requirements/profiles/linux-x86_64-cuda-training/profile.json"
                    }
                }
            }
        ),
    )
    _write(
        repo / "requirements/profiles/linux-x86_64-cuda-training/profile.json",
        json.dumps(
            {
                "profile_id": "linux-x86_64-cuda-training",
                "base_profile": {"path": "requirements/locks/linux-x86_64/profile.json"},
            }
        ),
    )
    _write(
        repo / "requirements/locks/linux-x86_64/profile.json",
        json.dumps(
            {
                "locks": {
                    "runtime": {"path": "requirements/locks/linux-x86_64/runtime.lock.txt"},
                    "dev": {"path": "requirements/locks/linux-x86_64/dev.lock.txt"},
                }
            }
        ),
    )
    _write(repo / "requirements/locks/linux-x86_64/runtime.lock.txt", "torch==2.13.0\n")
    _write(repo / "requirements/locks/linux-x86_64/dev.lock.txt", "pytest==9.1.1\nruff==0.16.4\n")
    return repo


def test_historical_pytest_without_dev_lock_is_missing_dependency(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(
        repo / ".github/workflows/broken.yml",
        """name: broken
jobs:
  test:
    steps:
      - run: |
          python -m pip install -r requirements/locks/linux-x86_64/runtime.lock.txt
          python -m pytest -q tests
""",
    )
    report = audit_repository(repo)
    row = report["workflows"][0]
    assert row["classification"] == "MISSING_DECLARED_DEPENDENCY"
    assert row["missing_packages"] == ["pytest"]


def test_dev_lock_proves_pytest_and_ruff(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(
        repo / ".github/workflows/valid.yml",
        """name: valid
jobs:
  test:
    steps:
      - run: |
          python -m pip install -r requirements/locks/linux-x86_64/dev.lock.txt
          python -m pytest -q tests
          python -m ruff check tools tests
""",
    )
    report = audit_repository(repo)
    row = report["workflows"][0]
    assert row["classification"] == "VALID"
    assert row["missing_packages"] == []


def test_stale_profile_reference_fails_closed(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(
        repo / ".github/workflows/stale.yml",
        """name: stale
jobs:
  test:
    steps:
      - run: python tools/bootstrap.py --profile requirements/profiles/does-not-exist/profile.json
""",
    )
    report = audit_repository(repo)
    row = report["workflows"][0]
    assert row["classification"] == "STALE_PROFILE_REFERENCE"
    assert row["declarations"]["stale_profile_references"] == ["does-not-exist"]


def test_dynamic_module_command_is_ambiguous(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(
        repo / ".github/workflows/dynamic.yml",
        """name: dynamic
jobs:
  test:
    steps:
      - run: $TOOL -m pytest
""",
    )
    report = audit_repository(repo)
    row = report["workflows"][0]
    assert row["classification"] == "AMBIGUOUS_DYNAMIC_COMMAND"


def test_inventory_scans_every_active_yaml_workflow(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _write(repo / ".github/workflows/a.yml", "name: a\njobs: {}\n")
    _write(repo / ".github/workflows/b.yaml", "name: b\njobs: {}\n")
    report = audit_repository(repo)
    assert report["inventory_count"] == 2
    assert {row["path"] for row in report["workflows"]} == {
        ".github/workflows/a.yml",
        ".github/workflows/b.yaml",
    }
