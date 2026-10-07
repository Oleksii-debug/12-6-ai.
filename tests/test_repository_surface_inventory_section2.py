from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.validate_section2_repository_surface_coverage as surface_validator
from tools.validate_section2_repository_surface_coverage import (
    _candidate_surface_paths,
    _load_strict_json,
    validate_repository_surface_coverage,
)


_ROOT = Path(__file__).parents[1]
_INVENTORY = (
    _ROOT
    / "configs"
    / "control"
    / "product_repository_executable_surface_rules_v1.json"
)
_CAPABILITIES = _ROOT / "configs" / "control" / "product_capabilities_v1.json"


def _validate(
    inventory: Path = _INVENTORY,
    capabilities: Path = _CAPABILITIES,
) -> dict[str, object]:
    return validate_repository_surface_coverage(
        repo_root=_ROOT,
        inventory_path=inventory,
        capability_registry_path=capabilities,
    )


def test_repository_executable_surface_coverage_is_exact_and_complete() -> None:
    result = _validate()

    assert result["observed_main_sha"] == "019944d5fe12334791f05f1232d13de4a12e37d3"
    assert result["observed_main_tree_sha"] == "c727add7897dd94bdb02493e0cd7a565be7e8d9f"
    assert result["current_repository_main_sha"] == "e5dbb7107d5b54f09a26d07d59f093ac05ede9c7"
    assert result["current_repository_main_tree_sha"] == "429a9933512f3d0c17f42d80193365e5df3f195a"
    assert result["qualified_current_equivalent_surface_count"] == 233
    assert result["accepted_main_surface_count"] == 119
    assert result["candidate_overlay_surface_count"] == 1
    assert result["checkout_surface_count"] == 120


def test_repository_executable_surface_distribution_is_pinned() -> None:
    result = _validate()

    assert result["main_capability_counts"] == {
        "accelerated-scaling-research": 2,
        "byte-tokenizer-runtime": 1,
        "checkpoint-integrity-mechanics": 1,
        "data-governance-mechanics": 102,
        "deterministic-packing-mechanics": 2,
        "learned20m-control-plane": 5,
        "model-spec-identity": 2,
        "package-runtime": 1,
        "portable-run-authority": 2,
        "project-control-plane": 1,
    }


def test_section2_validator_is_itself_an_explicit_candidate_overlay() -> None:
    payload = _load_strict_json(_INVENTORY)

    assert payload["candidate_overrides"] == [
        {
            "path": "tools/validate_section2_repository_surface_coverage.py",
            "capability_id": "executable-capability-map",
        }
    ]


def test_candidate_surface_paths_include_changed_existing_surface() -> None:
    current_main_blobs = {
        ".github/workflows/ci.yml": "a" * 40,
        "tools/existing.py": "b" * 40,
    }
    checkout_blobs = {
        ".github/workflows/ci.yml": "c" * 40,
        "tools/existing.py": "b" * 40,
        "tools/new.py": "d" * 40,
    }

    assert _candidate_surface_paths(current_main_blobs, checkout_blobs) == {
        ".github/workflows/ci.yml",
        "tools/new.py",
    }


def test_repository_surface_coverage_rejects_available_candidate_override(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["candidate_overrides"][0]["capability_id"] = "model-spec-identity"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain UNAVAILABLE"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_missing_candidate_override(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["candidate_overrides"] = []
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unmapped_checkout"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_unknown_rule_capability(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["rules"][0]["capability_id"] = "forged-capability"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown classified capability"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_count_reseal(tmp_path: Path) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["expected_main_surface_count"] -= 1
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="do not sum"):
        _validate(inventory)


def test_repository_surface_inventory_rejects_duplicate_json_members(
    tmp_path: Path,
) -> None:
    text = _INVENTORY.read_text(encoding="utf-8")
    tampered = text.replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    inventory = tmp_path / "inventory.json"
    inventory.write_text(tampered, encoding="utf-8")

    with pytest.raises(ValueError, match="strict unambiguous"):
        _load_strict_json(inventory)


def test_repository_surface_rules_are_nonambiguous_on_exact_main() -> None:
    # Full validation classifies every accepted-main surface through exactly one rule.
    _validate()


def test_repository_surface_coverage_rejects_resealed_current_main(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = payload["observed_main_sha"]
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match the live repository main ref"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_current_main_tree_reseal(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match current_repository_main_sha"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_duplicate_capability_ids(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["capabilities"].append(dict(payload["capabilities"][0]))
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="capability ids must be unique"):
        _validate(capabilities=capabilities)


def test_repository_surface_coverage_rejects_capability_registry_baseline_reseal(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["observed_main_sha"] = "e5dbb7107d5b54f09a26d07d59f093ac05ede9c7"
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match coverage baseline"):
        _validate(capabilities=capabilities)


def test_repository_surface_coverage_rejects_nonterminal_capability_ci(
    tmp_path: Path,
) -> None:
    payload = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    payload["observed_main_ci"]["conclusion"] = "failure"
    capabilities = tmp_path / "capabilities.json"
    capabilities.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must be terminal success"):
        _validate(capabilities=capabilities)


def test_git_probes_strip_ambient_git_redirection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hostile = {
        "Git_Dir": str(tmp_path / "forged.git"),
        "git_work_tree": str(tmp_path / "forged-worktree"),
        "gIt_CoNfIg_PaRaMeTeRs": "core.worktree=/forged",
    }
    for key, value in hostile.items():
        monkeypatch.setenv(key, value)
    observed_envs: list[dict[str, str]] = []
    sha = "a" * 40

    def fake_run(
        command: list[str],
        *,
        check: bool,
        capture_output: bool,
        text: bool,
        env: dict[str, str],
    ) -> object:
        assert check is False
        assert capture_output is True
        assert text is True
        observed_envs.append(dict(env))
        stdout = f"{sha}\\n" if "rev-parse" in command else ""
        if command[-1] == "HEAD":
            stdout = "ok\\n"
        return surface_validator.subprocess.CompletedProcess(
            command,
            0,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(surface_validator.subprocess, "run", fake_run)

    assert surface_validator._run_git(tmp_path, "show", "HEAD") == ["ok"]
    assert surface_validator._run_git_z(tmp_path, "ls-files", "-z") == []
    assert surface_validator._resolve_live_main_sha(tmp_path) == sha
    assert len(observed_envs) == 3
    for env in observed_envs:
        assert all(
            not key.upper().startswith("GIT_") or key == "GIT_OPTIONAL_LOCKS"
            for key in env
        )
        assert env["GIT_OPTIONAL_LOCKS"] == "0"


def test_worktree_capability_surface_drift_detects_tracked_and_untracked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run_git_z(_root: Path, *args: str) -> list[str]:
        if args == ("diff", "--name-only", "-z", "HEAD", "--"):
            return ["src/twelve_six/model.py", "README.md"]
        if args == ("ls-files", "-z", "--others", "--exclude-standard"):
            return ["tools/forged.py", "notes.txt"]
        raise AssertionError(args)

    monkeypatch.setattr(surface_validator, "_run_git_z", fake_run_git_z)

    tracked, untracked = surface_validator._worktree_capability_surface_drift(tmp_path)

    assert tracked == {"src/twelve_six/model.py"}
    assert untracked == {"tools/forged.py"}


def test_clean_capability_worktree_guard_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        surface_validator,
        "_worktree_capability_surface_drift",
        lambda _root: ({"pyproject.toml"}, {"tools/forged.sh"}),
    )

    with pytest.raises(ValueError, match="working tree capability surface drift"):
        surface_validator._require_clean_capability_worktree(tmp_path)


def test_surface_blob_map_identity_includes_regular_file_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    blob_sha = "a" * 40

    def fake_run_git(_root: Path, *args: str) -> list[str]:
        treeish = args[2]
        mode = "100644" if treeish == "main-tree" else "100755"
        return [f"{mode} blob {blob_sha}\ttools/mode_sensitive.py"]

    monkeypatch.setattr(surface_validator, "_run_git_z", fake_run_git)
    main_blobs = surface_validator._surface_blob_map(tmp_path, "main-tree")
    checkout_blobs = surface_validator._surface_blob_map(tmp_path, "checkout-tree")

    assert main_blobs["tools/mode_sensitive.py"] != checkout_blobs[
        "tools/mode_sensitive.py"
    ]
    assert _candidate_surface_paths(main_blobs, checkout_blobs) == {
        "tools/mode_sensitive.py"
    }


def test_surface_blob_map_preserves_unquoted_unicode_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = "tools/перевірка.py"
    line = "100644 blob " + ("a" * 40) + f"\t{path}"
    calls: list[tuple[str, ...]] = []

    def fake_run_git_z(_root: Path, *args: str) -> list[str]:
        calls.append(args)
        return [line]

    monkeypatch.setattr(surface_validator, "_run_git_z", fake_run_git_z)

    blobs = surface_validator._surface_blob_map(tmp_path, "HEAD")

    assert blobs[path] == "100644:" + ("a" * 40)
    assert calls == [("ls-tree", "-rz", "HEAD")]


def test_surface_blob_map_rejects_symlink_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    line = "120000 blob " + ("a" * 40) + "\ttools/forged.py"
    monkeypatch.setattr(surface_validator, "_run_git_z", lambda *_args: [line])

    with pytest.raises(ValueError, match="regular Git blob"):
        surface_validator._surface_blob_map(tmp_path, "HEAD")

