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

    assert result["observed_main_sha"] == "0e1f301c5123b4e52c111cb94264cfd61b60bf4b"
    assert result["observed_main_tree_sha"] == "4fd06e8836450e61ab47e39657c96c6b6f76792e"
    assert result["current_repository_main_sha"] == "cb94ca7a0c2b9a453356db45ef4d228c44e0ee21"
    assert result["current_repository_main_tree_sha"] == "17edf21d66709f6e8a7c217e138b33c0bf9a0217"
    assert result["qualified_current_equivalent_surface_count"] == 237
    assert result["accepted_main_surface_count"] == 120
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
        "executable-capability-map": 1,
        "learned20m-control-plane": 5,
        "model-spec-identity": 2,
        "package-runtime": 1,
        "portable-run-authority": 2,
        "project-control-plane": 1,
    }


def test_section2_validator_is_integrated_into_qualified_baseline() -> None:
    payload = _load_strict_json(_INVENTORY)

    assert payload["candidate_overrides"] == [
        {
            "path": "tools/validate_section2_repository_surface_coverage.py",
            "capability_id": "executable-capability-map",
        }
    ]
    assert {
        "rule_id": "section2-surface-validator",
        "selector": "exact",
        "pattern": "tools/validate_section2_repository_surface_coverage.py",
        "capability_id": "executable-capability-map",
    } in payload["rules"]


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
    payload["candidate_overrides"] = [
        {
            "path": "tools/validate_section2_repository_surface_coverage.py",
            "capability_id": "model-spec-identity",
        }
    ]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must remain UNAVAILABLE"):
        _validate(inventory)


def test_repository_surface_coverage_rejects_candidate_override_capability_remap(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    target = next(
        item
        for item in payload["candidate_overrides"]
        if item["path"] == "tools/validate_section2_repository_surface_coverage.py"
    )
    assert target["capability_id"] == "executable-capability-map"
    target["capability_id"] = "learned-20m-base"

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="cannot remap accepted-main executable capability",
    ):
        _validate(inventory)


def test_repository_surface_coverage_rejects_missing_candidate_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_surface_blob_map = surface_validator._surface_blob_map

    def drifted_surface_blob_map(root: Path, treeish: str) -> dict[str, str]:
        blobs = original_surface_blob_map(root, treeish)
        if treeish == "HEAD":
            blobs = dict(blobs)
            path = "tools/validate_section2_repository_surface_coverage.py"
            mode, _blob_sha = blobs[path].split(":", 1)
            blobs[path] = f"{mode}:{'f' * 40}"
        return blobs

    monkeypatch.setattr(
        surface_validator,
        "_surface_blob_map",
        drifted_surface_blob_map,
    )
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


def test_repository_surface_coverage_rejects_accepted_main_capability_remap(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    target = next(
        rule
        for rule in payload["rules"]
        if rule["rule_id"] == "section2-surface-validator"
    )
    assert target["capability_id"] == "executable-capability-map"
    target["capability_id"] = "model-spec-identity"
    payload["expected_main_capability_counts"]["model-spec-identity"] += 1
    del payload["expected_main_capability_counts"]["executable-capability-map"]

    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="accepted-main executable capability mapping drift",
    ):
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


def test_repository_surface_coverage_accepts_equivalent_historical_main_receipt(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = payload["observed_main_sha"]
    payload["current_repository_main_tree_sha"] = payload["observed_main_tree_sha"]
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    result = _validate(inventory)
    assert result["current_repository_main_sha"] == payload["observed_main_sha"]
    assert result["current_repository_main_tree_sha"] == payload["observed_main_tree_sha"]


def test_repository_surface_coverage_rejects_non_equivalent_historical_receipt(
    tmp_path: Path,
) -> None:
    payload = json.loads(_INVENTORY.read_text(encoding="utf-8"))
    payload["current_repository_main_sha"] = "5c041ca56edda55a5c3334f722361754051e121c"
    payload["current_repository_main_tree_sha"] = "95ad101c8965fd49c1711027146253a093fce2f8"
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="pinned current-main receipt capability-bearing surface drift",
    ):
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
    payload["observed_main_sha"] = "cb94ca7a0c2b9a453356db45ef4d228c44e0ee21"
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
        stdout = f"{sha}\n" if "rev-parse" in command else ""
        if command[-1] == "HEAD":
            stdout = "ok\n"
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

