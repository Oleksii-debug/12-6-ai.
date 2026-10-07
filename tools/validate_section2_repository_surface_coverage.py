from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


_REQUIRED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "observed_main_sha",
    "observed_main_tree_sha",
    "current_repository_main_sha",
    "current_repository_main_tree_sha",
    "expected_main_surface_count",
    "expected_main_capability_counts",
    "rules",
    "candidate_overrides",
}
_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _load_strict_json(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{path} is not strict unambiguous UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} root must be an object")
    return value


def _is_surface(path: str) -> bool:
    if path == "pyproject.toml":
        return True
    if path.startswith(".github/workflows/") and path.endswith((".yml", ".yaml")):
        return True
    if not path.startswith("tools/"):
        return False
    return path.endswith((".py", ".sh")) or path.startswith(
        "tools/eval647_transport_bin/"
    )


def _is_capability_surface(path: str) -> bool:
    return (
        path.startswith("src/twelve_six/")
        and path.endswith(".py")
    ) or _is_surface(path)


def _run_git(root: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(args)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout.splitlines()


def _resolve_live_main_sha(root: Path) -> str:
    for ref in ("refs/remotes/origin/main", "refs/heads/main"):
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", ref],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0:
            sha = completed.stdout.strip()
            if _SHA40_RE.fullmatch(sha) is None:
                raise ValueError("live main ref did not resolve to lowercase 40-hex")
            return sha
    raise ValueError("live main ref is unavailable for current-main equivalence proof")


def _surface_blob_map(root: Path, treeish: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in _run_git(root, "ls-tree", "-r", treeish):
        try:
            metadata, path = line.split("\t", 1)
            _mode, kind, blob_sha = metadata.split()
        except ValueError as exc:
            raise ValueError("git ls-tree emitted a non-canonical record") from exc
        if kind != "blob" or not _is_capability_surface(path):
            continue
        if _SHA40_RE.fullmatch(blob_sha) is None:
            raise ValueError("git ls-tree emitted a malformed blob SHA")
        result[path] = blob_sha
    return result


def _validate_rule(rule: object) -> dict[str, str]:
    if not isinstance(rule, dict) or set(rule) != {
        "rule_id",
        "selector",
        "pattern",
        "capability_id",
    }:
        raise ValueError("surface rule schema is non-canonical")
    for field in ("rule_id", "selector", "pattern", "capability_id"):
        if not isinstance(rule[field], str) or not rule[field]:
            raise ValueError(f"surface rule {field} must be non-empty text")
    if rule["selector"] not in {"exact", "regex"}:
        raise ValueError("surface rule selector must be exact or regex")
    if rule["selector"] == "regex":
        try:
            re.compile(rule["pattern"])
        except re.error as exc:
            raise ValueError("surface rule regex is invalid") from exc
    return rule  # type: ignore[return-value]


def _matching_rules(path: str, rules: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    matches = []
    for rule in rules:
        if rule["selector"] == "exact":
            matched = path == rule["pattern"]
        else:
            matched = re.fullmatch(rule["pattern"], path) is not None
        if matched:
            matches.append(rule)
    return matches


def _candidate_surface_paths(
    current_main_blobs: dict[str, str],
    checkout_blobs: dict[str, str],
) -> set[str]:
    """Return executable surfaces added or byte-changed from current main."""

    return {
        path
        for path, blob_sha in checkout_blobs.items()
        if current_main_blobs.get(path) != blob_sha
    }


def _classify_main_surface(
    path: str,
    rules: tuple[dict[str, str], ...],
) -> str:
    matches = _matching_rules(path, rules)
    if not matches:
        raise ValueError(f"unclassified accepted-main executable surface: {path}")
    if len(matches) != 1:
        ids = [rule["rule_id"] for rule in matches]
        raise ValueError(f"ambiguous executable surface classification {path}: {ids}")
    return matches[0]["capability_id"]


def validate_repository_surface_coverage(
    *,
    repo_root: Path,
    inventory_path: Path,
    capability_registry_path: Path,
) -> dict[str, object]:
    inventory = _load_strict_json(inventory_path)
    if set(inventory) != _REQUIRED_TOP_LEVEL_FIELDS:
        raise ValueError("repository executable surface inventory schema is non-canonical")
    schema_version = inventory["schema_version"]
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != 1
    ):
        raise ValueError("repository executable surface schema_version must equal integer 1")

    main_sha = inventory["observed_main_sha"]
    main_tree_sha = inventory["observed_main_tree_sha"]
    current_main_sha = inventory["current_repository_main_sha"]
    current_main_tree_sha = inventory["current_repository_main_tree_sha"]
    for field, value in (
        ("observed_main_sha", main_sha),
        ("observed_main_tree_sha", main_tree_sha),
        ("current_repository_main_sha", current_main_sha),
        ("current_repository_main_tree_sha", current_main_tree_sha),
    ):
        if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
            raise ValueError(f"{field} must be lowercase 40-hex")

    live_main_sha = _resolve_live_main_sha(repo_root)
    if current_main_sha != live_main_sha:
        raise ValueError(
            "current_repository_main_sha does not match the live repository main ref"
        )

    expected_count = inventory["expected_main_surface_count"]
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise ValueError("expected_main_surface_count must be an integer")
    if expected_count <= 0:
        raise ValueError("expected_main_surface_count must be positive")

    raw_counts = inventory["expected_main_capability_counts"]
    if not isinstance(raw_counts, dict) or not raw_counts:
        raise ValueError("expected_main_capability_counts must be a non-empty object")
    expected_counts: dict[str, int] = {}
    for capability_id, count in raw_counts.items():
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("capability count key must be non-empty text")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("capability counts must be positive integers")
        expected_counts[capability_id] = count
    if sum(expected_counts.values()) != expected_count:
        raise ValueError("expected capability counts do not sum to expected surface count")

    raw_rules = inventory["rules"]
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError("rules must be a non-empty array")
    rules = tuple(_validate_rule(rule) for rule in raw_rules)
    rule_ids = [rule["rule_id"] for rule in rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("surface rule ids must be unique")

    capability_registry = _load_strict_json(capability_registry_path)
    raw_capabilities = capability_registry.get("capabilities")
    if not isinstance(raw_capabilities, list):
        raise ValueError("capability registry capabilities must be an array")
    capabilities: dict[str, dict[str, Any]] = {}
    for capability in raw_capabilities:
        if not isinstance(capability, dict):
            raise ValueError("capability entry must be an object")
        capability_id = capability.get("capability_id")
        if not isinstance(capability_id, str) or not capability_id:
            raise ValueError("capability_id must be non-empty text")
        if capability_id in capabilities:
            raise ValueError("capability ids must be unique")
        capabilities[capability_id] = capability

    declared_capability_ids = {
        rule["capability_id"] for rule in rules
    } | set(expected_counts)
    for capability_id in declared_capability_ids:
        capability = capabilities.get(capability_id)
        if capability is None:
            raise ValueError(f"unknown classified capability: {capability_id}")
        journey_ids = capability.get("journey_ids")
        if not isinstance(journey_ids, list) or not journey_ids:
            raise ValueError(
                f"classified capability lacks user/operator journey: {capability_id}"
            )

    resolved_tree = _run_git(repo_root, "show", "-s", "--format=%T", main_sha)
    if resolved_tree != [main_tree_sha]:
        raise ValueError("observed_main_tree_sha does not match observed_main_sha")
    resolved_current_tree = _run_git(
        repo_root, "show", "-s", "--format=%T", current_main_sha
    )
    if resolved_current_tree != [current_main_tree_sha]:
        raise ValueError(
            "current_repository_main_tree_sha does not match current_repository_main_sha"
        )

    qualified_surface_blobs = _surface_blob_map(repo_root, main_tree_sha)
    current_surface_blobs = _surface_blob_map(repo_root, current_main_tree_sha)
    if current_surface_blobs != qualified_surface_blobs:
        qualified_paths = set(qualified_surface_blobs)
        current_paths = set(current_surface_blobs)
        changed = sorted(
            path
            for path in qualified_paths & current_paths
            if qualified_surface_blobs[path] != current_surface_blobs[path]
        )
        raise ValueError(
            "current main capability-bearing surface drift from qualified baseline: "
            f"added={sorted(current_paths - qualified_paths)}, "
            f"removed={sorted(qualified_paths - current_paths)}, "
            f"changed={changed}"
        )

    main_paths = sorted(
        path.strip()
        for path in _run_git(
            repo_root,
            "ls-tree",
            "-r",
            "--name-only",
            main_tree_sha,
        )
        if _is_surface(path.strip())
    )
    if len(main_paths) != expected_count:
        raise ValueError(
            "accepted-main executable surface count drift: "
            f"expected={expected_count}, actual={len(main_paths)}"
        )

    observed_counts = Counter(_classify_main_surface(path, rules) for path in main_paths)
    if dict(sorted(observed_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(
            "accepted-main executable capability distribution drift: "
            f"expected={expected_counts}, actual={dict(observed_counts)}"
        )

    raw_overrides = inventory["candidate_overrides"]
    if not isinstance(raw_overrides, list):
        raise ValueError("candidate_overrides must be an array")
    candidate_overrides: dict[str, str] = {}
    for item in raw_overrides:
        if not isinstance(item, dict) or set(item) != {"path", "capability_id"}:
            raise ValueError("candidate override schema is non-canonical")
        path = item["path"]
        capability_id = item["capability_id"]
        if not isinstance(path, str) or not _is_surface(path):
            raise ValueError("candidate override path is outside executable surface scope")
        if not isinstance(capability_id, str) or capability_id not in capabilities:
            raise ValueError("candidate override maps unknown capability")
        if path in candidate_overrides:
            raise ValueError("candidate override paths must be unique")
        journey_ids = capabilities[capability_id].get("journey_ids")
        if not isinstance(journey_ids, list) or not journey_ids:
            raise ValueError("candidate override capability lacks journey")
        candidate_overrides[path] = capability_id

    checkout_paths = sorted(
        path.strip()
        for path in _run_git(repo_root, "ls-files")
        if _is_surface(path.strip())
    )
    main_set = set(main_paths)
    checkout_set = set(checkout_paths)

    current_executable_blobs = {
        path: blob_sha
        for path, blob_sha in current_surface_blobs.items()
        if _is_surface(path)
    }
    checkout_executable_blobs = {
        path: blob_sha
        for path, blob_sha in _surface_blob_map(repo_root, "HEAD").items()
        if _is_surface(path)
    }
    candidate_actual = _candidate_surface_paths(
        current_executable_blobs,
        checkout_executable_blobs,
    )
    candidate_expected = set(candidate_overrides)
    if candidate_actual != candidate_expected:
        raise ValueError(
            "repository executable stacked inventory drift: "
            f"unmapped_checkout={sorted(candidate_actual - candidate_expected)}, "
            f"stale_candidate_overrides={sorted(candidate_expected - candidate_actual)}"
        )
    removed_from_checkout = main_set.difference(checkout_set)
    if removed_from_checkout:
        raise ValueError(
            "accepted-main executable surfaces disappeared from checkout: "
            f"{sorted(removed_from_checkout)}"
        )

    return {
        "observed_main_sha": main_sha,
        "observed_main_tree_sha": main_tree_sha,
        "current_repository_main_sha": current_main_sha,
        "current_repository_main_tree_sha": current_main_tree_sha,
        "qualified_current_equivalent_surface_count": len(qualified_surface_blobs),
        "accepted_main_surface_count": len(main_paths),
        "candidate_overlay_surface_count": len(candidate_actual),
        "checkout_surface_count": len(checkout_paths),
        "main_capability_counts": dict(sorted(observed_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate Section-2 repository executable capability coverage."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).parents[1])
    parser.add_argument(
        "--inventory",
        type=Path,
        default=Path(__file__).parents[1]
        / "configs"
        / "control"
        / "product_repository_executable_surface_rules_v1.json",
    )
    parser.add_argument(
        "--capability-registry",
        type=Path,
        default=Path(__file__).parents[1]
        / "configs"
        / "control"
        / "product_capabilities_v1.json",
    )
    args = parser.parse_args()
    result = validate_repository_surface_coverage(
        repo_root=args.repo_root,
        inventory_path=args.inventory,
        capability_registry_path=args.capability_registry,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
