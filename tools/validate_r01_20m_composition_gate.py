from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

SHA40 = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_ROLES = {
    "MODEL_MECHANICS",
    "RESEARCH_CORPUS_V1",
    "TOKENIZER",
    "CHECKPOINT_INTEGRITY",
    "SELECTION_EVALUATION",
    "TRAINER_RUNTIME",
    "TRAINING_RECIPE",
}
REQUIRED_REQUIREMENTS = {
    "single_executable_tree_required",
    "all_required_components_must_be_ancestors",
    "exact_candidate_head_ci_required",
    "component_green_without_composition_is_insufficient",
    "long_training_requires_composition_pass",
}


def _is_sha(value: object) -> bool:
    return isinstance(value, str) and SHA40.fullmatch(value) is not None


def validate_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if data.get("gate_id") != "R01-20M-SINGLE-COMPOSED-CANDIDATE-V1":
        errors.append("gate_id mismatch")

    status = data.get("status")
    if status not in {"BLOCKED_UNCOMPOSED", "READY_FOR_ANCESTRY_CHECK", "PASS"}:
        errors.append("status is invalid")

    requirements = data.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("requirements must be an object")
    else:
        for key in REQUIRED_REQUIREMENTS:
            if requirements.get(key) is not True:
                errors.append(f"requirements.{key} must be true")

    components = data.get("components")
    if not isinstance(components, list):
        errors.append("components must be an array")
        components = []

    roles: list[str] = []
    missing_authorities: list[str] = []
    for index, component in enumerate(components):
        if not isinstance(component, dict):
            errors.append(f"components[{index}] must be an object")
            continue
        role = component.get("role")
        if not isinstance(role, str):
            errors.append(f"components[{index}].role must be a string")
            continue
        roles.append(role)
        if component.get("required") is not True:
            errors.append(f"component {role} must remain required")
        authority_sha = component.get("authority_sha")
        if authority_sha is None:
            missing_authorities.append(role)
        elif not _is_sha(authority_sha):
            errors.append(f"component {role} authority_sha must be a 40-hex SHA or null")
        state = component.get("state")
        if not isinstance(state, str) or not state:
            errors.append(f"component {role} state must be non-empty")

    if len(roles) != len(set(roles)):
        errors.append("component roles must be unique")
    if set(roles) != REQUIRED_ROLES:
        errors.append("required component role set mismatch")

    candidate = data.get("candidate_head_sha")
    if candidate is not None and not _is_sha(candidate):
        errors.append("candidate_head_sha must be a 40-hex SHA or null")

    if status in {"READY_FOR_ANCESTRY_CHECK", "PASS"}:
        if not _is_sha(candidate):
            errors.append(f"{status} requires an exact candidate_head_sha")
        if missing_authorities:
            errors.append(f"{status} requires every authority_sha to be bound")

    boundary = data.get("truth_boundary")
    if not isinstance(boundary, dict):
        errors.append("truth_boundary must be an object")
    else:
        for key in (
            "learned_20m_checkpoint_exists",
            "long_training_authorized",
            "paid_compute_authorized",
            "stage_promotion_authorized",
        ):
            if boundary.get(key) is not False:
                errors.append(f"truth_boundary.{key} must remain false in v1")

    if status == "PASS":
        errors.append("committed v1 manifest must not self-assert PASS without live ancestry proof")

    return errors


def _git_commit_exists(repo_root: Path, sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _git_is_ancestor(repo_root: Path, ancestor: str, candidate: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, candidate],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def assess_composition(
    data: dict[str, Any],
    *,
    repo_root: Path = Path("."),
    commit_exists: Callable[[str], bool] | None = None,
    is_ancestor: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    errors = validate_manifest(data)
    if errors:
        return {"decision": "INVALID_MANIFEST", "errors": errors, "ready": False}

    components = {item["role"]: item for item in data["components"]}
    missing = sorted(
        role for role, item in components.items() if item.get("authority_sha") is None
    )
    if missing:
        return {
            "decision": "BLOCKED_MISSING_AUTHORITIES",
            "missing_roles": missing,
            "ready": False,
        }

    candidate = data.get("candidate_head_sha")
    if not isinstance(candidate, str):
        return {"decision": "BLOCKED_NO_CANDIDATE_HEAD", "ready": False}

    exists = commit_exists or (lambda sha: _git_commit_exists(repo_root, sha))
    ancestor = is_ancestor or (
        lambda component_sha, candidate_sha: _git_is_ancestor(
            repo_root, component_sha, candidate_sha
        )
    )

    if not exists(candidate):
        return {
            "decision": "BLOCKED_CANDIDATE_COMMIT_NOT_FOUND",
            "candidate_head_sha": candidate,
            "ready": False,
        }

    missing_commits: list[str] = []
    non_ancestors: list[str] = []
    for role in sorted(REQUIRED_ROLES):
        component_sha = components[role]["authority_sha"]
        assert isinstance(component_sha, str)
        if not exists(component_sha):
            missing_commits.append(role)
        elif not ancestor(component_sha, candidate):
            non_ancestors.append(role)

    if missing_commits:
        return {
            "decision": "BLOCKED_COMPONENT_COMMIT_NOT_FOUND",
            "missing_commit_roles": missing_commits,
            "candidate_head_sha": candidate,
            "ready": False,
        }
    if non_ancestors:
        return {
            "decision": "BLOCKED_COMPONENT_NOT_ANCESTOR",
            "non_ancestor_roles": non_ancestors,
            "candidate_head_sha": candidate,
            "ready": False,
        }

    return {
        "decision": "PASS_SINGLE_COMPOSED_CANDIDATE",
        "candidate_head_sha": candidate,
        "component_roles": sorted(REQUIRED_ROLES),
        "ready": True,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate R01 single composed candidate-head ancestry gate."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/research/r01_20m_composition_gate_v1.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--assert-ready",
        action="store_true",
        help="Exit nonzero unless all authorities are ancestors of one exact candidate head.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(json.dumps({"decision": "INVALID_MANIFEST", "ready": False}, sort_keys=True))
        return 1

    assessment = assess_composition(data, repo_root=args.repo_root)
    print(json.dumps(assessment, sort_keys=True))
    if assessment["decision"] == "INVALID_MANIFEST":
        return 1
    if args.assert_ready and not assessment["ready"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
