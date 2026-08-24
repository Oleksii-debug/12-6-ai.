"""Capture deterministic CI/release environment inventory evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from twelve_six.integration.environment_inventory import (
    EnvironmentInventoryError,
    capture_current_environment,
    validate_environment_inventory,
)


def _source_sha(argument: str | None) -> str:
    if argument:
        return argument
    pull_request_head = os.environ.get("SOURCE_SHA")
    if pull_request_head:
        return pull_request_head
    github_sha = os.environ.get("GITHUB_SHA")
    if github_sha:
        return github_sha
    raise EnvironmentInventoryError("source SHA is required via --source-sha, SOURCE_SHA or GITHUB_SHA")


def _repository(argument: str | None) -> str:
    value = argument or os.environ.get("GITHUB_REPOSITORY")
    if not value:
        raise EnvironmentInventoryError(
            "repository identity is required via --repository or GITHUB_REPOSITORY"
        )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository")
    parser.add_argument("--source-sha")
    parser.add_argument("--pyproject", default="pyproject.toml")
    args = parser.parse_args()

    try:
        inventory = capture_current_environment(
            repository=_repository(args.repository),
            source_sha=_source_sha(args.source_sha),
            pyproject_path=args.pyproject,
        )
        validate_environment_inventory(inventory)
    except EnvironmentInventoryError as exc:
        print(f"environment_inventory=fail: {exc}")
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"environment_inventory=pass path={output}")
    print(f"environment_inventory_sha256={inventory['inventory_sha256']}")
    print(f"environment_package_count={inventory['summary']['package_count']}")
    print(
        "environment_unresolved_license_metadata_count="
        f"{inventory['summary']['unresolved_license_metadata_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
