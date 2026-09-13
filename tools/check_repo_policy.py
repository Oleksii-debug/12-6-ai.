"""Validate tracked repository content against D10 release-hygiene policy."""

from __future__ import annotations

from pathlib import Path

from twelve_six.integration.repo_policy import RepositoryPolicyError, validate_repository_policy


def main() -> int:
    try:
        validate_repository_policy(Path("."))
    except RepositoryPolicyError as exc:
        print(str(exc))
        return 1
    print("repository_policy=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
