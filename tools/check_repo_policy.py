"""Validate tracked repository content against D10 release-hygiene policy."""

from __future__ import annotations

from pathlib import Path

from _integration_bootstrap import load_integration_module

ROOT = Path(__file__).resolve().parents[1]
_REPO_POLICY = load_integration_module(ROOT, "repo_policy")


def main() -> int:
    try:
        _REPO_POLICY.validate_repository_policy(ROOT)
    except _REPO_POLICY.RepositoryPolicyError as exc:
        print(str(exc))
        return 1
    print("repository_policy=pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
