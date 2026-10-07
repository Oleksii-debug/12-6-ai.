from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass

WORKFLOW_DIR = ".github/workflows/"
CANONICAL_WORKFLOW = ".github/workflows/ci.yml"


@dataclass(frozen=True)
class Change:
    status: str
    old_path: str | None
    new_path: str | None

    @property
    def destination(self) -> str | None:
        return self.new_path or self.old_path


def parse_name_status(output: str) -> list[Change]:
    changes: list[Change] = []
    for raw_line in output.splitlines():
        if not raw_line.strip():
            continue
        fields = raw_line.split("\t")
        status = fields[0]
        kind = status[:1]
        if kind in {"R", "C"}:
            if len(fields) != 3:
                raise ValueError(f"Malformed rename/copy line: {raw_line!r}")
            changes.append(Change(status=status, old_path=fields[1], new_path=fields[2]))
        else:
            if len(fields) != 2:
                raise ValueError(f"Malformed change line: {raw_line!r}")
            changes.append(Change(status=status, old_path=fields[1], new_path=None))
    return changes


def policy_violations(changes: list[Change]) -> list[str]:
    violations: list[str] = []
    for change in changes:
        kind = change.status[:1]
        destination = change.destination

        if kind == "D" and change.old_path == CANONICAL_WORKFLOW:
            violations.append("canonical CI workflow may not be deleted")
            continue

        if kind == "R" and change.old_path == CANONICAL_WORKFLOW:
            violations.append("canonical CI workflow may not be renamed")
            continue

        if (
            kind in {"A", "C", "R"}
            and destination is not None
            and destination.startswith(WORKFLOW_DIR)
            and destination != CANONICAL_WORKFLOW
        ):
            violations.append(f"new dedicated workflow is prohibited: {destination}")

    return violations


def git_name_status(base: str, head: str) -> str:
    result = subprocess.run(
        ["git", "diff", "--name-status", "--find-renames", "--find-copies", base, head],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when a PR creates a new dedicated GitHub Actions workflow."
    )
    parser.add_argument("--base", required=True, help="Base commit SHA")
    parser.add_argument("--head", required=True, help="Head commit SHA")
    args = parser.parse_args()

    changes = parse_name_status(git_name_status(args.base, args.head))
    violations = policy_violations(changes)
    if violations:
        print("CI workflow budget check: FAIL")
        for violation in violations:
            print(f"- {violation}")
        print(
            "Use the shared .github/workflows/ci.yml. "
            "A permanent additional workflow requires a separately reviewed CI policy change."
        )
        return 1

    print("CI workflow budget check: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
