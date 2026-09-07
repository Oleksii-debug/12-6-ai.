"""Fail-closed repository hygiene policy for integration and release branches."""

from __future__ import annotations

import subprocess
from pathlib import Path

MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".pth",
    ".pt",
    ".gguf",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tgz",
    ".tar.gz",
)
FORBIDDEN_TOP_LEVEL_DIRS = frozenset({"artifacts", "checkpoints"})


class RepositoryPolicyError(RuntimeError):
    """Raised when tracked repository content violates integration policy."""


def _is_forbidden_suffix(path: Path) -> bool:
    lowered = path.as_posix().lower()
    return any(lowered.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)


def validate_tracked_paths(repo_root: Path, relative_paths: tuple[str, ...]) -> None:
    """Validate already-resolved tracked paths against the git-content policy."""

    root = repo_root.resolve()
    violations: list[str] = []
    for relative in relative_paths:
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            violations.append(f"unsafe tracked path: {relative}")
            continue
        if candidate.parts and candidate.parts[0] in FORBIDDEN_TOP_LEVEL_DIRS:
            violations.append(f"tracked runtime artifact directory is forbidden: {relative}")
        if _is_forbidden_suffix(candidate):
            violations.append(f"tracked archive/model artifact format is forbidden: {relative}")

        full_path = root / candidate
        if not full_path.exists() and not full_path.is_symlink():
            violations.append(f"tracked path is missing from checkout: {relative}")
            continue
        if full_path.is_symlink():
            violations.append(f"tracked symlink is forbidden: {relative}")
            continue
        if full_path.is_file() and full_path.stat().st_size > MAX_TRACKED_BYTES:
            violations.append(
                f"tracked file exceeds {MAX_TRACKED_BYTES} bytes: {relative}"
            )

    if violations:
        raise RepositoryPolicyError("\n".join(sorted(violations)))


def tracked_paths(repo_root: Path) -> tuple[str, ...]:
    """Return exact tracked paths from git without following untracked filesystem content."""

    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        timeout=10,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RepositoryPolicyError(f"git ls-files failed: {stderr or result.returncode}")
    return tuple(
        item.decode("utf-8", errors="strict")
        for item in result.stdout.split(b"\0")
        if item
    )


def validate_repository_policy(repo_root: str | Path = ".") -> None:
    root = Path(repo_root).resolve()
    validate_tracked_paths(root, tracked_paths(root))
