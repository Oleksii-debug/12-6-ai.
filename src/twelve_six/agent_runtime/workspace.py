from __future__ import annotations

from pathlib import Path


class WorkspaceViolation(ValueError):
    pass


def resolve_workspace_path(workspace: Path, relative: str) -> Path:
    root = workspace.resolve(strict=True)
    requested = Path(relative)
    if requested.is_absolute():
        raise WorkspaceViolation("absolute paths are not allowed")
    candidate = (root / requested).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WorkspaceViolation("path escapes isolated workspace") from exc
    return candidate


def ensure_workspace(workspace: Path) -> Path:
    workspace.mkdir(parents=True, exist_ok=True)
    root = workspace.resolve(strict=True)
    if not root.is_dir():
        raise WorkspaceViolation("workspace must be a directory")
    return root
