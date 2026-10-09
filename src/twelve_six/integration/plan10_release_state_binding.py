"""Plan 10 S2: fail-closed assembly of existing component state (not a release claim).

Caller must stop writers for a cross-component consistent snapshot. SQLite's online
backup preserves each component's transactional state; component stores own schemas.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
import tempfile
from pathlib import Path

SCHEMA = "12-6.plan10.release-state.v1"
KINDS = frozenset({"configs", "manifests", "registries", "model", "checkpoints", "tasks", "memory"})
DB_KINDS = {"tasks", "memory"}
STORE_TABLES = {"tasks": {"tasks", "history"}, "memory": {"meta", "records"}}
MAX_BYTES = 256 * 1024 * 1024  # LOCAL_FREE transport cap; not a production size claim


class StateBindingError(ValueError):
    """Invalid, unsafe, or unsupported bundle."""


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(data: object) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _safe_files(root: Path) -> list[tuple[str, Path]]:
    if root.is_symlink() or not root.is_dir():
        raise StateBindingError("missing or unsafe component directory")
    result: list[tuple[str, Path]] = []
    for base, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            item = Path(base) / name
            mode = item.lstat().st_mode
            if stat.S_ISLNK(mode) or not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise StateBindingError("nonregular component member")
        for name in files:
            item = Path(base) / name
            relative = item.relative_to(root).as_posix()
            if any(part in ("", ".", "..") for part in Path(relative).parts):
                raise StateBindingError("invalid component member")
            result.append((relative, item))
    if not result:
        raise StateBindingError("empty component")
    return sorted(result)


def _read_regular(path: Path) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise StateBindingError("unsafe source member")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise StateBindingError("changed source member")
        if opened.st_size > MAX_BYTES:
            raise StateBindingError("oversized source member")
        with os.fdopen(os.dup(fd), "rb") as handle:
            payload = handle.read(MAX_BYTES + 1)
        if len(payload) > MAX_BYTES or os.fstat(fd).st_size != len(payload):
            raise StateBindingError("changed or oversized source")
        return payload
    finally:
        os.close(fd)


def _validate_db(db: sqlite3.Connection, kind: str) -> None:
    if db.execute("PRAGMA user_version").fetchone()[0] != 1:
        raise StateBindingError("unsupported store schema")
    tables = {row[0] for row in db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    if tables != STORE_TABLES[kind] or db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        raise StateBindingError("invalid canonical component store")


def _copy_database(src: Path, target: Path, kind: str) -> bytes:
    if src.is_symlink() or not src.is_file():
        raise StateBindingError("unsafe sqlite source")
    try:
        with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as origin:
            _validate_db(origin, kind)
            with sqlite3.connect(target) as out:
                origin.backup(out)
                _validate_db(out, kind)
        return _read_regular(target)
    except sqlite3.DatabaseError as exc:
        raise StateBindingError("invalid sqlite state") from exc


def capture(components: dict[str, Path], destination: Path, *, writers_stopped: bool) -> str:
    """Create atomic offline snapshot; return manifest SHA-256 for external pinning."""
    if writers_stopped is not True or set(components) != KINDS:
        raise StateBindingError("quiescence and all component categories are required")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink() or not destination.parent.is_dir():
        raise StateBindingError("unsafe or existing destination")
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".release-state-") as temp:
        stage = Path(temp) / "snapshot"
        stage.mkdir()
        members: dict[str, str] = {}
        total = 0
        for kind in sorted(KINDS):
            root = Path(components[kind])
            items = _safe_files(root)
            if kind in DB_KINDS and [name for name, _ in items] != ["state.sqlite"]:
                raise StateBindingError("expected one canonical SQLite store")
            for rel, src in items:
                key = f"{kind}/{rel}"
                target = stage / key
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind in DB_KINDS:
                    payload = _copy_database(src, target, kind)
                else:
                    payload = _read_regular(src)
                    target.write_bytes(payload)
                total += len(payload)
                if total > MAX_BYTES:
                    raise StateBindingError("bundle exceeds LOCAL_FREE cap")
                members[key] = _sha(payload)
        manifest = {"schema": SCHEMA, "members": members}
        raw = _canonical(manifest)
        (stage / "manifest.json").write_bytes(raw)
        digest = _sha(raw)
        os.rename(stage, destination)
        return digest


def restore(snapshot: Path, destination: Path, *, manifest_sha256: str) -> None:
    """Verify externally pinned bundle and atomically publish into a new root."""
    snapshot, destination = Path(snapshot), Path(destination)
    if (not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64
            or any(c not in "0123456789abcdef" for c in manifest_sha256)
            or destination.exists() or destination.is_symlink() or not destination.parent.is_dir()):
        raise StateBindingError("unsafe restore request")
    observed = dict(_safe_files(snapshot))
    if "manifest.json" not in observed:
        raise StateBindingError("missing manifest")
    raw = _read_regular(observed.pop("manifest.json"))
    if _sha(raw) != manifest_sha256:
        raise StateBindingError("untrusted manifest")
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise StateBindingError("malformed manifest") from exc
    if (not isinstance(manifest, dict) or set(manifest) != {"schema", "members"}
            or manifest["schema"] != SCHEMA or not isinstance(manifest["members"], dict)
            or _canonical(manifest) != raw or set(observed) != set(manifest["members"])):
        raise StateBindingError("unsupported or incomplete snapshot")
    kinds = set()
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".release-restore-") as temp:
        stage = Path(temp) / "restored"
        stage.mkdir()
        for key, expected in sorted(manifest["members"].items()):
            parts = key.split("/")
            if (len(parts) < 2 or parts[0] not in KINDS or any(
                    part in ("", ".", "..") for part in parts)
                    or not isinstance(expected, str) or len(expected) != 64):
                raise StateBindingError("unsafe manifest member")
            kinds.add(parts[0])
            payload = _read_regular(observed[key])
            if _sha(payload) != expected:
                raise StateBindingError("snapshot payload mismatch")
            target = stage / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        if kinds != KINDS:
            raise StateBindingError("missing component category")
        for kind in DB_KINDS:
            root = stage / kind
            if [name for name, _ in _safe_files(root)] != ["state.sqlite"]:
                raise StateBindingError("invalid store member")
            with sqlite3.connect(root / "state.sqlite") as db:
                _validate_db(db, kind)
        os.rename(stage, destination)
