"""Plan 2 S17: compare two physical rebuild trees against a pinned byte inventory.

This is a small, non-authorizing witness for S17.3. It neither builds a second
S3-S14 data pipeline nor certifies independent processes, legal admission,
32K tokenizer identity, holdout separation, or a production/Plan9 release.
The expected inventory hash MUST come from separately verified authority.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import unicodedata
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2-s17-clean-rebuild-byte-witness.v1"
HEX256 = re.compile(r"[0-9a-f]{64}\Z")


class RebuildDenied(ValueError):
    """Unsafe, incomplete, changed or unpinned physical rebuild output."""


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise RebuildDenied(reason)


def _root(path: Path) -> Path:
    _need(isinstance(path, Path), "rebuild root must be a Path")
    for part in (path, *path.parents):
        _need(not part.is_symlink(), "linked rebuild root or ancestor")
    _need(path.is_dir(), "rebuild root is not a directory")
    return path.resolve(strict=True)


def _hash_regular_file(path: Path, observed: os.stat_result) -> str:
    _need(stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1,
          "nonregular or hardlinked artifact member")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            _need(stat.S_ISREG(before.st_mode)
                  and (before.st_dev, before.st_ino) ==
                  (observed.st_dev, observed.st_ino),
                  "artifact member replaced before read")
            _need(before.st_nlink == 1,
                  "artifact member hardlinked before read")
            _need((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                  (observed.st_size, observed.st_mtime_ns, observed.st_ctime_ns),
                  "artifact member changed before read")
            hasher = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                hasher.update(chunk)
            after = os.fstat(stream.fileno())
    except OSError as exc:
        raise RebuildDenied("artifact member unreadable") from exc
    _need(after.st_nlink == 1,
          "artifact member hardlinked during read")
    _need((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
          (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
          "artifact member changed during read")
    try:
        latest = path.lstat()
    except OSError as exc:
        raise RebuildDenied("artifact member disappeared") from exc
    _need(stat.S_ISREG(latest.st_mode) and latest.st_nlink == 1,
          "artifact member hardlinked after read")
    _need((latest.st_dev, latest.st_ino, latest.st_size,
           latest.st_mtime_ns, latest.st_ctime_ns) ==
          (after.st_dev, after.st_ino, after.st_size,
           after.st_mtime_ns, after.st_ctime_ns),
          "artifact member changed after read")
    return hasher.hexdigest()


_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} | {
    f"{prefix}{index}" for prefix in ("COM", "LPT")
    for index in ("1", "2", "3", "4", "5", "6", "7", "8", "9", "¹", "²", "³")
}
_WINDOWS_UNSAFE = set('<>:"/\\|?*')


def _portable_key(relative: str, seen: set[str]) -> None:
    """Reject Linux-only aliases and invalid Windows member names before hashing."""
    parts = relative.split("/")
    for part in parts:
        _need(part == unicodedata.normalize("NFC", part),
              "nonportable Unicode normalization in artifact path")
        _need(bool(part) and part not in (".", "..") and
              not part.endswith((" ", ".")) and
              not any(ord(char) < 32 or char in _WINDOWS_UNSAFE for char in part) and
              part.split(".", 1)[0].upper() not in _WINDOWS_RESERVED,
              "nonportable Windows artifact path")
    key = relative.casefold()
    _need(key not in seen, "portable artifact path collision")
    seen.add(key)


def inventory(root: Path) -> tuple[dict[str, str], int]:
    """Exact physical file bytes, rejecting links and unusual filesystem members."""
    base = _root(root)
    files: dict[str, str] = {}
    portable_seen: set[str] = set()
    directories: set[str] = set()
    total_bytes = 0
    def deny_walk_error(exc: OSError) -> None:
        raise RebuildDenied("artifact directory traversal failed") from exc

    for current, dirs, members in os.walk(base, followlinks=False,
                                          onerror=deny_walk_error):
        directory = Path(current)
        for name in sorted(dirs):
            child = directory / name
            try:
                info = child.lstat()
            except OSError as exc:
                raise RebuildDenied("artifact directory disappeared") from exc
            _need(stat.S_ISDIR(info.st_mode), "linked or unusual artifact directory")
            relative_directory = child.relative_to(base).as_posix()
            _portable_key(relative_directory, portable_seen)
            directories.add(relative_directory)
        for name in sorted(members):
            child = directory / name
            try:
                info = child.lstat()
            except OSError as exc:
                raise RebuildDenied("artifact member disappeared") from exc
            relative = child.relative_to(base).as_posix()
            _need(relative not in files, "duplicate artifact path")
            _portable_key(relative, portable_seen)
            files[relative] = _hash_regular_file(child, info)
            total_bytes += info.st_size
    _need(bool(files), "empty rebuild tree")
    for directory in directories:
        _need(any(member.startswith(directory + "/") for member in files),
              "unrepresented empty artifact directory")
    return dict(sorted(files.items())), total_bytes



def _physical_tree_pin(root: Path) -> tuple[tuple[str, int, int, int, int, int], ...]:
    """Bounded identity witness; not an atomic filesystem snapshot."""
    pinned: list[tuple[str, int, int, int, int, int]] = []

    def on_error(exc: OSError) -> None:
        raise RebuildDenied("physical tree traversal failed") from exc

    for current, dirs, members in os.walk(root, followlinks=False,
                                          onerror=on_error):
        directory = Path(current)
        entries = [(directory, "d")]
        entries += [(directory / name, "d") for name in sorted(dirs)]
        entries += [(directory / name, "f") for name in sorted(members)]
        for path, kind in entries:
            try:
                info = path.lstat()
            except OSError as exc:
                raise RebuildDenied("physical member unavailable") from exc
            if kind == "d":
                _need(stat.S_ISDIR(info.st_mode), "linked or unusual directory")
            else:
                _need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1,
                      "linked or unusual file")
            pinned.append((kind + ":" + path.relative_to(root).as_posix(),
                           info.st_dev, info.st_ino, info.st_size,
                           info.st_mtime_ns, info.st_ctime_ns))
    return tuple(sorted(pinned))



def compare_rebuilds(first: Path, second: Path, *,
                     expected_inventory_sha256: str) -> dict[str, Any]:
    """Return a non-release byte-equality receipt; deny unpinned equality."""
    _need(type(expected_inventory_sha256) is str
          and HEX256.fullmatch(expected_inventory_sha256) is not None,
          "expected inventory must be independently pinned lowercase SHA-256")
    a = _root(first)
    b = _root(second)
    _need(a != b and a not in b.parents and b not in a.parents,
          "two rebuilds must use separate non-nested directories")
    a_pin = _physical_tree_pin(a)
    b_pin = _physical_tree_pin(b)
    a_members, a_bytes = inventory(a)
    b_members, b_bytes = inventory(b)
    _need(inventory(a) == (a_members, a_bytes)
          and inventory(b) == (b_members, b_bytes),
          "member bytes changed across scans")
    _need(_physical_tree_pin(a) == a_pin
          and _physical_tree_pin(b) == b_pin,
          "physical member identity changed across scans")
    a_hash = digest(canonical(a_members))
    b_hash = digest(canonical(b_members))
    _need(a_members == b_members and a_bytes == b_bytes,
          "physical rebuild member bytes differ")
    _need(a_hash == b_hash == expected_inventory_sha256,
          "physical rebuild inventory differs from independent expected hash")
    core = {
        "schema_version": SCHEMA,
        "expected_inventory_sha256": expected_inventory_sha256,
        "observed_inventory_sha256": a_hash,
        "member_count": len(a_members),
        "total_member_bytes": a_bytes,
        "rebuild_tree_count": 2,
        "byte_identical": True,
        "independent_clean_checkout_verified": False,
        "s16_corpus_admission_verified": False,
        "frozen_32k_tokenizer_identity_verified": False,
        "production_release_authorized": False,
        "plan9_training_authorized": False,
        "terminal_done": False,
    }
    return {**core, "witness_sha256": digest(canonical(core))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        result = compare_rebuilds(args.first, args.second,
                                  expected_inventory_sha256=args.expected_inventory_sha256)
    except RebuildDenied as exc:
        parser.exit(2, f"S17 rebuild witness DENIED: {exc}\n")
    print(canonical(result).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
