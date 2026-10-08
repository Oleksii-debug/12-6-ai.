"""Fail-closed reproducible offline environment contracts for 12-6 AI.

A lock binds real wheel archive bytes and a complete wheelhouse for an exact
Python/OS ABI. It never downloads dependencies, trains models, or assumes a
locally calculated digest is independently trusted evidence.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

PLATFORMS = ("linux_x86_64", "win_amd64")
MAX_MANIFEST = 1_000_000
MAX_WHEELS = 512
MAX_MEMBERS = 100_000
MAX_MEMBER_SIZE = 1_500_000_000
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_WHEEL = re.compile(
    r"([A-Za-z0-9_][A-Za-z0-9_.]*)-([A-Za-z0-9][A-Za-z0-9_.+!]*)-"
    r"([A-Za-z0-9_.]+)-([A-Za-z0-9_.]+)-([A-Za-z0-9_.]+)\.whl\Z"
)
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")


class EnvironmentLockError(ValueError):
    """Rejected platform, project, external archive, or change evidence."""


def _unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise EnvironmentLockError("duplicate JSON field")
        result[key] = value
    return result


def _bad_constant(_: str) -> None:
    raise EnvironmentLockError("nonfinite JSON token")


def _strict(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not raw or len(raw) > MAX_MANIFEST:
        raise EnvironmentLockError("lock JSON has invalid byte size")
    try:
        data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
                          parse_constant=_bad_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise EnvironmentLockError("invalid strict JSON lock") from exc
    if type(data) is not dict:
        raise EnvironmentLockError("JSON lock must be an object")
    stack: list[tuple[Any, int]] = [(data, 0)]
    visited = 0
    while stack:
        node, depth = stack.pop()
        visited += 1
        if visited > 16384 or depth > 32:
            raise EnvironmentLockError("JSON lock structural budget exceeded")
        if type(node) is dict:
            stack.extend((v, depth + 1) for v in node.values())
        elif type(node) is list:
            stack.extend((v, depth + 1) for v in node)
        elif type(node) is float:
            raise EnvironmentLockError("floats forbidden in environment lock")
    return data


def _sha(value: object) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheel_fields(filename: str) -> tuple[str, str, str, str, str]:
    if type(filename) is not str or len(filename) > 220 or "/" in filename or "\\" in filename:
        raise EnvironmentLockError("invalid wheel file basename")
    match = _WHEEL.fullmatch(filename)
    if match is None:
        raise EnvironmentLockError("invalid wheel filename")
    return match.groups()


def _compatible(filename: str, platform: str, python: str) -> bool:
    _, _, py, abi, arch = _wheel_fields(filename)
    tag = "cp" + python.replace(".", "")
    if not any(p in ("py3", "py" + python[0], tag) for p in py.split(".")):
        return False
    if not any(a in ("none", "abi3", tag) for a in abi.split(".")):
        return False
    return any(
        a == "any"
        or (platform == "win_amd64" and a == "win_amd64")
        or (platform == "linux_x86_64" and (
            a == "linux_x86_64"
            or re.fullmatch(r"(?:manylinux(?:_2_\d+|2014|2010|1))_x86_64", a)
            or re.fullmatch(r"musllinux_1_\d+_x86_64", a)
        ))
        for a in arch.split(".")
    )


def _canonical(data: dict[str, Any]) -> bytes:
    return (
        json.dumps(data, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def validate_lock(
    raw: bytes, *, platform: str, python: str, expected_project_sha256: str,
) -> dict[str, Any]:
    doc = _strict(raw)
    if set(doc) != {"schema_version", "platform", "python", "project_sha256", "wheels"}:
        raise EnvironmentLockError("environment lock schema drift")
    if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        raise EnvironmentLockError("environment lock version drift")
    if platform not in PLATFORMS or doc["platform"] != platform:
        raise EnvironmentLockError("platform lock identity mismatch")
    if type(python) is not str or not re.fullmatch(r"3\.(?:11|12|13)", python):
        raise EnvironmentLockError("unsupported Python identity")
    if doc["python"] != python:
        raise EnvironmentLockError("Python ABI identity mismatch")
    if not _sha(expected_project_sha256) or doc["project_sha256"] != expected_project_sha256:
        raise EnvironmentLockError("project dependency declaration changed")
    rows = doc["wheels"]
    if type(rows) is not list or not rows or len(rows) > MAX_WHEELS:
        raise EnvironmentLockError("wheel inventory empty or out of bounds")
    names: set[str] = set()
    files: list[str] = []
    for entry in rows:
        if type(entry) is not dict or set(entry) != {
            "name", "version", "filename", "sha256", "size"
        }:
            raise EnvironmentLockError("wheel record schema drift")
        name, version, filename = entry["name"], entry["version"], entry["filename"]
        if type(name) is not str or _NAME.fullmatch(name) is None:
            raise EnvironmentLockError("invalid wheel project identity")
        found_name, found_version, *_ = _wheel_fields(filename)
        if _normalized(name) != _normalized(found_name) or version != found_version:
            raise EnvironmentLockError("wheel name/version does not match filename")
        if _normalized(name) in names:
            raise EnvironmentLockError("duplicate wheel distribution")
        names.add(_normalized(name))
        if not _compatible(filename, platform, python):
            raise EnvironmentLockError("wheel platform/ABI incompatible")
        if not _sha(entry["sha256"]):
            raise EnvironmentLockError("wheel SHA256 missing")
        if type(entry["size"]) is not int or not 0 < entry["size"] <= 2_147_483_648:
            raise EnvironmentLockError("invalid wheel size")
        files.append(filename)
    if files != sorted(files) or len(files) != len(set(files)):
        raise EnvironmentLockError("wheel entries must be unique and sorted")
    if _canonical(doc) != raw:
        raise EnvironmentLockError("noncanonical environment lock JSON")
    return doc


def _trusted_regular(path: Path) -> None:
    candidate = Path(os.path.abspath(path))
    for parent in candidate.parents:
        st = parent.lstat()
        if not stat.S_ISDIR(st.st_mode) or parent.is_symlink():
            raise EnvironmentLockError("wheelhouse parent symlink or non-directory")
    info = candidate.lstat()
    if not stat.S_ISREG(info.st_mode) or candidate.is_symlink():
        raise EnvironmentLockError("wheel must be an ordinary non-symlink file")


def _read_member(wheel: zipfile.ZipFile, name: str) -> bytes:
    member = wheel.getinfo(name)
    if member.file_size > 2_000_000:
        raise EnvironmentLockError("wheel metadata exceeds size budget")
    return wheel.read(member)


def inspect_wheel(path: Path) -> tuple[str, str]:
    """Inspect every archived byte and require a closed, safe SHA256 RECORD."""
    _trusted_regular(path)
    project, version, *_ = _wheel_fields(path.name)
    try:
        with zipfile.ZipFile(path) as wheel:
            entries = wheel.infolist()
            if len(entries) < 3 or len(entries) > MAX_MEMBERS:
                raise EnvironmentLockError("wheel member inventory out of bounds")
            seen: set[str] = set()
            members: dict[str, zipfile.ZipInfo] = {}
            for member in entries:
                name = member.filename
                if (
                    name.startswith("/") or "\\" in name or ":" in name
                    or any(seg in ("", ".", "..") for seg in name.split("/"))
                    or name in seen or member.is_dir() or member.file_size > MAX_MEMBER_SIZE
                ):
                    raise EnvironmentLockError("unsafe or duplicate wheel archive path")
                mode = (member.external_attr >> 16) & 0o170000
                if mode not in (0, stat.S_IFREG):
                    raise EnvironmentLockError("wheel symlink or special archive member")
                seen.add(name)
                members[name] = member
            prefix = f"{project}-{version}.dist-info/"
            metadata_name, record_name, wheel_name = (
                prefix + "METADATA", prefix + "RECORD", prefix + "WHEEL"
            )
            if not all(x in members for x in (metadata_name, record_name, wheel_name)):
                raise EnvironmentLockError("required wheel metadata missing")
            from email.parser import Parser

            fields = Parser().parsestr(_read_member(wheel, metadata_name).decode("utf-8"))
            if (
                _normalized(fields.get("Name", "")) != _normalized(project)
                or fields.get("Version") != version
            ):
                raise EnvironmentLockError("embedded wheel package identity mismatch")
            if "Wheel-Version: 1.0" not in _read_member(wheel, wheel_name).decode("utf-8"):
                raise EnvironmentLockError("unsupported wheel format")
            try:
                rows = list(csv.reader(
                    io.StringIO(_read_member(wheel, record_name).decode("utf-8")),
                    strict=True,
                ))
            except csv.Error as exc:
                raise EnvironmentLockError("malformed wheel RECORD") from exc
            recorded: set[str] = set()
            for fields in rows:
                if len(fields) != 3:
                    raise EnvironmentLockError("invalid RECORD row")
                name, checksum, size = fields
                if name not in members or name in recorded:
                    raise EnvironmentLockError("unknown/duplicate wheel RECORD path")
                recorded.add(name)
                if name == record_name:
                    if checksum or size:
                        raise EnvironmentLockError("self RECORD must be unhashed")
                    continue
                if not checksum.startswith("sha256=") or not size.isdecimal():
                    raise EnvironmentLockError("unhashed wheel member")
                encoded = checksum[7:]
                try:
                    digest = base64.b64decode(
                        encoded + "=" * ((-len(encoded)) % 4),
                        altchars=b"-_", validate=True,
                    )
                except (ValueError, base64.binascii.Error) as exc:
                    raise EnvironmentLockError("malformed RECORD SHA256") from exc
                if len(digest) != 32 or int(size) != members[name].file_size:
                    raise EnvironmentLockError("wheel member metadata digest/size mismatch")
                actual = hashlib.sha256()
                with wheel.open(name) as fh:
                    for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                        actual.update(chunk)
                if actual.digest() != digest:
                    raise EnvironmentLockError("wheel member bytes changed")
            if recorded != set(members):
                raise EnvironmentLockError("unrecorded wheel archive payload")
    except (zipfile.BadZipFile, UnicodeError, KeyError) as exc:
        raise EnvironmentLockError("invalid wheel archive") from exc
    return _normalized(project), version

