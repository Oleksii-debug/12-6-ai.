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
                segments = name[:-1].split("/") if member.is_dir() else name.split("/")
                if (
                    name.startswith("/") or "\\" in name or ":" in name
                    or any(seg in ("", ".", "..") for seg in segments)
                    or name in seen or member.file_size > MAX_MEMBER_SIZE
                ):
                    raise EnvironmentLockError("unsafe or duplicate wheel archive path")
                mode = (member.external_attr >> 16) & 0o170000
                allowed_modes = (
                    (0, stat.S_IFDIR) if member.is_dir() else (0, stat.S_IFREG)
                )
                if mode not in allowed_modes:
                    raise EnvironmentLockError("wheel symlink or special archive member")
                seen.add(name)
                if member.is_dir():
                    if not name.endswith("/") or member.file_size:
                        raise EnvironmentLockError("nonempty or malformed wheel directory")
                    continue
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


def _digest_file(path: Path) -> tuple[str, int]:
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (
        before.st_dev, before.st_ino, before.st_size,
        before.st_mtime_ns, before.st_ctime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size,
        after.st_mtime_ns, after.st_ctime_ns
    ):
        raise EnvironmentLockError("wheel changed while hashing")
    return digest.hexdigest(), after.st_size


def validate_wheelhouse(lock: dict[str, Any], wheelhouse: Path) -> None:
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise EnvironmentLockError("wheelhouse must be a local real directory")
    supplied = {p.name for p in wheelhouse.iterdir()}
    expected = {record["filename"] for record in lock["wheels"]}
    if supplied != expected:
        raise EnvironmentLockError("wheelhouse contains missing or untracked artifacts")
    for item in lock["wheels"]:
        path = wheelhouse / item["filename"]
        _trusted_regular(path)
        digest, size = _digest_file(path)
        if size != item["size"] or digest != item["sha256"]:
            raise EnvironmentLockError("wheel file hash/size mismatch")
        name, version = inspect_wheel(path)
        if name != _normalized(item["name"]) or version != item["version"]:
            raise EnvironmentLockError("wheel embedded identity mismatch")


def _project_dependencies(pyproject: bytes) -> dict[str, tuple[int, ...]]:
    parsed = tomllib.loads(pyproject.decode("utf-8"))
    entries = parsed["project"]["dependencies"]
    direct: dict[str, tuple[int, ...]] = {}
    for requirement in entries:
        match = re.fullmatch(
            r"([A-Za-z0-9_][A-Za-z0-9_.-]*)\s*>=\s*([0-9][A-Za-z0-9.+]*)",
            requirement,
        )
        if match is None:
            raise EnvironmentLockError("unsupported project direct requirement syntax")
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", match.group(2)):
            raise EnvironmentLockError("non-numeric minimum requires qualified resolver")
        direct[_normalized(match.group(1))] = tuple(map(int, match.group(2).split(".")))
    return direct


def _ensure_runtime_dependencies(lock: dict[str, Any], pyproject: bytes) -> None:
    direct = _project_dependencies(pyproject)
    pinned = {entry["name"]: entry["version"] for entry in lock["wheels"]}
    missing = set(direct) - set(pinned)
    if missing:
        raise EnvironmentLockError(f"missing mandatory wheel projects: {sorted(missing)}")
    for name, minimum in direct.items():
        version = pinned[name].split("+", 1)[0]
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", version):
            raise EnvironmentLockError("unqualified pinned dependency version")
        actual = tuple(map(int, version.split(".")))
        count = max(len(minimum), len(actual))
        if (
            actual + (0,) * (count - len(actual))
            < minimum + (0,) * (count - len(minimum))
        ):
            raise EnvironmentLockError(f"pinned {name} version violates project minimum")


def build_lock(
    wheelhouse: Path, *, platform: str, python: str,
    pyproject: bytes, strict_project: bool = True,
) -> bytes:
    """Generate immutable exact artifact lock only from an existing wheelhouse."""
    if platform not in PLATFORMS or not re.fullmatch(r"3\.(?:11|12|13)", python):
        raise EnvironmentLockError("unsupported lock target")
    if not wheelhouse.is_dir() or wheelhouse.is_symlink():
        raise EnvironmentLockError("wheelhouse must exist and be a real directory")
    rows: list[dict[str, Any]] = []
    for path in sorted(wheelhouse.iterdir()):
        if not _compatible(path.name, platform, python):
            raise EnvironmentLockError("wheel cannot target requested platform/Python")
        name, version = inspect_wheel(path)
        sha, size = _digest_file(path)
        rows.append(dict(name=name, version=version, filename=path.name,
                         sha256=sha, size=size))
    raw = _canonical(dict(
        schema_version=1, platform=platform, python=python,
        project_sha256=hashlib.sha256(pyproject).hexdigest(), wheels=rows,
    ))
    parsed = validate_lock(
        raw, platform=platform, python=python,
        expected_project_sha256=hashlib.sha256(pyproject).hexdigest(),
    )
    if strict_project:
        _ensure_runtime_dependencies(parsed, pyproject)
    validate_wheelhouse(parsed, wheelhouse)
    return raw


def verify_or_restore(
    lock_bytes: bytes, *, pyproject: bytes, wheelhouse: Path,
    platform: str, python: str, target: Path, restore: bool = False,
) -> dict[str, Any]:
    """Verify or offline restore, denying overwrite and cross-platform execution."""
    doc = validate_lock(
        lock_bytes, platform=platform, python=python,
        expected_project_sha256=hashlib.sha256(pyproject).hexdigest(),
    )
    _ensure_runtime_dependencies(doc, pyproject)
    validate_wheelhouse(doc, wheelhouse)
    if not restore:
        return {"status": "VERIFIED_WHEELHOUSE", "wheels": len(doc["wheels"])}
    host = "win_amd64" if os.name == "nt" else "linux_x86_64"
    local_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if platform != host or python != local_python:
        raise EnvironmentLockError("cannot execute restore for a different host ABI")
    if target.exists() or target.is_symlink():
        raise EnvironmentLockError("restore destination must not exist")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Do not rename a prepared venv: console entry-point shebangs embed paths.
    target.mkdir(mode=0o700, exist_ok=False)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(target)],
                       check=True, timeout=120)
        executable = target / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        requirements = target / "plan1-install-requirements.txt"
        requirements.write_text(
            "\n".join(
                f'{(wheelhouse / x["filename"]).resolve().as_uri()} '
                f'--hash=sha256:{x["sha256"]}'
                for x in doc["wheels"]
            ) + "\n",
            encoding="utf-8",
        )
        subprocess.run(
            [str(executable), "-m", "pip", "install", "--no-index", "--no-deps",
             "--require-hashes", "--disable-pip-version-check", "-r", str(requirements)],
            check=True, timeout=900, capture_output=True, text=True,
        )
        subprocess.run(
            [str(executable), "-m", "pip", "check", "--disable-pip-version-check"],
            check=True, timeout=120, capture_output=True, text=True,
        )
        receipt = {
            "schema_version": 1, "platform": platform, "python": python,
            "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
            "project_sha256": doc["project_sha256"], "wheels": len(doc["wheels"]),
        }
        (target / "plan1-environment-receipt.json").write_bytes(_canonical(receipt))
        return {"status": "RESTORED_OFFLINE", "receipt": receipt}
    except BaseException:
        shutil.rmtree(target, ignore_errors=True)
        raise


def validate_semantic_transition(
    before: bytes, after: bytes, *, expected_old_sha256: str,
    expected_new_sha256: str, review: bytes | None,
    independently_pinned_review_sha256: str | None,
) -> dict[str, Any]:
    """Reject a dependency update without independently pinned versioned impacts."""
    if (
        not _sha(expected_old_sha256) or not _sha(expected_new_sha256)
        or hashlib.sha256(before).hexdigest() != expected_old_sha256
        or hashlib.sha256(after).hexdigest() != expected_new_sha256
    ):
        raise EnvironmentLockError("independently expected lock identities required")
    if before == after:
        return {"status": "UNCHANGED", "sha256": expected_old_sha256}
    if (
        review is None or independently_pinned_review_sha256 is None
        or not _sha(independently_pinned_review_sha256)
        or hashlib.sha256(review).hexdigest() != independently_pinned_review_sha256
    ):
        raise EnvironmentLockError("dependency change needs independent reviewed evidence")
    approval = _strict(review)
    if set(approval) != {
        "schema_version", "old_lock_sha256", "new_lock_sha256",
        "contract_version", "decision", "impact", "evidence_refs"
    }:
        raise EnvironmentLockError("dependency transition evidence schema drift")
    if (
        type(approval["schema_version"]) is not int or approval["schema_version"] != 1
        or approval["old_lock_sha256"] != expected_old_sha256
        or approval["new_lock_sha256"] != expected_new_sha256
        or approval["decision"] != "REVIEWED_APPROVED"
    ):
        raise EnvironmentLockError("dependency transition approval mismatch")
    if (
        type(approval["contract_version"]) is not str
        or re.fullmatch(r"[A-Z][A-Z0-9_-]{4,80}", approval["contract_version"]) is None
    ):
        raise EnvironmentLockError("explicit versioned contract revision required")
    impact = approval["impact"]
    required = {"numerical", "tokenizer", "data", "checkpoint", "inference"}
    if type(impact) is not dict or set(impact) != required:
        raise EnvironmentLockError("incomplete change semantic impact assessment")
    for name, evidence in impact.items():
        if type(evidence) is not dict or set(evidence) != {
            "status", "evidence_sha256"
        }:
            raise EnvironmentLockError(f"invalid {name} semantic evidence")
        if evidence["status"] not in ("UNCHANGED_VERIFIED", "CHANGED_REQUALIFIED"):
            raise EnvironmentLockError(f"unqualified {name} dependency semantics")
        if not _sha(evidence["evidence_sha256"]):
            raise EnvironmentLockError(f"missing {name} semantic evidence digest")
    refs = approval["evidence_refs"]
    if type(refs) is not list or not refs or len(refs) > 100:
        raise EnvironmentLockError("change evidence references missing")
    if any(
        type(value) is not str
        or not re.fullmatch(r"[A-Za-z0-9_./:#-]{8,200}", value)
        for value in refs
    ):
        raise EnvironmentLockError("invalid change evidence reference")
    if _canonical(approval) != review:
        raise EnvironmentLockError("noncanonical dependency evidence")
    return {
        "status": "REQUALIFIED_CHANGE",
        "contract_version": approval["contract_version"],
        "evidence_sha256": independently_pinned_review_sha256,
    }
