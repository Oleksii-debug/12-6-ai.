"""Fresh-process two-clean proof for canonical post-pack loss materialization.

The proof reuses :mod:`twelve_six.packing.loss_materialization`; it does not
tokenize, pack, or assign loss spans independently. Raw document text exists
only in the ephemeral input packet and is never copied into the durable proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.tokenization import ByteTokenizer

from .loss_materialization import (
    MATERIALIZATION_SCHEMA,
    LossMaterializationDocument,
    build_postpack_loss_materialization,
)

INPUT_SCHEMA = "12-6.postpack-two-clean-input.v4"
PROOF_SCHEMA = "12-6.postpack-two-clean-proof.v4"
IMPLEMENTATION_MANIFEST_SCHEMA = "12-6.d04-two-clean-implementation-manifest.v1"
RUNTIME_DEPENDENCY_MANIFEST_SCHEMA = "12-6.d04-runtime-dependency-manifest.v1"
_IMPLEMENTATION_PATHS = (
    "twelve_six/__init__.py",
    "twelve_six/packing/__init__.py",
    "twelve_six/packing/core.py",
    "twelve_six/packing/jsonl.py",
    "twelve_six/packing/loss_materialization.py",
    "twelve_six/packing/manifest.py",
    "twelve_six/packing/two_clean_build.py",
    "twelve_six/tokenization/__init__.py",
    "twelve_six/tokenization/base.py",
    "twelve_six/tokenization/byte.py",
)
_INPUT_KEYS = frozenset(
    {
        "schema_version",
        "terminal_corpus_authority_identity_sha256",
        "stage_bindings",
        "expected_tokenizer_identity_sha256",
        "expected_packing_identity_sha256",
        "expected_runtime_identity_sha256",
        "expected_implementation_manifest",
        "expected_implementation_manifest_identity_sha256",
        "expected_runtime_dependency_manifest",
        "expected_runtime_dependency_manifest_identity_sha256",
        "documents",
        "claim_boundary",
        "input_packet_identity_sha256",
    }
)
_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "input_packet_identity_sha256",
        "terminal_corpus_authority_identity_sha256",
        "stage_bindings",
        "tokenizer_identity_sha256",
        "packing_identity_sha256",
        "runtime_identity_sha256",
        "implementation_manifest",
        "implementation_manifest_identity_sha256",
        "runtime_dependency_manifest",
        "runtime_dependency_manifest_identity_sha256",
        "fresh_process_count",
        "byte_identical",
        "build_a_sha256",
        "build_b_sha256",
        "materialization_identity_sha256",
        "claim_boundary",
        "proof_identity_sha256",
    }
)
_REQUIRED_BINDINGS = (
    "normalization",
    "evaluation_reservations",
    "dedup",
    "split",
    "packing",
)
_HEX = frozenset("0123456789abcdef")
_DOCUMENT_KEYS = {
    "document_id",
    "text",
    "source_id",
    "language",
    "modality",
    "family_id",
    "normalized_payload_sha256",
    "source_bytes",
    "split",
    "dedup_cluster_id",
    "retained_after_dedup",
    "evaluation_reserved",
    "reserved_target_ranges",
}
_CLEAN_ENV_KEYS = frozenset(
    {
        "PYTHONPATH",
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
    }
)


class TwoCleanBuildError(ValueError):
    """Raised when independent clean materializations cannot be trusted."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_obj(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TwoCleanBuildError(f"trusted file cannot be hashed: {path.name}") from exc
    return digest.hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in _HEX for character in value)
    ):
        raise TwoCleanBuildError(f"{field} must be exact lowercase SHA-256")
    return value


def _normalize_bindings(value: Mapping[str, str]) -> dict[str, str]:
    if set(value) != set(_REQUIRED_BINDINGS):
        raise TwoCleanBuildError("stage_bindings contain an unexpected or missing stage")
    return {
        name: _require_sha256(value[name], f"stage_bindings.{name}")
        for name in _REQUIRED_BINDINGS
    }


def _normalize_implementation_manifest(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TwoCleanBuildError("implementation manifest must be an object")
    if set(value) != set(_IMPLEMENTATION_PATHS):
        raise TwoCleanBuildError(
            "implementation manifest has an unexpected or missing component"
        )
    return {
        path: _require_sha256(value[path], f"implementation_manifest.{path}")
        for path in _IMPLEMENTATION_PATHS
    }


def _implementation_manifest_identity(value: Mapping[str, str]) -> str:
    normalized = _normalize_implementation_manifest(value)
    return _sha256_obj(
        {
            "schema_version": IMPLEMENTATION_MANIFEST_SCHEMA,
            "components": normalized,
        }
    )


def current_implementation_manifest() -> dict[str, str]:
    """Hash the exact repository source closure used by the two-clean build."""
    source_root = _trusted_source_root()
    manifest: dict[str, str] = {}
    for relative_path in _IMPLEMENTATION_PATHS:
        candidate = source_root / relative_path
        if candidate.is_symlink() or not candidate.is_file():
            raise TwoCleanBuildError(
                f"implementation component is not a regular file: {relative_path}"
            )
        manifest[relative_path] = _sha256_file(candidate)
    return manifest


def _verify_implementation_binding(
    expected_manifest: Mapping[str, str],
    *,
    expected_identity_sha256: str,
) -> str:
    normalized = _normalize_implementation_manifest(expected_manifest)
    expected_identity = _require_sha256(
        expected_identity_sha256,
        "expected_implementation_manifest_identity_sha256",
    )
    if _implementation_manifest_identity(normalized) != expected_identity:
        raise TwoCleanBuildError("implementation manifest identity mismatch")
    observed = current_implementation_manifest()
    for relative_path in _IMPLEMENTATION_PATHS:
        if observed[relative_path] != normalized[relative_path]:
            raise TwoCleanBuildError(
                f"implementation source bytes mismatch: {relative_path}"
            )
    return expected_identity


def _normalize_runtime_dependency_manifest(
    value: Mapping[str, Mapping[str, str]],
) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or not value:
        raise TwoCleanBuildError("runtime dependency manifest must be a non-empty object")
    normalized: dict[str, dict[str, str]] = {}
    for module_name in sorted(value):
        if not isinstance(module_name, str) or not module_name:
            raise TwoCleanBuildError("runtime dependency module name must be non-empty")
        entry = value[module_name]
        if not isinstance(entry, Mapping):
            raise TwoCleanBuildError(
                f"runtime dependency entry must be an object: {module_name}"
            )
        kind = entry.get("kind")
        if kind in {"built-in", "frozen"}:
            if set(entry) != {"kind"}:
                raise TwoCleanBuildError(
                    f"runtime dependency {module_name} has unexpected fields"
                )
            normalized[module_name] = {"kind": kind}
            continue
        if kind != "stdlib" or set(entry) != {"kind", "path", "sha256"}:
            raise TwoCleanBuildError(
                f"runtime dependency {module_name} has unexpected fields"
            )
        relative_path = entry.get("path")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or relative_path.startswith("/")
            or "\\" in relative_path
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise TwoCleanBuildError(
                f"runtime dependency {module_name} has invalid relative path"
            )
        normalized[module_name] = {
            "kind": "stdlib",
            "path": relative_path,
            "sha256": _require_sha256(
                entry.get("sha256"),
                f"runtime_dependency_manifest.{module_name}.sha256",
            ),
        }
    return normalized


def _runtime_dependency_manifest_identity(
    value: Mapping[str, Mapping[str, str]],
) -> str:
    normalized = _normalize_runtime_dependency_manifest(value)
    return _sha256_obj(
        {
            "schema_version": RUNTIME_DEPENDENCY_MANIFEST_SCHEMA,
            "modules": normalized,
        }
    )


def _current_runtime_dependency_manifest() -> dict[str, dict[str, str]]:
    """Describe behavior-bearing clean-runtime modules without absolute paths."""
    source_root = _trusted_source_root()
    stdlib_root_value = sysconfig.get_path("stdlib")
    if not isinstance(stdlib_root_value, str) or not stdlib_root_value:
        raise TwoCleanBuildError("Python stdlib root is unavailable")
    try:
        stdlib_root = Path(stdlib_root_value).resolve(strict=True)
    except OSError as exc:
        raise TwoCleanBuildError("Python stdlib root cannot be resolved") from exc

    manifest: dict[str, dict[str, str]] = {}
    for module_name, module in sorted(sys.modules.items()):
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if origin in {"built-in", "frozen"}:
            manifest[module_name] = {"kind": origin}
            continue
        if not isinstance(origin, str) or not origin:
            continue
        raw_candidate = Path(origin)
        if raw_candidate.is_symlink():
            raise TwoCleanBuildError(
                f"runtime dependency origin is a symlink: {module_name}"
            )
        try:
            candidate = raw_candidate.resolve(strict=True)
        except OSError as exc:
            raise TwoCleanBuildError(
                f"runtime dependency origin cannot be resolved: {module_name}"
            ) from exc
        if candidate.is_relative_to(source_root):
            source_relative = candidate.relative_to(source_root).as_posix()
            if source_relative not in _IMPLEMENTATION_PATHS:
                raise TwoCleanBuildError(
                    "runtime dependency escaped implementation source closure: "
                    f"{module_name}"
                )
            continue
        if not candidate.is_relative_to(stdlib_root):
            raise TwoCleanBuildError(
                f"runtime dependency escaped stdlib/source closure: {module_name}"
            )
        if candidate.is_symlink() or not candidate.is_file():
            raise TwoCleanBuildError(
                f"runtime dependency is not a regular file: {module_name}"
            )
        manifest[module_name] = {
            "kind": "stdlib",
            "path": candidate.relative_to(stdlib_root).as_posix(),
            "sha256": _sha256_file(candidate),
        }
    return _normalize_runtime_dependency_manifest(manifest)


def _verify_runtime_dependency_binding(
    expected_manifest: Mapping[str, Mapping[str, str]],
    *,
    expected_identity_sha256: str,
) -> str:
    normalized = _normalize_runtime_dependency_manifest(expected_manifest)
    expected_identity = _require_sha256(
        expected_identity_sha256,
        "expected_runtime_dependency_manifest_identity_sha256",
    )
    if _runtime_dependency_manifest_identity(normalized) != expected_identity:
        raise TwoCleanBuildError("runtime dependency manifest identity mismatch")
    observed = _current_runtime_dependency_manifest()
    if observed != normalized:
        raise TwoCleanBuildError(
            "runtime dependency manifest does not match clean current runtime"
        )
    return expected_identity


def _trusted_python_executable(requested: str | None = None) -> Path:
    if not sys.executable:
        raise TwoCleanBuildError("python executable is unavailable")
    try:
        trusted = Path(sys.executable).resolve(strict=True)
        candidate = Path(requested or sys.executable).resolve(strict=True)
    except OSError as exc:
        raise TwoCleanBuildError("python executable cannot be resolved") from exc
    try:
        same_runtime = candidate.samefile(trusted)
    except OSError as exc:
        raise TwoCleanBuildError("python executable cannot be compared") from exc
    if not same_runtime:
        raise TwoCleanBuildError(
            "python_executable must resolve to the trusted current runtime"
        )
    return trusted


def _runtime_descriptor(executable: Path | None = None) -> dict[str, Any]:
    trusted = executable or _trusted_python_executable()
    cache_tag = getattr(sys.implementation, "cache_tag", None)
    if not isinstance(cache_tag, str) or not cache_tag:
        raise TwoCleanBuildError("Python runtime cache tag is unavailable")
    return {
        "implementation": sys.implementation.name,
        "version": [
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ],
        "cache_tag": cache_tag,
        "executable_sha256": _sha256_file(trusted),
    }


def current_runtime_identity_sha256() -> str:
    """Return a path-independent identity for the exact trusted Python runtime."""
    return _sha256_obj(_runtime_descriptor())


def _trusted_source_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    expected = source_root / "twelve_six" / "packing" / "two_clean_build.py"
    if not expected.is_file():
        raise TwoCleanBuildError("trusted twelve_six source root cannot be resolved")
    return source_root


def _clean_child_env(source_root: Path, pycache_root: Path) -> dict[str, str]:
    """Build a minimal child environment instead of inheriting caller Python hooks."""
    if pycache_root.exists():
        raise TwoCleanBuildError("child pycache root must start absent")
    env = {
        "PYTHONPATH": str(source_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(pycache_root),
    }
    if set(env) != set(_CLEAN_ENV_KEYS):
        raise AssertionError("clean child environment key drift")
    return env


def probe_clean_runtime_dependency_manifest(
    *, python_executable: str | None = None, timeout_seconds: int = 120
) -> dict[str, dict[str, str]]:
    """Observe clean-child runtime dependencies for external freezing/authorization."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")
    if timeout_seconds <= 0:
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")
    executable = _trusted_python_executable(python_executable)
    source_root = _trusted_source_root()
    with tempfile.TemporaryDirectory(prefix="twelve-six-runtime-probe-") as directory:
        root = Path(directory)
        completed = subprocess.run(
            [
                str(executable),
                "-S",
                "-B",
                "-m",
                "twelve_six.packing.two_clean_build",
                "--runtime-manifest-probe",
            ],
            cwd=root,
            env=_clean_child_env(source_root, root / "pycache"),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[-1000:]
        raise TwoCleanBuildError(f"runtime dependency probe failed: {detail}")
    if completed.stderr:
        raise TwoCleanBuildError("runtime dependency probe emitted stderr")
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise TwoCleanBuildError(
            "runtime dependency probe did not emit canonical JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise TwoCleanBuildError("runtime dependency probe root must be an object")
    normalized = _normalize_runtime_dependency_manifest(value)
    if _canonical_json_bytes(normalized).decode("utf-8") != completed.stdout:
        raise TwoCleanBuildError("runtime dependency probe bytes are not canonical")
    return normalized


def _document_row(document: LossMaterializationDocument) -> dict[str, Any]:
    return {
        "document_id": document.document_id,
        "text": document.text,
        "source_id": document.source_id,
        "language": document.language,
        "modality": document.modality,
        "family_id": document.family_id,
        "normalized_payload_sha256": document.normalized_payload_sha256,
        "source_bytes": document.source_bytes,
        "split": document.split,
        "dedup_cluster_id": document.dedup_cluster_id,
        "retained_after_dedup": document.retained_after_dedup,
        "evaluation_reserved": document.evaluation_reserved,
        "reserved_target_ranges": [list(item) for item in document.reserved_target_ranges],
    }


def make_input_packet(
    documents: Sequence[LossMaterializationDocument],
    *,
    terminal_corpus_authority_identity_sha256: str,
    stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_runtime_identity_sha256: str,
    expected_implementation_manifest: Mapping[str, str],
    expected_runtime_dependency_manifest: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Freeze one externally bound ephemeral build request."""
    if not documents:
        raise TwoCleanBuildError("documents must not be empty")
    ordered = sorted(documents, key=lambda item: item.document_id)
    if len({item.document_id for item in ordered}) != len(ordered):
        raise TwoCleanBuildError("document_id values must be unique")
    implementation_manifest = _normalize_implementation_manifest(
        expected_implementation_manifest
    )
    implementation_identity = _implementation_manifest_identity(
        implementation_manifest
    )
    runtime_dependency_manifest = _normalize_runtime_dependency_manifest(
        expected_runtime_dependency_manifest
    )
    runtime_dependency_identity = _runtime_dependency_manifest_identity(
        runtime_dependency_manifest
    )
    packet: dict[str, Any] = {
        "schema_version": INPUT_SCHEMA,
        "terminal_corpus_authority_identity_sha256": _require_sha256(
            terminal_corpus_authority_identity_sha256,
            "terminal_corpus_authority_identity_sha256",
        ),
        "stage_bindings": _normalize_bindings(stage_bindings),
        "expected_tokenizer_identity_sha256": _require_sha256(
            expected_tokenizer_identity_sha256,
            "expected_tokenizer_identity_sha256",
        ),
        "expected_packing_identity_sha256": _require_sha256(
            expected_packing_identity_sha256,
            "expected_packing_identity_sha256",
        ),
        "expected_runtime_identity_sha256": _require_sha256(
            expected_runtime_identity_sha256,
            "expected_runtime_identity_sha256",
        ),
        "expected_implementation_manifest": implementation_manifest,
        "expected_implementation_manifest_identity_sha256": implementation_identity,
        "expected_runtime_dependency_manifest": runtime_dependency_manifest,
        "expected_runtime_dependency_manifest_identity_sha256": runtime_dependency_identity,
        "documents": [_document_row(item) for item in ordered],
        "claim_boundary": {
            "ephemeral_input_contains_source_text": True,
            "durable_proof_contains_source_text": False,
            "authorizes_training": False,
            "authorizes_paid_compute": False,
        },
    }
    packet["input_packet_identity_sha256"] = _sha256_obj(packet)
    return packet


def _verify_input_packet(
    packet: Mapping[str, Any], *, expected_identity_sha256: str
) -> dict[str, Any]:
    expected = _require_sha256(
        expected_identity_sha256,
        "expected_input_packet_identity_sha256",
    )
    value = dict(packet)
    if set(value) != _INPUT_KEYS:
        raise TwoCleanBuildError("input packet has unexpected or missing fields")
    if value.get("schema_version") != INPUT_SCHEMA:
        raise TwoCleanBuildError("unexpected input packet schema")
    observed = _require_sha256(
        value.get("input_packet_identity_sha256"),
        "input_packet_identity_sha256",
    )
    body = dict(value)
    body.pop("input_packet_identity_sha256", None)
    if _sha256_obj(body) != observed:
        raise TwoCleanBuildError("input packet self-identity mismatch")
    if observed != expected:
        raise TwoCleanBuildError(
            "input packet does not match independently expected identity"
        )
    _require_sha256(
        value.get("terminal_corpus_authority_identity_sha256"),
        "terminal_corpus_authority_identity_sha256",
    )
    bindings = value.get("stage_bindings")
    if not isinstance(bindings, Mapping):
        raise TwoCleanBuildError("stage_bindings must be an object")
    value["stage_bindings"] = _normalize_bindings(bindings)
    for field in (
        "expected_tokenizer_identity_sha256",
        "expected_packing_identity_sha256",
        "expected_runtime_identity_sha256",
    ):
        value[field] = _require_sha256(value.get(field), field)
    implementation_manifest = value.get("expected_implementation_manifest")
    if not isinstance(implementation_manifest, Mapping):
        raise TwoCleanBuildError("expected_implementation_manifest must be an object")
    value["expected_implementation_manifest"] = _normalize_implementation_manifest(
        implementation_manifest
    )
    implementation_identity = _require_sha256(
        value.get("expected_implementation_manifest_identity_sha256"),
        "expected_implementation_manifest_identity_sha256",
    )
    if (
        _implementation_manifest_identity(value["expected_implementation_manifest"])
        != implementation_identity
    ):
        raise TwoCleanBuildError("implementation manifest identity mismatch")
    runtime_dependencies = value.get("expected_runtime_dependency_manifest")
    if not isinstance(runtime_dependencies, Mapping):
        raise TwoCleanBuildError(
            "expected_runtime_dependency_manifest must be an object"
        )
    value["expected_runtime_dependency_manifest"] = (
        _normalize_runtime_dependency_manifest(runtime_dependencies)
    )
    runtime_dependency_identity = _require_sha256(
        value.get("expected_runtime_dependency_manifest_identity_sha256"),
        "expected_runtime_dependency_manifest_identity_sha256",
    )
    if (
        _runtime_dependency_manifest_identity(
            value["expected_runtime_dependency_manifest"]
        )
        != runtime_dependency_identity
    ):
        raise TwoCleanBuildError("runtime dependency manifest identity mismatch")
    boundary = value.get("claim_boundary")
    if boundary != {
        "ephemeral_input_contains_source_text": True,
        "durable_proof_contains_source_text": False,
        "authorizes_training": False,
        "authorizes_paid_compute": False,
    }:
        raise TwoCleanBuildError("input claim boundary drift")
    rows = value.get("documents")
    if not isinstance(rows, list) or not rows:
        raise TwoCleanBuildError("documents must be a non-empty list")
    return value


def _document_from_row(row: Mapping[str, Any]) -> LossMaterializationDocument:
    if set(row) != _DOCUMENT_KEYS:
        raise TwoCleanBuildError("document row has unexpected or missing fields")
    ranges = row["reserved_target_ranges"]
    if not isinstance(ranges, list):
        raise TwoCleanBuildError("reserved_target_ranges must be a list")
    normalized_ranges: list[tuple[int, int]] = []
    for item in ranges:
        if not isinstance(item, list) or len(item) != 2:
            raise TwoCleanBuildError(
                "reserved_target_ranges entries must be two-item lists"
            )
        normalized_ranges.append((item[0], item[1]))
    return LossMaterializationDocument(
        document_id=row["document_id"],
        text=row["text"],
        source_id=row["source_id"],
        language=row["language"],
        modality=row["modality"],
        family_id=row["family_id"],
        normalized_payload_sha256=row["normalized_payload_sha256"],
        source_bytes=row["source_bytes"],
        split=row["split"],
        dedup_cluster_id=row["dedup_cluster_id"],
        retained_after_dedup=row["retained_after_dedup"],
        evaluation_reserved=row["evaluation_reserved"],
        reserved_target_ranges=tuple(normalized_ranges),
    )


def _build_one(packet: Mapping[str, Any]) -> dict[str, Any]:
    rows = packet["documents"]
    documents = tuple(_document_from_row(row) for row in rows)
    return build_postpack_loss_materialization(
        documents,
        ByteTokenizer(),
        terminal_corpus_authority_identity_sha256=(
            packet["terminal_corpus_authority_identity_sha256"]
        ),
        stage_bindings=packet["stage_bindings"],
    )


def _verify_materialization_bytes(
    payload: bytes,
    *,
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TwoCleanBuildError("materialization is not canonical UTF-8 JSON") from exc
    if not isinstance(value, dict) or value.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise TwoCleanBuildError("unexpected post-pack materialization schema")
    observed_identity = _require_sha256(
        value.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    body = dict(value)
    body.pop("materialization_identity_sha256", None)
    if _sha256_obj(body) != observed_identity:
        raise TwoCleanBuildError("post-pack materialization self-identity mismatch")
    if _canonical_json_bytes(value) != payload:
        raise TwoCleanBuildError("post-pack materialization bytes are not canonical")
    expected_corpus = _require_sha256(
        expected_corpus_identity_sha256,
        "expected_corpus_identity_sha256",
    )
    if value.get("terminal_corpus_authority_identity_sha256") != expected_corpus:
        raise TwoCleanBuildError("terminal corpus authority drifted during clean build")
    expected_bindings = _normalize_bindings(expected_stage_bindings)
    if value.get("stage_bindings") != expected_bindings:
        raise TwoCleanBuildError("stage binding drifted during clean build")

    tokenizer = value.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        raise TwoCleanBuildError("materialization tokenizer authority is missing")
    expected_tokenizer = _require_sha256(
        expected_tokenizer_identity_sha256,
        "expected_tokenizer_identity_sha256",
    )
    observed_tokenizer = _require_sha256(
        tokenizer.get("identity_sha256"),
        "materialization.tokenizer.identity_sha256",
    )
    if observed_tokenizer != expected_tokenizer:
        raise TwoCleanBuildError("tokenizer identity drifted during clean build")

    packing = value.get("packing")
    if not isinstance(packing, Mapping):
        raise TwoCleanBuildError("materialization packing authority is missing")
    expected_packing = _require_sha256(
        expected_packing_identity_sha256,
        "expected_packing_identity_sha256",
    )
    observed_packing = _require_sha256(
        packing.get("identity_sha256"),
        "materialization.packing.identity_sha256",
    )
    if observed_packing != expected_packing:
        raise TwoCleanBuildError("packing identity drifted during clean build")
    return value


def compare_clean_build_bytes(
    first: bytes,
    second: bytes,
    *,
    expected_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
) -> dict[str, Any]:
    """Require literal equality of two independently produced canonical outputs."""
    first_value = _verify_materialization_bytes(
        first,
        expected_corpus_identity_sha256=expected_corpus_identity_sha256,
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
    )
    second_value = _verify_materialization_bytes(
        second,
        expected_corpus_identity_sha256=expected_corpus_identity_sha256,
        expected_stage_bindings=expected_stage_bindings,
        expected_tokenizer_identity_sha256=expected_tokenizer_identity_sha256,
        expected_packing_identity_sha256=expected_packing_identity_sha256,
    )
    if first != second:
        raise TwoCleanBuildError("independent post-pack materialization bytes differ")
    if first_value["materialization_identity_sha256"] != second_value[
        "materialization_identity_sha256"
    ]:
        raise TwoCleanBuildError("independent materialization identities differ")
    return first_value


def _verify_runtime_binding(packet: Mapping[str, Any]) -> str:
    expected = _require_sha256(
        packet.get("expected_runtime_identity_sha256"),
        "expected_runtime_identity_sha256",
    )
    observed = current_runtime_identity_sha256()
    if observed != expected:
        raise TwoCleanBuildError("runtime identity does not match trusted current runtime")
    return observed


def _run_child(input_path: Path, output_path: Path, expected_identity: str) -> None:
    packet = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(packet, dict):
        raise TwoCleanBuildError("input packet must be a JSON object")
    verified = _verify_input_packet(
        packet,
        expected_identity_sha256=expected_identity,
    )
    _verify_runtime_binding(verified)
    _verify_implementation_binding(
        verified["expected_implementation_manifest"],
        expected_identity_sha256=(
            verified["expected_implementation_manifest_identity_sha256"]
        ),
    )
    _verify_runtime_dependency_binding(
        verified["expected_runtime_dependency_manifest"],
        expected_identity_sha256=(
            verified["expected_runtime_dependency_manifest_identity_sha256"]
        ),
    )
    materialization = _build_one(verified)
    _verify_runtime_dependency_binding(
        verified["expected_runtime_dependency_manifest"],
        expected_identity_sha256=(
            verified["expected_runtime_dependency_manifest_identity_sha256"]
        ),
    )
    output_path.write_bytes(_canonical_json_bytes(materialization))


def prove_two_clean_build(
    packet: Mapping[str, Any],
    *,
    expected_input_packet_identity_sha256: str,
    expected_implementation_manifest_identity_sha256: str,
    expected_runtime_dependency_manifest_identity_sha256: str,
    python_executable: str | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Run two isolated fresh Python processes and emit a text-free equality proof."""
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, int):
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")
    if timeout_seconds <= 0:
        raise TwoCleanBuildError("timeout_seconds must be a positive integer")

    verified = _verify_input_packet(
        packet,
        expected_identity_sha256=expected_input_packet_identity_sha256,
    )
    runtime_identity = _verify_runtime_binding(verified)
    independent_implementation_identity = _require_sha256(
        expected_implementation_manifest_identity_sha256,
        "expected_implementation_manifest_identity_sha256",
    )
    if (
        verified["expected_implementation_manifest_identity_sha256"]
        != independent_implementation_identity
    ):
        raise TwoCleanBuildError(
            "input packet implementation manifest does not match independently expected identity"
        )
    implementation_identity = _verify_implementation_binding(
        verified["expected_implementation_manifest"],
        expected_identity_sha256=independent_implementation_identity,
    )
    independent_runtime_dependency_identity = _require_sha256(
        expected_runtime_dependency_manifest_identity_sha256,
        "expected_runtime_dependency_manifest_identity_sha256",
    )
    if (
        verified["expected_runtime_dependency_manifest_identity_sha256"]
        != independent_runtime_dependency_identity
    ):
        raise TwoCleanBuildError(
            "input packet runtime dependency manifest does not match "
            "independently expected identity"
        )
    executable = _trusted_python_executable(python_executable)
    packet_identity = verified["input_packet_identity_sha256"]
    corpus_identity = verified["terminal_corpus_authority_identity_sha256"]
    tokenizer_identity = verified["expected_tokenizer_identity_sha256"]
    packing_identity = verified["expected_packing_identity_sha256"]
    bindings = verified["stage_bindings"]
    source_root = _trusted_source_root()

    with tempfile.TemporaryDirectory(prefix="twelve-six-two-clean-") as directory:
        root = Path(directory)
        input_path = root / "input.json"
        input_path.write_bytes(_canonical_json_bytes(verified))
        outputs: list[bytes] = []
        output_hashes: list[str] = []
        for name in ("clean-a", "clean-b"):
            work = root / name
            work.mkdir()
            output_path = work / "postpack.json"
            command = [
                str(executable),
                "-S",
                "-B",
                "-m",
                "twelve_six.packing.two_clean_build",
                "--child",
                str(input_path),
                str(output_path),
                "--expected-input-identity",
                packet_identity,
            ]
            completed = subprocess.run(
                command,
                cwd=work,
                env=_clean_child_env(source_root, work / "pycache"),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout).strip()[-1000:]
                raise TwoCleanBuildError(f"{name} fresh process failed: {detail}")
            if completed.stdout or completed.stderr:
                raise TwoCleanBuildError(
                    f"{name} fresh process emitted unexpected output"
                )
            if not output_path.is_file():
                raise TwoCleanBuildError(f"{name} did not produce postpack.json")
            payload = output_path.read_bytes()
            outputs.append(payload)
            output_hashes.append(_sha256_bytes(payload))

    materialization = compare_clean_build_bytes(
        outputs[0],
        outputs[1],
        expected_corpus_identity_sha256=corpus_identity,
        expected_stage_bindings=bindings,
        expected_tokenizer_identity_sha256=tokenizer_identity,
        expected_packing_identity_sha256=packing_identity,
    )
    proof: dict[str, Any] = {
        "schema_version": PROOF_SCHEMA,
        "input_packet_identity_sha256": packet_identity,
        "terminal_corpus_authority_identity_sha256": corpus_identity,
        "stage_bindings": dict(bindings),
        "tokenizer_identity_sha256": tokenizer_identity,
        "packing_identity_sha256": packing_identity,
        "runtime_identity_sha256": runtime_identity,
        "implementation_manifest": dict(
            verified["expected_implementation_manifest"]
        ),
        "implementation_manifest_identity_sha256": implementation_identity,
        "runtime_dependency_manifest": dict(
            verified["expected_runtime_dependency_manifest"]
        ),
        "runtime_dependency_manifest_identity_sha256": (
            independent_runtime_dependency_identity
        ),
        "fresh_process_count": 2,
        "byte_identical": True,
        "build_a_sha256": output_hashes[0],
        "build_b_sha256": output_hashes[1],
        "materialization_identity_sha256": materialization[
            "materialization_identity_sha256"
        ],
        "claim_boundary": {
            "contains_source_text": False,
            "authorizes_training": False,
            "authorizes_paid_compute": False,
            "creates_positive_unique_loss_authority": False,
        },
    }
    proof["proof_identity_sha256"] = _sha256_obj(proof)
    return proof


def verify_proof(
    proof: Mapping[str, Any],
    *,
    expected_proof_identity_sha256: str,
    expected_input_packet_identity_sha256: str,
    expected_terminal_corpus_identity_sha256: str,
    expected_stage_bindings: Mapping[str, str],
    expected_tokenizer_identity_sha256: str,
    expected_packing_identity_sha256: str,
    expected_runtime_identity_sha256: str,
    expected_implementation_manifest: Mapping[str, str],
    expected_implementation_manifest_identity_sha256: str,
    expected_runtime_dependency_manifest: Mapping[str, Mapping[str, str]],
    expected_runtime_dependency_manifest_identity_sha256: str,
) -> dict[str, Any]:
    """Independently verify the durable V4 proof without trusting producer prose."""
    value = dict(proof)
    if set(value) != _PROOF_KEYS:
        raise TwoCleanBuildError("two-clean proof has unexpected or missing fields")
    if value.get("schema_version") != PROOF_SCHEMA:
        raise TwoCleanBuildError("unexpected two-clean proof schema")
    observed = _require_sha256(
        value.get("proof_identity_sha256"),
        "proof_identity_sha256",
    )
    body = dict(value)
    body.pop("proof_identity_sha256", None)
    if _sha256_obj(body) != observed:
        raise TwoCleanBuildError("two-clean proof self-identity mismatch")
    expected_proof = _require_sha256(
        expected_proof_identity_sha256,
        "expected_proof_identity_sha256",
    )
    if observed != expected_proof:
        raise TwoCleanBuildError("two-clean proof does not match expected identity")

    exact_fields = {
        "input_packet_identity_sha256": expected_input_packet_identity_sha256,
        "terminal_corpus_authority_identity_sha256": (
            expected_terminal_corpus_identity_sha256
        ),
        "tokenizer_identity_sha256": expected_tokenizer_identity_sha256,
        "packing_identity_sha256": expected_packing_identity_sha256,
        "runtime_identity_sha256": expected_runtime_identity_sha256,
    }
    for field, expected_value in exact_fields.items():
        expected = _require_sha256(expected_value, f"expected_{field}")
        if value.get(field) != expected:
            raise TwoCleanBuildError(f"two-clean proof {field} mismatch")
    expected_bindings = _normalize_bindings(expected_stage_bindings)
    if value.get("stage_bindings") != expected_bindings:
        raise TwoCleanBuildError("two-clean proof stage binding mismatch")
    expected_manifest = _normalize_implementation_manifest(
        expected_implementation_manifest
    )
    expected_manifest_identity = _require_sha256(
        expected_implementation_manifest_identity_sha256,
        "expected_implementation_manifest_identity_sha256",
    )
    if _implementation_manifest_identity(expected_manifest) != expected_manifest_identity:
        raise TwoCleanBuildError("expected implementation manifest identity mismatch")
    if value.get("implementation_manifest") != expected_manifest:
        raise TwoCleanBuildError("two-clean proof implementation manifest mismatch")
    if value.get("implementation_manifest_identity_sha256") != expected_manifest_identity:
        raise TwoCleanBuildError(
            "two-clean proof implementation manifest identity mismatch"
        )
    expected_runtime_dependencies = _normalize_runtime_dependency_manifest(
        expected_runtime_dependency_manifest
    )
    expected_runtime_dependency_identity = _require_sha256(
        expected_runtime_dependency_manifest_identity_sha256,
        "expected_runtime_dependency_manifest_identity_sha256",
    )
    if (
        _runtime_dependency_manifest_identity(expected_runtime_dependencies)
        != expected_runtime_dependency_identity
    ):
        raise TwoCleanBuildError(
            "expected runtime dependency manifest identity mismatch"
        )
    if value.get("runtime_dependency_manifest") != expected_runtime_dependencies:
        raise TwoCleanBuildError("two-clean proof runtime dependency manifest mismatch")
    if (
        value.get("runtime_dependency_manifest_identity_sha256")
        != expected_runtime_dependency_identity
    ):
        raise TwoCleanBuildError(
            "two-clean proof runtime dependency manifest identity mismatch"
        )
    if value.get("fresh_process_count") != 2:
        raise TwoCleanBuildError("two-clean proof must bind exactly two fresh processes")
    if value.get("byte_identical") is not True:
        raise TwoCleanBuildError("two-clean proof does not establish byte identity")
    build_a = _require_sha256(value.get("build_a_sha256"), "build_a_sha256")
    build_b = _require_sha256(value.get("build_b_sha256"), "build_b_sha256")
    if build_a != build_b:
        raise TwoCleanBuildError("two-clean proof build hashes differ")
    _require_sha256(
        value.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    if value.get("claim_boundary") != {
        "contains_source_text": False,
        "authorizes_training": False,
        "authorizes_paid_compute": False,
        "creates_positive_unique_loss_authority": False,
    }:
        raise TwoCleanBuildError("two-clean proof claim boundary drift")
    return value


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", nargs=2, metavar=("INPUT", "OUTPUT"))
    parser.add_argument("--expected-input-identity")
    parser.add_argument("--runtime-manifest-probe", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.runtime_manifest_probe:
        if args.child is not None or args.expected_input_identity is not None:
            raise TwoCleanBuildError("runtime manifest probe cannot accept child arguments")
        print(_canonical_json_bytes(_current_runtime_dependency_manifest()).decode("utf-8"), end="")
        return 0
    if args.child is None or args.expected_input_identity is None:
        raise TwoCleanBuildError(
            "module CLI is child-only; use prove_two_clean_build() from the trusted parent"
        )
    _run_child(
        Path(args.child[0]),
        Path(args.child[1]),
        args.expected_input_identity,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
