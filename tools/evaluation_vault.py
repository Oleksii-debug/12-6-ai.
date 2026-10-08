"""Plan 4: fail-closed repository-local reserved evaluation boundary.

This is a component-level filesystem/authority boundary, not an OS sandbox:
the evaluator must run under a separate account/process from training for
hostile-code isolation. It never returns reserved labels or metrics before
a verified external terminal release.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan4-evaluation-vault.v1"


class EvaluationBoundaryError(ValueError):
    """Typed, non-disclosing evaluation boundary failure."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise EvaluationBoundaryError("duplicate JSON field")
        result[key] = value
    return result


def _sha(value: object) -> bool:
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _safe_path(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for part in (path, *path.parents):
        if part.is_symlink():
            raise EvaluationBoundaryError("symlink path forbidden")
    return path


def _within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def _regular_private(path: Path) -> bytes:
    path = _safe_path(path)
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
            raise EvaluationBoundaryError("reserved input must be a regular single-link file")
        with os.fdopen(os.dup(fd), "rb") as stream:
            result = stream.read(2_000_001)
        if len(result) > 2_000_000:
            raise EvaluationBoundaryError("reserved input exceeds local fixture limit")
        if os.fstat(fd).st_size != len(result):
            raise EvaluationBoundaryError("reserved input changed during read")
        return result
    finally:
        os.close(fd)


class EvaluationVault:
    """Never supply reserved examples, answers or metrics to a training caller."""

    def __init__(self, root: Path, *, training_roots: tuple[Path, ...]) -> None:
        if type(training_roots) is not tuple or not training_roots:
            raise EvaluationBoundaryError("explicit training roots required")
        self.root = _safe_path(root)
        self.training_roots = tuple(_safe_path(p) for p in training_roots)
        for training_root in self.training_roots:
            if _within(self.root, training_root) or _within(training_root, self.root):
                raise EvaluationBoundaryError("training and evaluation roots overlap")
        if self.root.exists() and not self.root.is_dir():
            raise EvaluationBoundaryError("evaluation root is not a directory")
        self.reserved = self.root / "reserved"
        self.results = self.root / "results"
        if self.root.exists():
            for directory in (self.root, self.reserved, self.results):
                if not directory.is_dir() or directory.is_symlink():
                    raise EvaluationBoundaryError("evaluation directory drift")
                if os.name == "posix" and directory.stat().st_mode & 0o077:
                    raise EvaluationBoundaryError("evaluation directory not private")
        else:
            self.root.mkdir(mode=0o700, parents=False)
            self.reserved.mkdir(mode=0o700)
            self.results.mkdir(mode=0o700)

    def reserve(self, *, dataset: bytes, dataset_version: str) -> dict[str, str]:
        """Import reserved labels via evaluator-only entrypoint; never return them."""
        if type(dataset) is not bytes or not dataset or len(dataset) > 2_000_000:
            raise EvaluationBoundaryError("invalid bounded reserved dataset")
        if type(dataset_version) is not str or not dataset_version.strip():
            raise EvaluationBoundaryError("dataset version required")
        try:
            rows = [json.loads(line, object_pairs_hook=_strict_object,
                               parse_constant=lambda _: (_ for _ in ()).throw(
                                   EvaluationBoundaryError("invalid JSON constant")))
                    for line in dataset.splitlines()]
        except EvaluationBoundaryError:
            raise
        except EvaluationBoundaryError:
            raise
        except (ValueError, UnicodeDecodeError) as exc:
            raise EvaluationBoundaryError("invalid reserved dataset") from exc
        if (not rows or any(type(row) is not dict or set(row) != {"id", "answer"}
                            or type(row["id"]) is not str or type(row["answer"]) is not str
                            for row in rows)):
            raise EvaluationBoundaryError("invalid reserved dataset shape")
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)) or not all(ids):
            raise EvaluationBoundaryError("duplicate or blank reserved IDs")
        digest = _digest(dataset)
        path = self.reserved / (digest + ".jsonl")
        try:
            with path.open("xb") as f:
                f.write(dataset)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(path, 0o600)
        except FileExistsError:
            if _digest(_regular_private(path)) != digest:
                raise EvaluationBoundaryError("reserved dataset collision") from None
        manifest = self.reserved / (digest + ".manifest.json")
        metadata = _canonical({"schema_version": SCHEMA, "dataset_sha256": digest,
                               "dataset_version": dataset_version})
        try:
            with manifest.open("xb") as f:
                f.write(metadata)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(manifest, 0o600)
        except FileExistsError:
            if _regular_private(manifest) != metadata:
                raise EvaluationBoundaryError("immutable dataset version mismatch") from None
        return {"schema_version": SCHEMA, "dataset_sha256": digest,
                "dataset_version": dataset_version}

    def evaluate(
        self,
        *,
        dataset_ref: dict[str, str],
        predictions: dict[str, str],
        model_sha256: str,
        config_sha256: str,
        evaluator_sha256: str,
    ) -> dict[str, str]:
        """Sealed scoring. Returned receipt contains no labels, features or scores."""
        if type(dataset_ref) is not dict or set(dataset_ref) != {
            "schema_version", "dataset_sha256", "dataset_version"
        } or dataset_ref["schema_version"] != SCHEMA or not _sha(dataset_ref["dataset_sha256"]):
            raise EvaluationBoundaryError("invalid dataset reference")
        if not all(_sha(x) for x in (model_sha256, config_sha256, evaluator_sha256)):
            raise EvaluationBoundaryError("model/config/evaluator identities required")
        if (type(predictions) is not dict or len(predictions) > 100_000
                or any(type(k) is not str or type(v) is not str
                       or len(k) > 1024 or len(v) > 10_000
                       for k, v in predictions.items())):
            raise EvaluationBoundaryError("invalid predictions")
        try:
            metadata = _regular_private(
                self.reserved / (dataset_ref["dataset_sha256"] + ".manifest.json")
            )
            if json.loads(metadata) != dataset_ref:
                raise EvaluationBoundaryError("dataset version binding mismatch")
            data = _regular_private(self.reserved / (dataset_ref["dataset_sha256"] + ".jsonl"))
            if _digest(data) != dataset_ref["dataset_sha256"]:
                raise EvaluationBoundaryError("reserved digest mismatch")
            rows = [json.loads(line) for line in data.splitlines()]
            if set(predictions) != {r["id"] for r in rows}:
                raise EvaluationBoundaryError("candidate ID mismatch")
            correct = sum(predictions[row["id"]] == row["answer"] for row in rows)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, EvaluationBoundaryError):
                raise
            raise EvaluationBoundaryError("reserved evaluation failed") from None
        identity = {"schema_version": SCHEMA, "dataset_sha256": dataset_ref["dataset_sha256"],
                    "dataset_version": dataset_ref["dataset_version"],
                    "model_sha256": model_sha256, "config_sha256": config_sha256,
                    "evaluator_sha256": evaluator_sha256,
                    "predictions_sha256": _digest(_canonical(predictions))}
        evaluation_id = _digest(_canonical(identity))
        payload = {**identity, "evaluation_id": evaluation_id, "sample_count": len(rows),
                   "correct_count": correct, "accuracy": correct / len(rows)}
        result_path = self.results / (evaluation_id + ".json")
        encoded = _canonical(payload)
        seal_path = self.results / (evaluation_id + ".sha256")
        try:
            with result_path.open("xb") as f:
                f.write(encoded)
                f.flush()
                os.fsync(f.fileno())
            os.chmod(result_path, 0o600)
        except FileExistsError:
            if _regular_private(result_path) != encoded:
                raise EvaluationBoundaryError("immutable result mismatch") from None
        try:
            with seal_path.open("xb") as f:
                f.write((_digest(encoded) + "\\n").encode())
                f.flush()
                os.fsync(f.fileno())
            os.chmod(seal_path, 0o600)
        except FileExistsError:
            if _regular_private(seal_path) != (_digest(encoded) + "\\n").encode():
                raise EvaluationBoundaryError("immutable result seal mismatch") from None
        return {"schema_version": SCHEMA, "evaluation_id": evaluation_id, "state": "SEALED"}

    def terminal_report(
        self,
        *,
        sealed: dict[str, str],
        trusted_terminal_ids: frozenset[str],
    ) -> dict[str, Any]:
        """Release only an independently verified exact terminal evaluation identity."""
        if (type(sealed) is not dict or set(sealed) != {"schema_version", "evaluation_id", "state"}
                or sealed.get("schema_version") != SCHEMA or sealed.get("state") != "SEALED"
                or not _sha(sealed.get("evaluation_id"))):
            raise EvaluationBoundaryError("invalid sealed receipt")
        if type(trusted_terminal_ids) is not frozenset or sealed["evaluation_id"] not in trusted_terminal_ids:
            raise EvaluationBoundaryError("terminal evaluation authority unverified")
        try:
            raw = _regular_private(self.results / (sealed["evaluation_id"] + ".json"))
            expected_seal = _regular_private(
                self.results / (sealed["evaluation_id"] + ".sha256")
            )
            if expected_seal != (_digest(raw) + "\\n").encode():
                raise EvaluationBoundaryError("result digest drift")
            payload = json.loads(raw)
            if payload["evaluation_id"] != sealed["evaluation_id"]:
                raise EvaluationBoundaryError("result identity drift")
            identity = {k: payload[k] for k in (
                "schema_version", "dataset_sha256", "dataset_version", "model_sha256",
                "config_sha256", "evaluator_sha256", "predictions_sha256"
            )}
            if _digest(_canonical(identity)) != sealed["evaluation_id"]:
                raise EvaluationBoundaryError("result binding drift")
        except (OSError, ValueError, KeyError, TypeError):
            raise EvaluationBoundaryError("terminal result verification failed") from None
        return payload
