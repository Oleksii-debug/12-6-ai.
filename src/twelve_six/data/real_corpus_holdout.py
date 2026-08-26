"""Immutable first-party real-corpus held-out sets and exclusion proofs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

HOLDOUT_SCHEMA = "12-6.real-corpus-holdout.v1"
EXCLUSION_PROOF_SCHEMA = "12-6.real-corpus-exclusion-proof.v1"
MODALITIES = ("ua", "en", "code")
REQUIRED_SOURCE_KIND = "EXTERNAL_REAL"


class RealCorpusHoldoutError(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_sha256(name: str, value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise RealCorpusHoldoutError(f"{name} must be lowercase 64-hex SHA-256")
    return value


@dataclass(frozen=True)
class RealHeldoutRecord:
    record_id: str
    modality: str
    source_id: str
    source_family: str
    source_version: str
    source_snapshot_sha256: str
    text: str
    source_kind: str
    evaluation_use_authority_ref: str
    provenance_ref: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "RealHeldoutRecord":
        try:
            record = cls(
                record_id=str(value["record_id"]),
                modality=str(value["modality"]),
                source_id=str(value["source_id"]),
                source_family=str(value["source_family"]),
                source_version=str(value["source_version"]),
                source_snapshot_sha256=str(value["source_snapshot_sha256"]),
                text=str(value["text"]),
                source_kind=str(value["source_kind"]),
                evaluation_use_authority_ref=str(value["evaluation_use_authority_ref"]),
                provenance_ref=str(value["provenance_ref"]),
            )
        except KeyError as exc:
            raise RealCorpusHoldoutError(f"held-out record missing field {exc.args[0]}") from exc
        record.validate()
        return record

    def validate(self) -> None:
        for name in (
            "record_id",
            "source_id",
            "source_family",
            "source_version",
            "evaluation_use_authority_ref",
            "provenance_ref",
        ):
            if not getattr(self, name).strip():
                raise RealCorpusHoldoutError(f"{name} must be non-empty")
        if self.modality not in MODALITIES:
            raise RealCorpusHoldoutError(f"unsupported modality: {self.modality}")
        if self.source_kind != REQUIRED_SOURCE_KIND:
            raise RealCorpusHoldoutError(
                "canonical real-corpus holdouts require source_kind=EXTERNAL_REAL"
            )
        require_sha256("source_snapshot_sha256", self.source_snapshot_sha256)
        if not self.text:
            raise RealCorpusHoldoutError("held-out text must be non-empty")
        try:
            encoded = self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RealCorpusHoldoutError("held-out text must be valid UTF-8") from exc
        if not encoded:
            raise RealCorpusHoldoutError("held-out text must contain source bytes")

    def frozen_row(self) -> dict[str, Any]:
        self.validate()
        source_bytes = self.text.encode("utf-8")
        return {
            "record_id": self.record_id,
            "modality": self.modality,
            "source_id": self.source_id,
            "source_family": self.source_family,
            "source_version": self.source_version,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "source_kind": self.source_kind,
            "evaluation_use_authority_ref": self.evaluation_use_authority_ref,
            "provenance_ref": self.provenance_ref,
            "text": self.text,
            "content_sha256": sha256_bytes(source_bytes),
            "source_bytes": len(source_bytes),
        }


def _manifest_without_identity(
    rows: list[dict[str, Any]],
    file_meta: Mapping[str, Mapping[str, Any]],
    *,
    suite_name: str,
    evaluation_corpus_identity_sha256: str,
    benchmark_registry_sha256: str,
    decontamination_reference_bundle_sha256: str,
    decontamination_report_sha256: str,
) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = str(row["source_family"])
        entry = families.setdefault(
            family,
            {"documents": 0, "source_bytes": 0, "modalities": set(), "source_ids": set()},
        )
        entry["documents"] += 1
        entry["source_bytes"] += int(row["source_bytes"])
        entry["modalities"].add(str(row["modality"]))
        entry["source_ids"].add(str(row["source_id"]))
    family_payload = {
        key: {
            "documents": value["documents"],
            "source_bytes": value["source_bytes"],
            "modalities": sorted(value["modalities"]),
            "source_ids": sorted(value["source_ids"]),
        }
        for key, value in sorted(families.items())
    }
    reserved = {
        "record_ids": sorted(str(row["record_id"]) for row in rows),
        "content_sha256": sorted(str(row["content_sha256"]) for row in rows),
        "source_versions": sorted(
            f"{row['source_id']}@{row['source_version']}" for row in rows
        ),
    }
    return {
        "schema_version": HOLDOUT_SCHEMA,
        "suite_name": suite_name,
        "authority": "FIRST_PARTY_REAL_CORPUS_HELDOUT_NOT_EXTERNAL_BENCHMARK",
        "quality_authority": "FIRST_PARTY_HELDOUT_ONLY",
        "modalities": list(MODALITIES),
        "required_source_kind": REQUIRED_SOURCE_KIND,
        "upstream": {
            "evaluation_corpus_identity_sha256": evaluation_corpus_identity_sha256,
            "benchmark_registry_sha256": benchmark_registry_sha256,
            "decontamination_reference_bundle_sha256": decontamination_reference_bundle_sha256,
            "decontamination_report_sha256": decontamination_report_sha256,
        },
        "files": {key: dict(value) for key, value in sorted(file_meta.items())},
        "source_families": family_payload,
        "reserved": reserved,
        "exclusion_registry_sha256": hash_json(reserved),
        "truth_boundary": {
            "project_authored_records_admitted": False,
            "external_benchmark_required": False,
            "evaluation_use_authority_required_per_record": True,
            "holdout_must_be_excluded_before_model_training": True,
            "holdout_must_be_excluded_before_learned_tokenizer_fit": True,
        },
    }


def build_immutable_holdout(
    records: Iterable[RealHeldoutRecord | Mapping[str, Any]],
    output_dir: Path,
    *,
    suite_name: str,
    evaluation_corpus_identity_sha256: str,
    benchmark_registry_sha256: str,
    decontamination_reference_bundle_sha256: str,
    decontamination_report_sha256: str,
) -> dict[str, Any]:
    for name, value in (
        ("evaluation_corpus_identity_sha256", evaluation_corpus_identity_sha256),
        ("benchmark_registry_sha256", benchmark_registry_sha256),
        ("decontamination_reference_bundle_sha256", decontamination_reference_bundle_sha256),
        ("decontamination_report_sha256", decontamination_report_sha256),
    ):
        require_sha256(name, value)
    if not suite_name.strip():
        raise RealCorpusHoldoutError("suite_name must be non-empty")

    parsed = [
        value if isinstance(value, RealHeldoutRecord) else RealHeldoutRecord.from_mapping(value)
        for value in records
    ]
    if not parsed:
        raise RealCorpusHoldoutError("held-out set must not be empty")
    for value in parsed:
        value.validate()

    rows = [value.frozen_row() for value in parsed]
    rows.sort(
        key=lambda row: (
            str(row["modality"]), str(row["source_family"]), str(row["record_id"])
        )
    )
    modalities = {str(row["modality"]) for row in rows}
    if modalities != set(MODALITIES):
        missing = sorted(set(MODALITIES) - modalities)
        raise RealCorpusHoldoutError(f"held-out set missing modalities: {missing}")
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RealCorpusHoldoutError("duplicate held-out record_id")
    contents = [str(row["content_sha256"]) for row in rows]
    if len(contents) != len(set(contents)):
        raise RealCorpusHoldoutError("duplicate exact held-out content")

    rendered: dict[str, bytes] = {}
    file_meta: dict[str, dict[str, Any]] = {}
    for modality in MODALITIES:
        subset = [row for row in rows if row["modality"] == modality]
        blob = b"".join(canonical_json_bytes(row) + b"\n" for row in subset)
        filename = f"{modality}.jsonl"
        rendered[filename] = blob
        file_meta[modality] = {
            "path": filename,
            "sha256": sha256_bytes(blob),
            "documents": len(subset),
            "source_bytes": sum(int(row["source_bytes"]) for row in subset),
        }

    unsigned = _manifest_without_identity(
        rows,
        file_meta,
        suite_name=suite_name,
        evaluation_corpus_identity_sha256=evaluation_corpus_identity_sha256,
        benchmark_registry_sha256=benchmark_registry_sha256,
        decontamination_reference_bundle_sha256=decontamination_reference_bundle_sha256,
        decontamination_report_sha256=decontamination_report_sha256,
    )
    manifest = dict(unsigned)
    manifest["heldout_identity_sha256"] = hash_json(unsigned)
    manifest_blob = json.dumps(
        manifest, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
    ).encode("utf-8") + b"\n"
    rendered["manifest.json"] = manifest_blob

    if output_dir.exists():
        if not output_dir.is_dir():
            raise RealCorpusHoldoutError("held-out destination exists and is not a directory")
        existing = {path.name for path in output_dir.iterdir()}
        if existing != set(rendered):
            raise RealCorpusHoldoutError("immutable held-out destination inventory differs")
        for name, blob in rendered.items():
            if (output_dir / name).read_bytes() != blob:
                raise RealCorpusHoldoutError("immutable held-out destination bytes differ")
        return verify_immutable_holdout(output_dir)

    output_dir.mkdir(parents=True, exist_ok=False)
    for name, blob in rendered.items():
        (output_dir / name).write_bytes(blob)
    return verify_immutable_holdout(output_dir)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RealCorpusHoldoutError(f"unable to read {path}") from exc
    if not isinstance(value, dict):
        raise RealCorpusHoldoutError(f"{path} must contain a JSON object")
    return value


def load_heldout_rows(output_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = verify_immutable_holdout(output_dir)
    rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        path = output_dir / str(manifest["files"][modality]["path"])
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RealCorpusHoldoutError("held-out JSONL rows must be objects")
                rows.append(value)
    return manifest, rows


def verify_immutable_holdout(output_dir: Path) -> dict[str, Any]:
    if not output_dir.is_dir():
        raise RealCorpusHoldoutError("held-out directory is missing")
    expected_inventory = {"manifest.json", *(f"{m}.jsonl" for m in MODALITIES)}
    actual_inventory = {path.name for path in output_dir.iterdir()}
    if actual_inventory != expected_inventory:
        raise RealCorpusHoldoutError("held-out directory inventory mismatch")
    manifest = _load_json_object(output_dir / "manifest.json")
    if manifest.get("schema_version") != HOLDOUT_SCHEMA:
        raise RealCorpusHoldoutError("held-out schema mismatch")
    supplied_identity = str(manifest.get("heldout_identity_sha256", ""))
    require_sha256("heldout_identity_sha256", supplied_identity)
    unsigned = dict(manifest)
    unsigned.pop("heldout_identity_sha256", None)
    if hash_json(unsigned) != supplied_identity:
        raise RealCorpusHoldoutError("held-out manifest identity mismatch")
    if manifest.get("modalities") != list(MODALITIES):
        raise RealCorpusHoldoutError("held-out modality contract mismatch")
    if manifest.get("required_source_kind") != REQUIRED_SOURCE_KIND:
        raise RealCorpusHoldoutError("held-out source-kind contract weakened")
    for field, value in manifest.get("upstream", {}).items():
        if field.endswith("sha256"):
            require_sha256(field, str(value))

    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    reconstructed_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        meta = manifest.get("files", {}).get(modality)
        if not isinstance(meta, dict) or meta.get("path") != f"{modality}.jsonl":
            raise RealCorpusHoldoutError(f"held-out file metadata mismatch for {modality}")
        path = output_dir / f"{modality}.jsonl"
        if sha256_file(path) != meta.get("sha256"):
            raise RealCorpusHoldoutError(f"held-out file hash mismatch for {modality}")
        document_count = 0
        source_bytes = 0
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                if not raw.endswith("\n"):
                    raise RealCorpusHoldoutError("held-out JSONL rows must end with newline")
                row = json.loads(raw)
                if not isinstance(row, dict):
                    raise RealCorpusHoldoutError("held-out JSONL row must be an object")
                if row.get("modality") != modality:
                    raise RealCorpusHoldoutError("held-out row modality/file mismatch")
                if row.get("source_kind") != REQUIRED_SOURCE_KIND:
                    raise RealCorpusHoldoutError("non-real source admitted to held-out set")
                if not str(row.get("evaluation_use_authority_ref", "")).strip():
                    raise RealCorpusHoldoutError("held-out row lacks evaluation-use authority")
                text = row.get("text")
                if not isinstance(text, str) or not text:
                    raise RealCorpusHoldoutError("held-out row text invalid")
                encoded = text.encode("utf-8")
                if row.get("source_bytes") != len(encoded):
                    raise RealCorpusHoldoutError("held-out source-byte count mismatch")
                if row.get("content_sha256") != sha256_bytes(encoded):
                    raise RealCorpusHoldoutError("held-out content hash mismatch")
                require_sha256(
                    "source_snapshot_sha256", str(row.get("source_snapshot_sha256", ""))
                )
                record_id = str(row.get("record_id", ""))
                content_hash = str(row["content_sha256"])
                if not record_id or record_id in seen_ids:
                    raise RealCorpusHoldoutError("duplicate/empty held-out record_id")
                if content_hash in seen_content:
                    raise RealCorpusHoldoutError("duplicate held-out exact content")
                seen_ids.add(record_id)
                seen_content.add(content_hash)
                document_count += 1
                source_bytes += len(encoded)
                reconstructed_rows.append(row)
        if document_count != meta.get("documents") or source_bytes != meta.get("source_bytes"):
            raise RealCorpusHoldoutError(f"held-out aggregate mismatch for {modality}")
        if document_count <= 0:
            raise RealCorpusHoldoutError(f"held-out modality {modality} is empty")

    reserved = manifest.get("reserved")
    if not isinstance(reserved, dict):
        raise RealCorpusHoldoutError("held-out reserved registry missing")
    expected_reserved = {
        "record_ids": sorted(seen_ids),
        "content_sha256": sorted(seen_content),
        "source_versions": sorted(
            f"{row['source_id']}@{row['source_version']}" for row in reconstructed_rows
        ),
    }
    if reserved != expected_reserved:
        raise RealCorpusHoldoutError("held-out reserved registry mismatch")
    if manifest.get("exclusion_registry_sha256") != hash_json(expected_reserved):
        raise RealCorpusHoldoutError("held-out exclusion registry identity mismatch")
    return manifest


def _candidate_overlap(
    candidate_records: Iterable[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, list[str]]:
    reserved = manifest.get("reserved")
    if not isinstance(reserved, Mapping):
        raise RealCorpusHoldoutError("held-out reserved registry missing")
    reserved_ids = set(str(v) for v in reserved.get("record_ids", []))
    reserved_content = set(str(v) for v in reserved.get("content_sha256", []))
    id_overlap: set[str] = set()
    content_overlap: set[str] = set()
    for row in candidate_records:
        record_id = row.get("record_id")
        if record_id is not None and str(record_id) in reserved_ids:
            id_overlap.add(str(record_id))
        content_hash = row.get("content_sha256")
        if content_hash is None and isinstance(row.get("text"), str):
            content_hash = sha256_bytes(str(row["text"]).encode("utf-8"))
        if content_hash is not None and str(content_hash) in reserved_content:
            content_overlap.add(str(content_hash))
    return {
        "record_id_overlap": sorted(id_overlap),
        "content_sha256_overlap": sorted(content_overlap),
    }


def build_exclusion_proof(
    candidate_records: Iterable[Mapping[str, Any]],
    heldout_manifest: Mapping[str, Any],
    *,
    purpose: str,
    candidate_identity_sha256: str,
) -> dict[str, Any]:
    if purpose not in {"TOKENIZER_FIT", "MODEL_TRAINING"}:
        raise RealCorpusHoldoutError("unsupported exclusion-proof purpose")
    require_sha256("candidate_identity_sha256", candidate_identity_sha256)
    heldout_identity = require_sha256(
        "heldout_identity_sha256", str(heldout_manifest.get("heldout_identity_sha256", ""))
    )
    overlap = _candidate_overlap(candidate_records, heldout_manifest)
    if overlap["record_id_overlap"] or overlap["content_sha256_overlap"]:
        raise RealCorpusHoldoutError(f"{purpose} candidate overlaps immutable held-out set")
    value = {
        "schema_version": EXCLUSION_PROOF_SCHEMA,
        "purpose": purpose,
        "status": "PASS_EXCLUDED_BEFORE_USE",
        "candidate_identity_sha256": candidate_identity_sha256,
        "heldout_identity_sha256": heldout_identity,
        "exclusion_registry_sha256": heldout_manifest.get("exclusion_registry_sha256"),
        "record_id_overlap": [],
        "content_sha256_overlap": [],
    }
    value["proof_sha256"] = hash_json(value)
    return value


def build_fixed_tokenizer_no_fit_proof(
    heldout_manifest: Mapping[str, Any], *, tokenizer_identity_sha256: str
) -> dict[str, Any]:
    require_sha256("tokenizer_identity_sha256", tokenizer_identity_sha256)
    heldout_identity = require_sha256(
        "heldout_identity_sha256", str(heldout_manifest.get("heldout_identity_sha256", ""))
    )
    value = {
        "schema_version": EXCLUSION_PROOF_SCHEMA,
        "purpose": "TOKENIZER_FIT",
        "status": "NOT_APPLICABLE_FIXED_TOKENIZER_NO_FIT_CORPUS",
        "candidate_identity_sha256": tokenizer_identity_sha256,
        "heldout_identity_sha256": heldout_identity,
        "exclusion_registry_sha256": heldout_manifest.get("exclusion_registry_sha256"),
        "record_id_overlap": [],
        "content_sha256_overlap": [],
    }
    value["proof_sha256"] = hash_json(value)
    return value


def validate_exclusion_proof(
    proof: Mapping[str, Any], *, heldout_identity_sha256: str, purpose: str
) -> str:
    if proof.get("schema_version") != EXCLUSION_PROOF_SCHEMA:
        raise RealCorpusHoldoutError("exclusion proof schema mismatch")
    if proof.get("purpose") != purpose:
        raise RealCorpusHoldoutError("exclusion proof purpose mismatch")
    if proof.get("heldout_identity_sha256") != heldout_identity_sha256:
        raise RealCorpusHoldoutError("exclusion proof held-out identity mismatch")
    supplied = require_sha256("proof_sha256", str(proof.get("proof_sha256", "")))
    unsigned = dict(proof)
    unsigned.pop("proof_sha256", None)
    if supplied != hash_json(unsigned):
        raise RealCorpusHoldoutError("exclusion proof self-hash mismatch")
    if proof.get("record_id_overlap") != [] or proof.get("content_sha256_overlap") != []:
        raise RealCorpusHoldoutError("exclusion proof contains held-out overlap")
    status = str(proof.get("status", ""))
    allowed = {"PASS_EXCLUDED_BEFORE_USE"}
    if purpose == "TOKENIZER_FIT":
        allowed.add("NOT_APPLICABLE_FIXED_TOKENIZER_NO_FIT_CORPUS")
    if status not in allowed:
        raise RealCorpusHoldoutError("exclusion proof status is not admissible")
    return supplied
