"""Deterministic exact content deduplication for D03 training-eligible corpus shards.

The authoritative seen/not-seen primitive remains D03's SQLiteExactDedupIndex.
This module adds deterministic corpus orchestration, provenance sidecars,
metrics, retained identities, and shard-boundary restart checkpoints. Near
deduplication remains the DataTrove MinHash path owned by D03/DATA-12.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .corpus_foundation import CorpusFoundationError, SQLiteExactDedupIndex

POLICY_SCHEMA = "12-6.exact-dedup-policy.v1"
STATE_SCHEMA = "12-6.exact-dedup-state.v1"
METRICS_SCHEMA = "12-6.exact-dedup-metrics.v1"
RUN_SCHEMA = "12-6.exact-dedup-run.v1"
CORPUS_SCHEMA = "12-6.exact-dedup-corpus.v1"
DATA25_MANIFEST_SCHEMA = "12-6.corpus-manifest.v1"


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise CorpusFoundationError(f"{field} must be lowercase SHA-256 hex")
    return value


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CorpusFoundationError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class ExactDedupPolicy:
    """Identity-bearing exact policy over already-normalized corpus records."""

    key_field: str = "content_sha256"
    text_field: str = "text"
    key_semantics: str = "sha256-normalized-utf8-text-v1"
    eligible_split: str = "train"
    winner_selection: str = "canonical-shard-path-then-line-v1"
    output_serialization: str = "canonical-jsonl-utf8-v1"
    alias_provenance: str = "full-record-metadata-minus-text-v1"
    preserve_text_bytes: bool = True
    schema_version: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != POLICY_SCHEMA:
            raise CorpusFoundationError("unsupported exact dedup policy schema")
        for field in (
            "key_field",
            "text_field",
            "key_semantics",
            "eligible_split",
            "winner_selection",
            "output_serialization",
            "alias_provenance",
        ):
            require_text(getattr(self, field), field)
        if self.key_field != "content_sha256":
            raise CorpusFoundationError("D03 exact dedup key must remain content_sha256")
        if self.key_semantics != "sha256-normalized-utf8-text-v1":
            raise CorpusFoundationError("unsupported exact-key semantics")
        if self.winner_selection != "canonical-shard-path-then-line-v1":
            raise CorpusFoundationError("unsupported deterministic winner policy")
        if self.preserve_text_bytes is not True:
            raise CorpusFoundationError("exact dedup must preserve normalized text bytes")

    def manifest(self) -> dict[str, Any]:
        core = asdict(self)
        return {**core, "policy_sha256": sha256_bytes(canonical_json_bytes(core))}


@dataclass(frozen=True)
class CorpusShard:
    logical_path: str
    path: Path
    sha256: str
    documents: int | None = None
    size_bytes: int | None = None

    def __post_init__(self) -> None:
        require_text(self.logical_path, "logical_path")
        require_sha256(self.sha256, "shard.sha256")


@dataclass(frozen=True)
class CorpusInput:
    manifest_path: Path
    corpus_identity_sha256: str
    shards: tuple[CorpusShard, ...]


class ProvenanceExactDedupIndex(SQLiteExactDedupIndex):
    """Provenance tables layered on the incumbent D03 exact fingerprint index."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS exact_winners ("
            "sha256 TEXT PRIMARY KEY, winner_json TEXT NOT NULL) WITHOUT ROWID"
        )
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS exact_aliases ("
            "sequence INTEGER PRIMARY KEY AUTOINCREMENT, sha256 TEXT NOT NULL, "
            "winner_json TEXT NOT NULL, alias_json TEXT NOT NULL, relation TEXT NOT NULL)"
        )

    def claim(
        self, fingerprint: str, provenance: Mapping[str, Any]
    ) -> Mapping[str, Any] | None:
        duplicate = self.seen_or_add(fingerprint)
        encoded = json.dumps(
            dict(provenance), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        if not duplicate:
            self._connection.execute(
                "INSERT INTO exact_winners(sha256, winner_json) VALUES (?, ?)",
                (fingerprint, encoded),
            )
            return None
        row = self._connection.execute(
            "SELECT winner_json FROM exact_winners WHERE sha256 = ?", (fingerprint,)
        ).fetchone()
        if row is None:
            raise CorpusFoundationError("fingerprint exists without winner provenance")
        winner = json.loads(row[0])
        relation = (
            "within_source"
            if winner.get("source_id") == provenance.get("source_id")
            else "cross_source"
        )
        self._connection.execute(
            "INSERT INTO exact_aliases(sha256, winner_json, alias_json, relation) "
            "VALUES (?, ?, ?, ?)",
            (fingerprint, row[0], encoded, relation),
        )
        return winner

    def rollback(self) -> None:
        self._connection.rollback()

    def aliases(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT sha256, winner_json, alias_json, relation "
            "FROM exact_aliases ORDER BY sequence"
        ).fetchall()
        return [
            {
                "content_sha256": fingerprint,
                "winner": json.loads(winner),
                "alias": json.loads(alias),
                "relation": relation,
            }
            for fingerprint, winner, alias, relation in rows
        ]

    def duplicate_groups(self) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        winners = self._connection.execute(
            "SELECT sha256, winner_json FROM exact_winners ORDER BY sha256"
        ).fetchall()
        for fingerprint, winner in winners:
            aliases = self._connection.execute(
                "SELECT alias_json, relation FROM exact_aliases "
                "WHERE sha256 = ? ORDER BY sequence",
                (fingerprint,),
            ).fetchall()
            if not aliases:
                continue
            relations = [row[1] for row in aliases]
            output.append(
                {
                    "content_sha256": fingerprint,
                    "winner": json.loads(winner),
                    "aliases": [json.loads(row[0]) for row in aliases],
                    "group_size": 1 + len(aliases),
                    "cross_source": "cross_source" in relations,
                    "within_source": "within_source" in relations,
                }
            )
        return output


def load_corpus_input(manifest_path: str | Path) -> CorpusInput:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorpusFoundationError("corpus manifest must be an object")
    if payload.get("schema_version") != DATA25_MANIFEST_SCHEMA:
        raise CorpusFoundationError("unsupported corpus manifest schema")
    identity = require_sha256(
        payload.get("corpus_identity_sha256"), "corpus_identity_sha256"
    )
    core = dict(payload)
    core.pop("corpus_identity_sha256", None)
    if sha256_bytes(canonical_json_bytes(core)) != identity:
        raise CorpusFoundationError("corpus manifest identity mismatch")
    raw_shards = payload.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise CorpusFoundationError("corpus manifest has no physical shards")
    shards: list[CorpusShard] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_shards):
        if not isinstance(item, dict):
            raise CorpusFoundationError(f"shards[{index}] must be an object")
        logical = require_text(item.get("path"), f"shards[{index}].path")
        if logical in seen:
            raise CorpusFoundationError(f"duplicate shard path: {logical}")
        seen.add(logical)
        shards.append(
            CorpusShard(
                logical_path=logical,
                path=path.parent / logical,
                sha256=require_sha256(item.get("sha256"), f"shards[{index}].sha256"),
                documents=item.get("documents"),
                size_bytes=item.get("size_bytes"),
            )
        )
    return CorpusInput(
        manifest_path=path,
        corpus_identity_sha256=identity,
        shards=tuple(sorted(shards, key=lambda shard: shard.logical_path)),
    )


def _distribution_bucket() -> dict[str, int]:
    return {"documents": 0, "content_utf8_bytes": 0, "tokens": 0}


def _bump(
    distribution: dict[str, dict[str, int]],
    key: str,
    *,
    content_bytes: int,
    tokens: int,
) -> None:
    bucket = distribution.setdefault(key, _distribution_bucket())
    bucket["documents"] += 1
    bucket["content_utf8_bytes"] += content_bytes
    bucket["tokens"] += tokens


def _merge_distributions(
    target: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, Any]]
) -> None:
    for key, incoming in source.items():
        bucket = target.setdefault(key, _distribution_bucket())
        for metric in bucket:
            bucket[metric] += int(incoming.get(metric, 0))


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CorpusFoundationError(
                    f"{path}:{line_number}: invalid strict-UTF8 JSONL"
                ) from exc
            if not isinstance(value, dict):
                raise CorpusFoundationError(f"{path}:{line_number}: object required")
            yield line_number, value


def _validate_training_record(
    record: Mapping[str, Any], policy: ExactDedupPolicy, where: str
) -> tuple[str, int, int]:
    for field in ("record_id", "source_id", "source_version"):
        require_text(record.get(field), f"{where}.{field}")
    text = record.get(policy.text_field)
    if not isinstance(text, str):
        raise CorpusFoundationError(f"{where}.{policy.text_field} must be string")
    payload = text.encode("utf-8", "strict")
    digest = sha256_bytes(payload)
    declared = require_sha256(record.get(policy.key_field), f"{where}.{policy.key_field}")
    if digest != declared:
        raise CorpusFoundationError(f"{where}: normalized content hash mismatch")
    tokens = record.get("byte_tokens", len(payload))
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        raise CorpusFoundationError(f"{where}.byte_tokens must be non-negative integer")
    if "byte_tokens" in record and tokens != len(payload):
        raise CorpusFoundationError(f"{where}: byte-token/content-byte mismatch")
    return declared, len(payload), tokens


def _provenance(
    record: Mapping[str, Any], *, logical_shard: str, line_number: int, text_field: str
) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if key != text_field}
    result["input_shard"] = logical_shard
    result["input_line"] = line_number
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _checkpoint_path(root: Path, index: int) -> Path:
    return root / "evidence" / "checkpoints" / f"{index:05d}.json"


def _artifact(path: Path, root: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "documents": records,
    }


def _state_core(corpus: CorpusInput, policy: ExactDedupPolicy) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "input_corpus_identity_sha256": corpus.corpus_identity_sha256,
        "dedup_policy_sha256": policy.manifest()["policy_sha256"],
        "input_shards": [
            {"path": shard.logical_path, "sha256": shard.sha256}
            for shard in corpus.shards
        ],
    }


def _load_checkpoints(root: Path, count: int) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for index in range(count):
        path = _checkpoint_path(root, index)
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("input_shard_index") != index:
            raise CorpusFoundationError(f"invalid exact-dedup checkpoint {path}")
        result[index] = value
    return result


def run_exact_dedup(
    *,
    corpus_manifest: str | Path,
    output_dir: str | Path,
    policy: ExactDedupPolicy | None = None,
    resume: bool = False,
    stop_after_input_shards: int | None = None,
) -> dict[str, Any]:
    """Deduplicate all train records in the immutable DATA-25 physical corpus."""

    corpus = load_corpus_input(corpus_manifest)
    policy = policy or ExactDedupPolicy()
    root = Path(output_dir)
    if not resume and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "shards").mkdir(exist_ok=True)
    (root / "evidence" / "checkpoints").mkdir(parents=True, exist_ok=True)

    for shard in corpus.shards:
        if not shard.path.is_file():
            raise CorpusFoundationError(f"missing physical shard: {shard.path}")
        if sha256_file(shard.path) != shard.sha256:
            raise CorpusFoundationError(f"physical shard hash mismatch: {shard.logical_path}")

    state_core = _state_core(corpus, policy)
    state = {
        **state_core,
        "state_sha256": sha256_bytes(canonical_json_bytes(state_core)),
    }
    state_path = root / "evidence" / "state.json"
    if resume:
        if not state_path.exists():
            raise CorpusFoundationError("resume requested without exact-dedup state")
        existing = json.loads(state_path.read_text(encoding="utf-8"))
        if existing != state:
            raise CorpusFoundationError("resume state/input/policy identity mismatch")
    else:
        _write_json(state_path, state)

    completed = _load_checkpoints(root, len(corpus.shards))
    processed_this_call = 0
    with ProvenanceExactDedupIndex(root / "exact_index.sqlite3") as index:
        for shard_index, shard in enumerate(corpus.shards):
            if shard_index in completed:
                continue
            per = {
                "input_documents": 0,
                "input_content_utf8_bytes": 0,
                "input_tokens": 0,
                "output_documents": 0,
                "output_content_utf8_bytes": 0,
                "output_tokens": 0,
                "documents_removed": 0,
                "bytes_removed": 0,
                "tokens_removed": 0,
                "skipped_non_train_documents": 0,
                "distribution_before": {"language": {}, "modality": {}},
                "distribution_after": {"language": {}, "modality": {}},
            }
            final_path = root / "shards" / Path(shard.logical_path).name
            temp_path = final_path.with_suffix(final_path.suffix + ".tmp")
            survivor_count = 0
            try:
                with temp_path.open("wb") as output:
                    for line_number, record in _read_jsonl(shard.path):
                        if str(record.get("split", "")).casefold() != policy.eligible_split:
                            per["skipped_non_train_documents"] += 1
                            continue
                        where = f"{shard.logical_path}:{line_number}"
                        digest, content_bytes, tokens = _validate_training_record(
                            record, policy, where
                        )
                        language = str(
                            record.get("language", record.get("stratum", "unknown"))
                        )
                        modality = str(record.get("modality", "unknown"))
                        per["input_documents"] += 1
                        per["input_content_utf8_bytes"] += content_bytes
                        per["input_tokens"] += tokens
                        _bump(
                            per["distribution_before"]["language"],
                            language,
                            content_bytes=content_bytes,
                            tokens=tokens,
                        )
                        _bump(
                            per["distribution_before"]["modality"],
                            modality,
                            content_bytes=content_bytes,
                            tokens=tokens,
                        )
                        provenance = _provenance(
                            record,
                            logical_shard=shard.logical_path,
                            line_number=line_number,
                            text_field=policy.text_field,
                        )
                        if index.claim(digest, provenance) is not None:
                            per["documents_removed"] += 1
                            per["bytes_removed"] += content_bytes
                            per["tokens_removed"] += tokens
                            continue
                        output.write(canonical_json_bytes(record))
                        survivor_count += 1
                        per["output_documents"] += 1
                        per["output_content_utf8_bytes"] += content_bytes
                        per["output_tokens"] += tokens
                        _bump(
                            per["distribution_after"]["language"],
                            language,
                            content_bytes=content_bytes,
                            tokens=tokens,
                        )
                        _bump(
                            per["distribution_after"]["modality"],
                            modality,
                            content_bytes=content_bytes,
                            tokens=tokens,
                        )
                index.commit()
                artifact = None
                if survivor_count:
                    temp_path.replace(final_path)
                    artifact = _artifact(final_path, root, survivor_count)
                else:
                    temp_path.unlink(missing_ok=True)
                checkpoint = {
                    "schema_version": "12-6.exact-dedup-checkpoint.v1",
                    "input_shard_index": shard_index,
                    "input_shard": {"path": shard.logical_path, "sha256": shard.sha256},
                    "output_shard": artifact,
                    "metrics": per,
                }
                checkpoint["checkpoint_sha256"] = sha256_bytes(
                    canonical_json_bytes(checkpoint)
                )
                _write_json(_checkpoint_path(root, shard_index), checkpoint)
                completed[shard_index] = checkpoint
            except Exception:
                index.rollback()
                temp_path.unlink(missing_ok=True)
                raise

            processed_this_call += 1
            if (
                stop_after_input_shards is not None
                and processed_this_call >= stop_after_input_shards
            ):
                return {
                    "status": "PARTIAL",
                    "completed_input_shards": len(completed),
                    "input_shards": len(corpus.shards),
                    "state_sha256": state["state_sha256"],
                }

        aliases = index.aliases()
        groups = index.duplicate_groups()

    if len(completed) != len(corpus.shards):
        raise CorpusFoundationError("exact dedup finished with incomplete checkpoints")

    before_language: dict[str, dict[str, int]] = {}
    before_modality: dict[str, dict[str, int]] = {}
    after_language: dict[str, dict[str, int]] = {}
    after_modality: dict[str, dict[str, int]] = {}
    totals = {
        "input_documents": 0,
        "input_content_utf8_bytes": 0,
        "input_tokens": 0,
        "output_documents": 0,
        "output_content_utf8_bytes": 0,
        "output_tokens": 0,
        "documents_removed": 0,
        "bytes_removed": 0,
        "tokens_removed": 0,
        "skipped_non_train_documents": 0,
    }
    output_shards: list[dict[str, Any]] = []
    for shard_index in sorted(completed):
        checkpoint = completed[shard_index]
        per = checkpoint["metrics"]
        for key in totals:
            totals[key] += int(per[key])
        _merge_distributions(before_language, per["distribution_before"]["language"])
        _merge_distributions(before_modality, per["distribution_before"]["modality"])
        _merge_distributions(after_language, per["distribution_after"]["language"])
        _merge_distributions(after_modality, per["distribution_after"]["modality"])
        if checkpoint["output_shard"] is not None:
            output_shards.append(checkpoint["output_shard"])

    if totals["output_documents"] + totals["documents_removed"] != totals["input_documents"]:
        raise CorpusFoundationError("document accounting mismatch")
    if (
        totals["output_content_utf8_bytes"] + totals["bytes_removed"]
        != totals["input_content_utf8_bytes"]
    ):
        raise CorpusFoundationError("content-byte accounting mismatch")

    aliases_path = root / "evidence" / "discarded_aliases.jsonl"
    groups_path = root / "evidence" / "duplicate_groups.jsonl"
    aliases_path.write_bytes(b"".join(canonical_json_bytes(row) for row in aliases))
    groups_path.write_bytes(b"".join(canonical_json_bytes(row) for row in groups))

    metrics_core = {
        "schema_version": METRICS_SCHEMA,
        "input_corpus_identity_sha256": corpus.corpus_identity_sha256,
        "dedup_policy_sha256": policy.manifest()["policy_sha256"],
        **totals,
        "exact_duplicate_groups": len(groups),
        "cross_source_duplicate_groups": sum(bool(g["cross_source"]) for g in groups),
        "within_source_duplicate_groups": sum(bool(g["within_source"]) for g in groups),
        "cross_source_duplicate_aliases": sum(
            row["relation"] == "cross_source" for row in aliases
        ),
        "within_source_duplicate_aliases": sum(
            row["relation"] == "within_source" for row in aliases
        ),
        "distribution_before": {
            "language": dict(sorted(before_language.items())),
            "modality": dict(sorted(before_modality.items())),
        },
        "distribution_after": {
            "language": dict(sorted(after_language.items())),
            "modality": dict(sorted(after_modality.items())),
        },
    }
    metrics = {
        **metrics_core,
        "metrics_sha256": sha256_bytes(canonical_json_bytes(metrics_core)),
    }
    metrics_path = root / "evidence" / "metrics.json"
    _write_json(metrics_path, metrics)

    run_core = {
        "schema_version": RUN_SCHEMA,
        "input_corpus_identity_sha256": corpus.corpus_identity_sha256,
        "dedup_policy": policy.manifest(),
        "state_sha256": state["state_sha256"],
        "metrics_sha256": metrics["metrics_sha256"],
        "output_shards": output_shards,
        "evidence": {
            "discarded_aliases_sha256": sha256_file(aliases_path),
            "duplicate_groups_sha256": sha256_file(groups_path),
        },
    }
    run_manifest = {
        **run_core,
        "run_sha256": sha256_bytes(canonical_json_bytes(run_core)),
    }
    _write_json(root / "evidence" / "run_manifest.json", run_manifest)

    corpus_core = {
        "schema_version": CORPUS_SCHEMA,
        "parent_corpus_identity_sha256": corpus.corpus_identity_sha256,
        "dedup_policy_sha256": policy.manifest()["policy_sha256"],
        "run_sha256": run_manifest["run_sha256"],
        "metrics_sha256": metrics["metrics_sha256"],
        "shards": output_shards,
    }
    output_manifest = {
        **corpus_core,
        "corpus_identity_sha256": sha256_bytes(canonical_json_bytes(corpus_core)),
    }
    _write_json(root / "manifest.json", output_manifest)
    return {
        "status": "COMPLETE",
        "corpus_identity_sha256": output_manifest["corpus_identity_sha256"],
        "run_sha256": run_manifest["run_sha256"],
        "metrics": metrics,
        "output_shards": output_shards,
    }


def rebuild_twice_and_assert_identical(
    *,
    corpus_manifest: str | Path,
    output_root: str | Path,
    policy: ExactDedupPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or ExactDedupPolicy()
    root = Path(output_root)
    first = run_exact_dedup(
        corpus_manifest=corpus_manifest,
        output_dir=root / "build-a",
        policy=policy,
    )
    second = run_exact_dedup(
        corpus_manifest=corpus_manifest,
        output_dir=root / "build-b",
        policy=policy,
    )
    if first["corpus_identity_sha256"] != second["corpus_identity_sha256"]:
        raise CorpusFoundationError("exact-dedup clean rebuild changed corpus identity")
    if first["run_sha256"] != second["run_sha256"]:
        raise CorpusFoundationError("exact-dedup clean rebuild changed run identity")
    return {"first": first, "second": second, "identical": True}
