"""EVAL-290 immutable external-real Ukrainian selection-validation materializer.

The selector is reserved before future training. It consumes only exact Wave-1
rights-admitted Ukrainian source objects that have no observed training exposure
at the EVAL-290 cutoff, excludes every EVAL-233 final-test record/family, and
emits a deterministic selection-validation set for model/tokenizer/hyperparameter
selection. It is never a final-test or tokenizer-fit/training corpus.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

WORKER_ID = "EVAL-290-UA-SELECTION-VALIDATION-V1"
SOURCE_SCHEMA = "12-6.eval290-ua-selection-sources.v1"
RIGHTS_SCHEMA = "12-6.eval290-ua-selection-rights.v1"
RESERVATION_SCHEMA = "12-6.eval290-ua-selection-reservation.v1"
SET_SCHEMA = "12-6.eval290-ua-selection-validation-set.v1"
MANIFEST_SCHEMA = "12-6.eval290-ua-selection-validation-manifest.v1"

SOURCE_CONFIG = Path("configs/evaluation/eval290_ua_selection_sources_v1.json")
RIGHTS_CONFIG = Path("configs/evaluation/eval290_ua_selection_rights_v1.json")
RESERVATION_CONFIG = Path("configs/evaluation/eval290_ua_selection_reservation_v1.json")
COMMITTED_DATA = Path("data/evaluation/eval290_ua_selection_validation_v1.jsonl")
COMMITTED_MANIFEST = Path("configs/evaluation/eval290_ua_selection_validation_manifest_v1.json")

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()\-.]{8,}\d)(?!\d)")
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
HUGO_RE = re.compile(r"{{[<%].*?[>%]}}", re.S)
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
MARKDOWN_PUNCT_RE = re.compile(r"^[#>*+\-`~]+\s*|[*_`~]", re.M)


class Eval290Error(RuntimeError):
    pass


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Eval290Error(f"unable to read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise Eval290Error(f"expected JSON object: {path}")
    return value


def _load_contracts(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    sources = _read_json(repo_root / SOURCE_CONFIG)
    rights = _read_json(repo_root / RIGHTS_CONFIG)
    reservation = _read_json(repo_root / RESERVATION_CONFIG)
    if sources.get("schema_version") != SOURCE_SCHEMA:
        raise Eval290Error("source config schema drift")
    if rights.get("schema_version") != RIGHTS_SCHEMA:
        raise Eval290Error("rights config schema drift")
    if reservation.get("schema_version") != RESERVATION_SCHEMA:
        raise Eval290Error("reservation config schema drift")
    if sources.get("worker_id") != WORKER_ID or rights.get("worker_id") != WORKER_ID:
        raise Eval290Error("worker identity drift")
    if reservation.get("worker_id") != WORKER_ID:
        raise Eval290Error("reservation worker identity drift")
    commit = reservation.get("reservation_commit_sha")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Eval290Error("reservation_commit_sha is not bound to an immutable Git commit")
    if reservation.get("status") != "RESERVED_BY_DETERMINISTIC_SELECTOR_BEFORE_FUTURE_TRAINING":
        raise Eval290Error("selection reservation is not active")
    return sources, rights, reservation


def _http_get(url: str, *, limit: int = 256 << 20) -> bytes:
    req = Request(url, headers={"User-Agent": "12-6-eval290/1.0", "Accept-Encoding": "identity"})
    with urlopen(req, timeout=180) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > limit:
            raise Eval290Error(f"source exceeds acquisition limit: {url}")
        payload = response.read(limit + 1)
    if len(payload) > limit:
        raise Eval290Error(f"source exceeds acquisition limit: {url}")
    return payload


def _normalize_lines(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.replace("\r\n", "\n").replace("\r", "\n"))
    lines = [" ".join(line.split()) for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _normalize_markdown_visible(payload: bytes) -> str:
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise Eval290Error("Kubernetes source is not strict UTF-8") from exc
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end >= 0:
            text = text[end + 5 :]
    text = HTML_COMMENT_RE.sub(" ", text)
    text = HUGO_RE.sub(" ", text)
    text = IMAGE_RE.sub(lambda m: m.group(1), text)
    text = LINK_RE.sub(lambda m: m.group(1), text)
    text = MARKDOWN_PUNCT_RE.sub("", text)
    return _normalize_lines(text)[:50000]


def _normalize_generic(text: str) -> str:
    return _normalize_lines(text)


def _split_text(text: str, target_chars: int, min_chars: int, max_chars: int) -> list[str]:
    units: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            units.append(line)
            continue
        start = 0
        while start < len(line):
            units.append(line[start : start + target_chars])
            start += target_chars
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        add = len(unit) + (1 if current else 0)
        if current and length + add > target_chars:
            chunk = "\n".join(current).strip()
            if chunk:
                chunks.append(chunk)
            current = []
            length = 0
        current.append(unit)
        length += len(unit) + (1 if length else 0)
    if current:
        chunks.append("\n".join(current).strip())
    normalized: list[str] = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            for start in range(0, len(chunk), target_chars):
                piece = chunk[start : start + target_chars].strip()
                if piece:
                    normalized.append(piece)
        else:
            normalized.append(chunk)
    return [chunk for chunk in normalized if min_chars <= len(chunk) <= max_chars]


def _quality_ok(text: str, selector: dict[str, Any]) -> bool:
    if EMAIL_RE.search(text) or PHONE_RE.search(text):
        return False
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return False
    alpha = [c for c in chars if c.isalpha()]
    if len(alpha) / len(chars) < float(selector["min_alpha_ratio"]):
        return False
    cyrillic = [c for c in alpha if "CYRILLIC" in unicodedata.name(c, "")]
    if not alpha or len(cyrillic) / len(alpha) < float(selector["min_cyrillic_alpha_ratio"]):
        return False
    return True


def _candidate(*, source: dict[str, Any], locator: str, text: str, selector: dict[str, Any]) -> dict[str, Any] | None:
    text = text.strip()
    if not _quality_ok(text, selector):
        return None
    content_sha = sha256_bytes(text.encode("utf-8"))
    rank = sha256_bytes((str(selector["seed"]) + "\0" + str(source["source_id"]) + "\0" + locator + "\0" + content_sha).encode("utf-8"))
    record_id = "eval290-ua-" + sha256_bytes((str(source["source_id"]) + "\0" + locator + "\0" + content_sha).encode("utf-8"))[:24]
    return {"record_id": record_id, "candidate_locator": locator, "selector_rank_sha256": rank, "content_sha256": content_sha, "utf8_bytes": len(text.encode("utf-8")), "text": text}


def _kubernetes_candidates(source: dict[str, Any], payload: bytes, selector: dict[str, Any]) -> list[dict[str, Any]]:
    text = _normalize_markdown_visible(payload)
    expected = source.get("normalized_sha256")
    got = sha256_bytes(text.encode("utf-8"))
    if got != expected:
        raise Eval290Error(f"{source['source_id']}: DATA-228 normalization identity drift: {got}")
    chunks = _split_text(text, int(selector["target_chars"]), int(selector["min_chars"]), int(selector["max_chars"]))
    out = []
    for index, text_chunk in enumerate(chunks):
        row = _candidate(source=source, locator=f"normalized-chunk:{index}", text=text_chunk, selector=selector)
        if row is not None:
            out.append(row)
    return out


def _perestoroha_candidates(source: dict[str, Any], payload: bytes, selector: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise Eval290Error("pyarrow is required for the pinned Perestoroha Parquet object") from exc
    field = str(source.get("field", "transcription"))
    table = pq.read_table(pa.BufferReader(payload), columns=[field])
    values = table[field].to_pylist()
    out: list[dict[str, Any]] = []
    for row_index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            continue
        text = _normalize_generic(value)
        chunks = _split_text(text, int(selector["target_chars"]), int(selector["min_chars"]), int(selector["max_chars"]))
        for chunk_index, text_chunk in enumerate(chunks):
            row = _candidate(source=source, locator=f"parquet-row:{row_index}:chunk:{chunk_index}", text=text_chunk, selector=selector)
            if row is not None:
                out.append(row)
    return out


def _final_test_identity(repo_root: Path, source_config: dict[str, Any]) -> tuple[set[str], set[str]]:
    authority = source_config["final_test_authority"]
    seed_path = repo_root / str(authority["seed_path"])
    payload = seed_path.read_bytes()
    if sha256_bytes(payload) != authority["seed_sha256"]:
        raise Eval290Error("EVAL-233 final-test seed identity drift")
    try:
        data = gzip.decompress(payload)
    except (OSError, EOFError) as exc:
        raise Eval290Error("unable to decompress EVAL-233 final-test seed") from exc
    hashes: set[str] = set()
    families: set[str] = set()
    for raw in data.splitlines():
        row = json.loads(raw)
        text = row.get("text")
        if not isinstance(text, str) or not text:
            raise Eval290Error("invalid final-test text row")
        hashes.add(sha256_bytes(text.encode("utf-8")))
        families.add(str(row.get("source_family", "")))
    if len(hashes) != int(authority["documents"]):
        raise Eval290Error("EVAL-233 final-test record count/hash uniqueness drift")
    return hashes, families


def _source_rights(source: dict[str, Any], rights: dict[str, Any]) -> dict[str, Any]:
    decision = rights.get("sources", {}).get(source["source_id"])
    if not isinstance(decision, dict):
        raise Eval290Error(f"missing selection rights decision for {source['source_id']}")
    if decision.get("evaluation_status") != "APPROVED_FOR_SELECTION_VALIDATION":
        raise Eval290Error(f"selection-validation rights not approved for {source['source_id']}")
    if decision.get("raw_sha256") != source.get("raw_sha256"):
        raise Eval290Error(f"rights/source raw identity mismatch for {source['source_id']}")
    if decision.get("uses", {}).get("selection_validation") != "ALLOWED":
        raise Eval290Error(f"selection-validation use not explicitly allowed for {source['source_id']}")
    return decision


def _enrich_record(source: dict[str, Any], candidate: dict[str, Any], rights: dict[str, Any], reservation: dict[str, Any]) -> dict[str, Any]:
    decision = _source_rights(source, rights)
    return {
        "record_id": candidate["record_id"], "modality": "ua", "language": "uk", "source_kind": "EXTERNAL_REAL",
        "source_id": source["source_id"], "source_family": source["source_family"], "source_version": source["source_version"],
        "source_raw_sha256": source["raw_sha256"], "source_raw_bytes": source.get("runtime_raw_bytes", source.get("raw_bytes")),
        "candidate_locator": candidate["candidate_locator"], "selector_rank_sha256": candidate["selector_rank_sha256"],
        "content_sha256": candidate["content_sha256"], "utf8_bytes": candidate["utf8_bytes"], "license_id": decision["license_id"],
        "rights_authority_ref": str(RIGHTS_CONFIG), "reservation_authority_ref": str(RESERVATION_CONFIG),
        "reservation_commit_sha": reservation["reservation_commit_sha"], "purpose": "selection-validation",
        "selection_eligible": True, "model_selection_eligible": True, "tokenizer_selection_eligible": True,
        "hyperparameter_selection_eligible": True, "tokenizer_fit_eligible": False, "training_eligible": False,
        "future_training_prohibited": True, "final_test_eligible": False, "final_reporting_eligible": False,
        "text": candidate["text"],
    }


def _effective_family_count(bytes_by_family: dict[str, int]) -> tuple[float, float]:
    total = sum(bytes_by_family.values())
    if total <= 0:
        return 0.0, 0.0
    shares = [value / total for value in bytes_by_family.values()]
    entropy = -sum(p * math.log(p) for p in shares if p > 0)
    return entropy, math.exp(entropy)


def _validate_diversity(rows: list[dict[str, Any]], gate: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(row["source_family"]) for row in rows)
    bytes_by_family: Counter[str] = Counter()
    for row in rows:
        bytes_by_family[str(row["source_family"])] += int(row["utf8_bytes"])
    total = sum(bytes_by_family.values())
    shares = {k: v / total for k, v in sorted(bytes_by_family.items())}
    entropy, effective = _effective_family_count(dict(bytes_by_family))
    if len(counts) < int(gate["minimum_independent_source_families"]):
        raise Eval290Error("source-family diversity gate failed")
    if min(counts.values()) < int(gate["minimum_records_per_family"]):
        raise Eval290Error("minimum records per family gate failed")
    if max(shares.values()) > float(gate["maximum_family_byte_share"]):
        raise Eval290Error("top-family byte-share gate failed")
    if effective < float(gate["minimum_effective_family_count"]):
        raise Eval290Error("effective-family-count gate failed")
    return {"independent_source_families": len(counts), "records_by_family": dict(sorted(counts.items())), "bytes_by_family": dict(sorted(bytes_by_family.items())), "family_byte_shares": shares, "top_family_byte_share": max(shares.values()), "entropy_nats": entropy, "effective_family_count": effective}


def _serialize_jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def build(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    source_config, rights, reservation = _load_contracts(repo_root)
    selector = source_config["selector"]
    gate = source_config["diversity_gate"]
    final_hashes, final_families = _final_test_identity(repo_root, source_config)
    source_families = {str(source["source_family"]) for source in source_config["sources"]}
    if source_families & final_families:
        raise Eval290Error("selection source family overlaps existing final-test family")
    if len(source_families) != len(source_config["sources"]):
        raise Eval290Error("selection sources are not independent at family level")
    selected: list[dict[str, Any]] = []
    source_bindings: list[dict[str, Any]] = []
    for source in sorted(source_config["sources"], key=lambda item: item["source_family"]):
        _source_rights(source, rights)
        exposure = source.get("pre_reservation_training_exposure", {})
        if exposure.get("status") != "NO_EXECUTED_TRAINING_EVIDENCE_OBSERVED_AT_CUTOFF":
            raise Eval290Error(f"training exposure not cleared for {source['source_id']}")
        payload = _http_get(str(source["acquisition_url"]))
        raw_sha = sha256_bytes(payload)
        if raw_sha != source["raw_sha256"]:
            raise Eval290Error(f"raw source SHA drift for {source['source_id']}: {raw_sha}")
        if source.get("raw_bytes") is not None and len(payload) != int(source["raw_bytes"]):
            raise Eval290Error(f"raw source size drift for {source['source_id']}")
        runtime_source = dict(source)
        runtime_source["runtime_raw_bytes"] = len(payload)
        if source["adapter"] == "markdown_visible_v1":
            candidates = _kubernetes_candidates(runtime_source, payload, selector)
        elif source["adapter"] == "parquet_transcription":
            candidates = _perestoroha_candidates(runtime_source, payload, selector)
        else:
            raise Eval290Error(f"unsupported source adapter: {source['adapter']}")
        candidates = [row for row in candidates if row["content_sha256"] not in final_hashes]
        candidates.sort(key=lambda row: (row["selector_rank_sha256"], row["record_id"]))
        need = int(selector["records_per_family"])
        if len(candidates) < need:
            raise Eval290Error(f"{source['source_id']}: only {len(candidates)} eligible candidates, need {need}")
        chosen = candidates[:need]
        selected.extend(_enrich_record(runtime_source, row, rights, reservation) for row in chosen)
        source_bindings.append({"source_id": source["source_id"], "source_family": source["source_family"], "source_version": source["source_version"], "raw_sha256": raw_sha, "raw_bytes": len(payload), "candidate_count_after_quality_and_final_test_exclusion": len(candidates), "selected_records": len(chosen), "wave1_admission": source["wave1_admission"], "pre_reservation_training_exposure": exposure})
    selected.sort(key=lambda row: (row["source_family"], row["selector_rank_sha256"], row["record_id"]))
    content_hashes = [str(row["content_sha256"]) for row in selected]
    record_ids = [str(row["record_id"]) for row in selected]
    if len(set(content_hashes)) != len(content_hashes):
        raise Eval290Error("duplicate content selected")
    if len(set(record_ids)) != len(record_ids):
        raise Eval290Error("duplicate record_id selected")
    overlap = sorted(set(content_hashes) & final_hashes)
    if overlap:
        raise Eval290Error(f"selection/final-test exact content overlap: {overlap}")
    diversity = _validate_diversity(selected, gate)
    data_bytes = _serialize_jsonl(selected)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / COMMITTED_DATA.name
    data_path.write_bytes(data_bytes)
    record_bindings = [{"record_id": row["record_id"], "source_id": row["source_id"], "source_family": row["source_family"], "candidate_locator": row["candidate_locator"], "content_sha256": row["content_sha256"], "utf8_bytes": row["utf8_bytes"], "selector_rank_sha256": row["selector_rank_sha256"]} for row in selected]
    unsigned = {
        "schema_version": MANIFEST_SCHEMA, "set_schema_version": SET_SCHEMA, "worker_id": WORKER_ID,
        "status": "IMMUTABLE_RESERVED_UA_SELECTION_VALIDATION", "purpose": "selection-validation", "language": "uk", "modality": "ua", "documents": len(selected),
        "data_file": {"path": str(COMMITTED_DATA), "bytes": len(data_bytes), "sha256": sha256_bytes(data_bytes)},
        "records": record_bindings, "source_bindings": source_bindings,
        "rights_config": {"path": str(RIGHTS_CONFIG), "sha256": sha256_bytes((repo_root / RIGHTS_CONFIG).read_bytes()), "all_sources_selection_validation_approved": True},
        "source_config": {"path": str(SOURCE_CONFIG), "sha256": sha256_bytes((repo_root / SOURCE_CONFIG).read_bytes())},
        "reservation": {"path": str(RESERVATION_CONFIG), "sha256": sha256_bytes((repo_root / RESERVATION_CONFIG).read_bytes()), "reservation_commit_sha": reservation["reservation_commit_sha"], "future_training_prohibited": True, "tokenizer_fit_prohibited": True},
        "final_test_disjointness": {"eval233_set_identity_sha256": source_config["final_test_authority"]["set_identity_sha256"], "eval233_seed_sha256": source_config["final_test_authority"]["seed_sha256"], "exact_content_hash_overlap": [], "source_family_overlap": [], "final_test_records_copied": False},
        "diversity": diversity,
        "eligibility": {"model_selection": True, "tokenizer_selection": True, "hyperparameter_selection": True, "tokenizer_fit": False, "training": False, "final_test": False, "final_reporting": False},
        "deterministic_rebuild": {"selector": selector, "selector_contract_sha256": hash_json(selector), "canonical_jsonl": "UTF-8; one canonical JSON object per line; LF terminator"},
        "local_free_only": True,
    }
    manifest = dict(unsigned)
    manifest["set_identity_sha256"] = hash_json(unsigned)
    manifest_path = output_dir / COMMITTED_MANIFEST.name
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(repo_root: Path, output_dir: Path, *, compare_committed: bool = False) -> dict[str, Any]:
    manifest_path = output_dir / COMMITTED_MANIFEST.name
    data_path = output_dir / COMMITTED_DATA.name
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise Eval290Error("manifest schema drift")
    if manifest.get("status") != "IMMUTABLE_RESERVED_UA_SELECTION_VALIDATION":
        raise Eval290Error("selection-validation manifest is not immutable/reserved")
    unsigned = dict(manifest)
    claimed = unsigned.pop("set_identity_sha256", None)
    if claimed != hash_json(unsigned):
        raise Eval290Error("selection-validation manifest self-hash mismatch")
    data = data_path.read_bytes()
    if manifest["data_file"]["sha256"] != sha256_bytes(data):
        raise Eval290Error("selection-validation JSONL hash mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line]
    if len(rows) != manifest["documents"]:
        raise Eval290Error("selection-validation document count mismatch")
    if any(row.get("training_eligible") is not False for row in rows):
        raise Eval290Error("a selection record is marked training eligible")
    if any(row.get("final_reporting_eligible") is not False for row in rows):
        raise Eval290Error("a selection record is marked final-report eligible")
    source_config, _, _ = _load_contracts(repo_root)
    final_hashes, final_families = _final_test_identity(repo_root, source_config)
    if {row["content_sha256"] for row in rows} & final_hashes:
        raise Eval290Error("selection/final-test content overlap")
    if {row["source_family"] for row in rows} & final_families:
        raise Eval290Error("selection/final-test family overlap")
    _validate_diversity(rows, source_config["diversity_gate"])
    if compare_committed:
        committed_data = repo_root / COMMITTED_DATA
        committed_manifest = repo_root / COMMITTED_MANIFEST
        if not committed_data.is_file() or not committed_manifest.is_file():
            raise Eval290Error("committed immutable selection files are missing")
        if committed_data.read_bytes() != data:
            raise Eval290Error("deterministic rebuild differs from committed JSONL")
        if committed_manifest.read_bytes() != manifest_path.read_bytes():
            raise Eval290Error("deterministic rebuild differs from committed manifest")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--repo-root", type=Path, default=Path("."))
    build_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--repo-root", type=Path, default=Path("."))
    verify_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser.add_argument("--compare-committed", action="store_true")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.command == "build":
        result = build(repo_root, output_dir)
    else:
        result = verify(repo_root, output_dir, compare_committed=args.compare_committed)
    print(json.dumps({"status": result["status"], "documents": result["documents"], "set_identity_sha256": result["set_identity_sha256"], "data_sha256": result["data_file"]["sha256"], "independent_source_families": result["diversity"]["independent_source_families"]}, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
