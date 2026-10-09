"""Plan2 S11 fixture-only tokenizer architecture selection, never a fitted runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from tools.plan2_physical_materialization_v1 import (
    _atomic_write,
    _read_destination,
    _read_source,
)
from twelve_six.tokenization import byte as incumbent

SCHEMA = "12-6.plan2-tokenizer-architecture-evidence.v1"
POLICY_PATH = "configs/data/plan2_tokenizer_candidate_policy_v1.json"
FIXTURE_PATH = "configs/data/plan2_tokenizer_candidate_fixture_v1.json"
POLICY_BLOB = "9e2ed75ed609c259fee0c5f5862cd2449daa2823"
FIXTURE_BLOB = "d33413882096b2b89fd9f4dbc853917efb297afa"
BYTE_BLOB = "ee21cc40ba07d6e3bc82b20a8642658284f74959"
STRATA = {"uk", "en", "code", "mixed", "boundary", "adversarial"}
IDS = ("canonical-byte-baseline-v1", "byte-bpe-32k-candidate-v1")


class CandidateError(ValueError):
    """Fail closed on untrusted policy, corpus, runtime drift or output."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CandidateError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob(raw: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        _require(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _bad_constant(_: str) -> None:
    raise CandidateError("nonfinite JSON")


def _pinned_json(raw: bytes, expected: str) -> dict[str, Any]:
    _require(type(raw) is bytes and _git_blob(raw) == expected,
             "unapproved candidate policy or fixture Git blob")
    try:
        obj = json.loads(raw.decode("utf-8", "strict"),
                         object_pairs_hook=_pairs, parse_constant=_bad_constant)
    except (ValueError, TypeError, UnicodeError) as exc:
        raise CandidateError("invalid candidate JSON") from exc
    _require(type(obj) is dict, "candidate JSON object required")
    return obj


def _policy(raw: bytes) -> list[dict[str, Any]]:
    p = _pinned_json(raw, POLICY_BLOB)
    _require(set(p) == {"schema_version", "revision", "purpose", "candidates"}
             and p["schema_version"] == "12-6.plan2-tokenizer-architecture-policy.v1"
             and p["purpose"] == "LOCAL_FREE_ARCHITECTURE_PROBE_NO_FIT"
             and type(p["revision"]) is str and bool(p["revision"].strip()),
             "invalid tokenizer architecture policy")
    rows = p["candidates"]
    _require(type(rows) is list and len(rows) == 2, "two independent architectures needed")
    for i, row in enumerate(rows):
        _require(type(row) is dict and set(row) == {
            "candidate_id", "algorithm", "normalization", "vocab_target",
            "special_token_names", "probe_pieces",
        }, "invalid candidate schema")
        _require(row["candidate_id"] == IDS[i] and row["normalization"] == "none",
                 "candidate ID or normalization drift")
        _require(type(row["vocab_target"]) is int and
                 row["vocab_target"] == (256 if i == 0 else 32768),
                 "invalid vocabulary target")
        _require(type(row["special_token_names"]) is list
                 and all(type(v) is str and v for v in row["special_token_names"])
                 and len(set(row["special_token_names"])) == len(row["special_token_names"]),
                 "invalid prospective special names")
        pieces = row["probe_pieces"]
        _require(type(pieces) is list and
                 all(type(v) is str and v for v in pieces)
                 and len(set(pieces)) == len(pieces), "invalid merge probes")
        if i == 0:
            _require(row["algorithm"] == "utf8_byte"
                     and not row["special_token_names"] and not pieces,
                     "canonical baseline modified")
        else:
            _require(row["algorithm"] == "byte_bpe" and bool(pieces)
                     and row["special_token_names"] ==
                     ["<pad>", "<bos>", "<eos>", "<unk>"],
                     "future BPE candidate modified")
    return rows


def _fixture(raw: bytes) -> list[dict[str, str]]:
    p = _pinned_json(raw, FIXTURE_BLOB)
    _require(set(p) == {"schema_version", "purpose", "samples"}
             and p["schema_version"] == "12-6.plan2-tokenizer-architecture-fixture.v1"
             and p["purpose"] == "LOCAL_FREE_HANDWRITTEN_TEXTS_NO_CORPUS",
             "fixture authority invalid")
    rows = p["samples"]
    _require(type(rows) is list and len(rows) == 6, "fixture sample count invalid")
    labels = set()
    ids = set()
    for row in rows:
        _require(type(row) is dict and set(row) == {"sample_id", "stratum", "text"},
                 "fixture sample keys invalid")
        label, rid, text = row["stratum"], row["sample_id"], row["text"]
        _require(type(label) is str and label in STRATA and label not in labels,
                 "fixture stratum duplicated or missing")
        _require(type(rid) is str and re.fullmatch(r"[a-z][a-z0-9_]{2,39}", rid)
                 is not None and rid not in ids, "fixture identity invalid")
        _require(type(text) is str and bool(text), "fixture text empty")
        text.encode("utf-8", "strict")
        labels.add(label)
        ids.add(rid)
    _require(labels == STRATA, "multilingual/code/robustness coverage missing")
    return rows


def _chunks(raw: bytes, candidate: dict[str, Any]) -> list[bytes]:
    if candidate["algorithm"] == "utf8_byte":
        return [bytes((value,)) for value in raw]
    merges = sorted((part.encode("utf-8") for part in candidate["probe_pieces"]),
                    key=lambda part: (-len(part), part))
    chunks: list[bytes] = []
    i = 0
    while i < len(raw):
        found = next((piece for piece in merges if raw.startswith(piece, i)), None)
        piece = found if found is not None else raw[i:i + 1]
        chunks.append(piece)
        i += len(piece)
    return chunks


def inspect(policy_raw: bytes, fixture_raw: bytes) -> dict[str, Any]:
    policy, samples = _policy(policy_raw), _fixture(fixture_raw)
    _require(_git_blob(Path(incumbent.__file__).read_bytes()) == BYTE_BLOB,
             "canonical byte implementation identity drift")
    baseline = incumbent.ByteTokenizer()
    comparisons: list[dict[str, Any]] = []
    for candidate in policy:
        total_bytes, total_units = 0, 0
        metrics = []
        for row in samples:
            text = row["text"]
            raw = text.encode("utf-8", "strict")
            chunks = _chunks(raw, candidate)
            _require(b"".join(chunks) == raw and raw.decode("utf-8") == text,
                     "UTF8/code/combining boundary corruption")
            if candidate["algorithm"] == "utf8_byte":
                _require(len(chunks) == len(baseline.encode(text)),
                         "candidate baseline differs from canonical byte tokenizer")
            units = len(chunks)
            _require(0 < units <= len(raw), "invalid proxy compression count")
            total_bytes += len(raw)
            total_units += units
            metrics.append({"sample_id": row["sample_id"], "stratum": row["stratum"],
                            "text_sha256": _sha(raw), "utf8_bytes": len(raw),
                            "fixture_proxy_units": units, "roundtrip_exact": True})
        comparisons.append({
            "candidate_id": candidate["candidate_id"],
            "architecture_sha256": _sha(_canonical(candidate)),
            "algorithm": candidate["algorithm"], "normalization": candidate["normalization"],
            "vocab_target": candidate["vocab_target"],
            "prospective_special_names_without_token_ids":
                candidate["special_token_names"],
            "fitted_vocab_sha256": None, "tokenizer_fit_executed": False,
            "total_utf8_bytes": total_bytes, "total_fixture_proxy_units": total_units,
            "proxy_token_saving_ppm": (total_bytes - total_units) * 1_000_000 // total_bytes,
            "sample_metrics": metrics,
        })
    byte, bpe = comparisons
    _require(byte["total_fixture_proxy_units"] == byte["total_utf8_bytes"],
             "incumbent byte baseline drift")
    selected = (IDS[1] if bpe["total_fixture_proxy_units"] <
                byte["total_fixture_proxy_units"] else IDS[0])
    core = {
        "schema_version": SCHEMA, "policy_git_blob": POLICY_BLOB,
        "fixture_git_blob": FIXTURE_BLOB, "incumbent_byte_git_blob": BYTE_BLOB,
        "canonical_byte_config_sha256": incumbent.BYTE_TOKENIZER_HASH,
        "canonical_byte_vocab_sha256": incumbent.BYTE_VOCAB_HASH,
        "sample_count": len(samples), "candidate_comparisons": comparisons,
        "recommended_architecture_for_future_fit": selected,
        "recommendation_basis": "HANDWRITTEN_FIXTURE_PROXY_NOT_FITTED_COMPRESSION",
        "fitted_tokenizer_runtime_approved": False, "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False, "checkpoint_reuse_authorized": False,
        "real_final_test_material_accessed": False, "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": _sha(_canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    _require(not any(p.is_symlink() for p in (destination, *destination.parents)),
             "symlink candidate destination")
    try:
        report = inspect(_read_source(root, POLICY_PATH),
                         _read_source(root, FIXTURE_PATH))
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "tokenizer-candidate-manifest.json"
        blob = _canonical(report)
        if path.exists() or path.is_symlink():
            _require(_read_destination(path) == blob, "immutable candidate manifest changed")
        else:
            _atomic_write(destination, path, blob)
        _require(_read_destination(path) == blob, "candidate exact readback drift")
        return report
    except (ValueError, TypeError, OSError) as exc:
        raise CandidateError("tokenizer candidate stage denied") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S11 fixture-only tokenizer probe")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.root, args.out_dir)
    print(json.dumps({"status": "PASS_ARCHITECTURE_PROBE_ONLY",
                      "manifest_sha256": result["manifest_sha256"],
                      "recommended_architecture_for_future_fit":
                          result["recommended_architecture_for_future_fit"],
                      "tokenizer_fit_authorized": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
