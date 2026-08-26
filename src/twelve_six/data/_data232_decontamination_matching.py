"""Deterministic, hash-only DATA-232 training/evaluation decontamination authority."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence

SCHEMA = "12-6.data232-decontamination-report.v2"
ALGORITHM = "data232-deterministic-overlap-cluster-v2"
NORMALIZATION = "data232-contamination-normalization-v2"
CODE_SKELETON = "data232-code-skeleton-v1"
DEFAULT_THRESHOLDS = {
    "natural_shingle_tokens": 3,
    "natural_near_jaccard": 0.80,
    "natural_fragment_containment": 0.88,
    "natural_fragment_min_tokens": 18,
    "code_shingle_tokens": 7,
    "code_near_jaccard": 0.86,
    "code_fragment_containment": 0.90,
    "code_fragment_min_tokens": 16,
    "code_copy_jaccard": 0.82,
    "code_copy_min_tokens": 16,
}
OUTCOME_PARTS = ("accuracy", "bpb", "loss", "margin", "metric", "outcome", "perplexity", "result", "score")
INVISIBLE = dict.fromkeys(map(ord, "\ufeff\u00ad\u200b\u200c\u200d\u2060"), None)
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
CODE_TOKEN_RE = re.compile(
    r"(?:[A-Za-z_][A-Za-z0-9_]*)|(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?)|"
    r"(?:==|!=|<=|>=|->|=>|::|\+\+|--|&&|\|\||<<|>>|\*\*)|(?:[^\s])"
)
LINE_COMMENT = re.compile(r"(?m)(?://|#).*$")
BLOCK_COMMENT = re.compile(r"(?s)/\*.*?\*/")
STRING = re.compile(
    r"(?s)(?:'''(?:\\.|[^\\])*?'''|\"\"\"(?:\\.|[^\\])*?\"\"\"|"
    r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\")"
)
KEYWORDS = set(
    "and as async await break case catch class const continue def delete do else elif except export false "
    "finally for from function if import in instanceof interface lambda let match new none not null or package "
    "pass raise return select static struct switch throw true try type var when where while with yield".split()
)


class DecontaminationError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_identity(label: str, value: Any) -> str:
    return sha256_bytes(label.encode() + b"\0" + canonical_bytes(value))


def normalize_for_contamination(text: str, modality: str) -> str:
    value = unicodedata.normalize("NFKC", text).translate(INVISIBLE).replace("\r\n", "\n").replace("\r", "\n")
    if modality == "code":
        lines = [re.sub(r"[\t \f\v]+", " ", line).strip() for line in value.split("\n")]
        return "\n".join(lines).strip()
    return re.sub(r"\s+", " ", value.casefold()).strip()


def code_skeleton_tokens(text: str) -> tuple[str, ...]:
    value = unicodedata.normalize("NFKC", text).translate(INVISIBLE)
    value = STRING.sub(" STR ", LINE_COMMENT.sub(" COMMENT ", BLOCK_COMMENT.sub(" COMMENT ", value)))
    out = []
    for token in CODE_TOKEN_RE.findall(value):
        low = token.lower()
        if token in {"COMMENT", "STR"} or low in KEYWORDS:
            out.append(token if token in {"COMMENT", "STR"} else low)
        elif re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            out.append("ID")
        elif re.fullmatch(r"0[xX][0-9A-Fa-f]+|\d+(?:\.\d+)?", token):
            out.append("NUM")
        else:
            out.append(token)
    return tuple(out)


def _shingles(tokens: Sequence[str], width: int) -> frozenset[str]:
    if not tokens:
        return frozenset()
    if len(tokens) < width:
        return frozenset({"\x1f".join(tokens)})
    return frozenset("\x1f".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1))


def _record(value: Mapping[str, Any]) -> dict[str, Any]:
    required = ("record_id", "source_id", "source_family", "modality", "text")
    if any(not isinstance(value.get(key), str) for key in required):
        raise DecontaminationError("record requires string record_id/source_id/source_family/modality/text")
    modality = value["modality"].lower()
    if modality not in {"uk", "ua", "en", "code", "text"}:
        raise DecontaminationError(f"unsupported modality: {modality}")
    lineage = value.get("lineage_family")
    if lineage is not None and not isinstance(lineage, str):
        raise DecontaminationError("lineage_family must be string or null")
    return {**value, "modality": modality}


def _fingerprint(value: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    record = _record(value)
    normalized = normalize_for_contamination(record["text"], record["modality"])
    tokens = tuple(TOKEN_RE.findall(normalized))
    code = record["modality"] == "code"
    width = int(thresholds["code_shingle_tokens"] if code else thresholds["natural_shingle_tokens"])
    skeleton = code_skeleton_tokens(record["text"]) if code else ()
    return {
        "record": record,
        "raw": sha256_bytes(record["text"].encode()),
        "normalized": sha256_bytes(normalized.encode()),
        "tokens": tokens,
        "shingles": _shingles(tokens, width),
        "skeleton": skeleton,
        "skeleton_shingles": _shingles(skeleton, int(thresholds["code_shingle_tokens"])),
    }


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def _containment(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / min(len(a), len(b)) if a and b else 0.0


def _pair(train: dict[str, Any], other: dict[str, Any], other_kind: str, t: Mapping[str, Any]) -> list[dict[str, Any]]:
    matches = []
    if train["raw"] == other["raw"]:
        matches.append(("raw_exact", 1.0))
    if train["normalized"] == other["normalized"] and train["raw"] != other["raw"]:
        matches.append(("normalized_exact", 1.0))
    same_code = train["record"]["modality"] == other["record"]["modality"] == "code"
    near = _jaccard(train["shingles"], other["shingles"])
    near_limit = float(t["code_near_jaccard"] if same_code else t["natural_near_jaccard"])
    if near >= near_limit and train["normalized"] != other["normalized"]:
        matches.append(("near_match", near))
    min_tokens = int(t["code_fragment_min_tokens"] if same_code else t["natural_fragment_min_tokens"])
    frag_limit = float(t["code_fragment_containment"] if same_code else t["natural_fragment_containment"])
    frag = _containment(train["shingles"], other["shingles"])
    if min(len(train["tokens"]), len(other["tokens"])) >= min_tokens and frag >= frag_limit:
        if not any(kind in {"raw_exact", "normalized_exact"} for kind, _ in matches):
            matches.append(("document_fragment", frag))
    if same_code:
        copy = _jaccard(train["skeleton_shingles"], other["skeleton_shingles"])
        if min(len(train["skeleton"]), len(other["skeleton"])) >= int(t["code_copy_min_tokens"]):
            if copy >= float(t["code_copy_jaccard"]):
                matches.append(("code_fork_copy", copy))
    result = []
    cross = train["record"]["source_family"] != other["record"]["source_family"]
    for kind, score in matches:
        result.append(
            {
                "match_type": kind,
                "score": round(score, 12),
                "train_record_id": train["record"]["record_id"],
                f"{other_kind}_record_id": other["record"]["record_id"],
                "train_raw_sha256": train["raw"],
                f"{other_kind}_raw_sha256": other["raw"],
                "train_normalized_sha256": train["normalized"],
                f"{other_kind}_normalized_sha256": other["normalized"],
                "cross_source_family": cross,
            }
        )
    return result


def _blocked_pairs(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> set[tuple[int, int]]:
    index: dict[str, set[int]] = defaultdict(set)
    for j, fp in enumerate(right):
        keys = {"r:" + fp["raw"], "n:" + fp["normalized"]}
        keys |= {"s:" + x for x in fp["shingles"]} | {"c:" + x for x in fp["skeleton_shingles"]}
        for key in keys:
            index[key].add(j)
    pairs = set()
    for i, fp in enumerate(left):
        keys = {"r:" + fp["raw"], "n:" + fp["normalized"]}
        keys |= {"s:" + x for x in fp["shingles"]} | {"c:" + x for x in fp["skeleton_shingles"]}
        for key in keys:
            pairs |= {(i, j) for j in index.get(key, ())}
    return pairs


def _train_pairs(train: Sequence[dict[str, Any]]) -> set[tuple[int, int]]:
    index: dict[str, set[int]] = defaultdict(set)
    pairs = set()
    for i, fp in enumerate(train):
        keys = {"r:" + fp["raw"], "n:" + fp["normalized"]}
        keys |= {"s:" + x for x in fp["shingles"]} | {"c:" + x for x in fp["skeleton_shingles"]}
        candidates = set().union(*(index.get(key, set()) for key in keys)) if keys else set()
        pairs |= {(j, i) for j in candidates}
        for key in keys:
            index[key].add(i)
    return pairs


def _forbidden_key(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            here = f"{path}.{key}"
            if any(part in str(key).lower() for part in OUTCOME_PARTS):
                return here
            found = _forbidden_key(child, here)
            if found:
                return found
    elif isinstance(value, list):
        for i, child in enumerate(value):
            found = _forbidden_key(child, f"{path}[{i}]")
            if found:
                return found
    return None


def validate_authority_metadata(authorities: Mapping[str, Any]) -> None:
    found = _forbidden_key(authorities)
    if found:
        raise DecontaminationError(f"outcome-bearing evaluation metadata is forbidden: {found}")
    rows = authorities.get("authorities")
    if not isinstance(rows, list) or not rows:
        raise DecontaminationError("at least one immutable evaluation authority is required")
    for row in rows:
        for key in ("authority_id", "identity_sha256", "role", "source_sha"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise DecontaminationError(f"authority missing {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", row["identity_sha256"]):
            raise DecontaminationError("authority identity_sha256 must be lowercase SHA-256")
        if not re.fullmatch(r"[0-9a-f]{40}", row["source_sha"]):
            raise DecontaminationError("authority source_sha must be lowercase Git SHA")


def authority_composite_identity(authorities: Mapping[str, Any], roles: set[str]) -> str | None:
    rows = [
        {key: row[key] for key in ("authority_id", "identity_sha256", "role", "source_sha")}
        for row in authorities["authorities"]
        if row["role"] in roles
    ]
    return stable_identity("data232-authority-composite-v1", sorted(rows, key=lambda x: x["authority_id"])) if rows else None


def _thresholds(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(DEFAULT_THRESHOLDS)
    if overrides:
        unknown = set(overrides) - set(result)
        if unknown:
            raise DecontaminationError("unknown thresholds: " + ",".join(sorted(unknown)))
        result.update(overrides)
    return result
