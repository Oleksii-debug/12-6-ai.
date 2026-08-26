from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

THRESHOLD = 0.85
NGRAM = 5
EXPECTED_SELECTION = (
    "httpx-timeouts-doc:b699f15c8df51d20",
    "requests-authentication-doc:96dc99e811e40013",
    "flask-requestchecksum-doc:25bc38b2a43b2205",
)
EXPECTED_FINAL = ("pytest-capture-doc:5de89bc0e3fbd1a3",)
V1_ID_BY_DOCUMENT = {
    "eval291-en-httpx-timeouts-v1": EXPECTED_SELECTION[0],
    "eval291-en-requests-authentication-v1": EXPECTED_SELECTION[1],
}

class AuthorityError(RuntimeError):
    pass

def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)

def _shingles(text: str, n: int = NGRAM) -> set[tuple[str, ...]]:
    tokens = _tokens(text)
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}

def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def assert_no_exposure(candidate_texts: Iterable[str], consumer_texts: Iterable[str], threshold: float = THRESHOLD) -> None:
    consumers = list(consumer_texts)
    for candidate in candidate_texts:
        csha = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        for consumer in consumers:
            if csha == hashlib.sha256(consumer.encode("utf-8")).hexdigest():
                raise AuthorityError("exact evaluation/training exposure")
            if jaccard(candidate, consumer) >= threshold:
                raise AuthorityError("near-copy evaluation/training exposure")

def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def _no_final_plaintext(obj) -> None:
    forbidden = {"text", "content", "payload_text", "outcome", "outcomes"}
    if isinstance(obj, dict):
        bad = forbidden & set(obj)
        if bad:
            raise AuthorityError(f"final-test manifest exposes forbidden fields: {sorted(bad)}")
        for value in obj.values():
            _no_final_plaintext(value)
    elif isinstance(obj, list):
        for value in obj:
            _no_final_plaintext(value)

def materialize_selection_validation(root: Path) -> list[dict]:
    inherited = _jsonl(root / "data/evaluation/eval291/selection-validation/en.jsonl")
    extension = _jsonl(root / "data/evaluation/eval291/v2/selection-validation/extension-en.jsonl")
    rows = inherited + extension
    for row in rows:
        if "membership_id" not in row:
            row["membership_id"] = V1_ID_BY_DOCUMENT.get(row.get("document_id"))
    return rows

def validate(root: Path) -> dict:
    reservation = _json(root / "data/evaluation/eval291/v2/reservation.json")
    if reservation.get("state") != "SEALED":
        raise AuthorityError("reservation is not sealed")
    rsha = reservation.get("reservation_commit_sha")
    if rsha != "d3f7fead8c04cafd535d1e574a7203523b54464d":
        raise AuthorityError("reservation epoch changed")
    config = _json(root / "configs/evaluation/eval291_en_selection_validation_v2.json")
    if config["reservation"]["commit_sha"] != rsha:
        raise AuthorityError("config/reservation mismatch")
    if config["policy"]["training_eligible"] or config["policy"]["tokenizer_fit_eligible"]:
        raise AuthorityError("evaluation material became consumer-eligible")
    manifest = _json(root / "data/evaluation/eval291/v2/membership-manifest.json")
    selection_members, final_members = manifest["selection_validation"], manifest["final_test"]
    if tuple(x["membership_id"] for x in selection_members) != EXPECTED_SELECTION:
        raise AuthorityError("selection membership/order changed")
    if tuple(x["membership_id"] for x in final_members) != EXPECTED_FINAL:
        raise AuthorityError("final membership/order changed")
    if len({x["source_family"] for x in selection_members + final_members}) != 4:
        raise AuthorityError("independent family count changed")
    rows = materialize_selection_validation(root)
    if tuple(row["membership_id"] for row in rows) != EXPECTED_SELECTION:
        raise AuthorityError("materialized selection order changed")
    for row, member in zip(rows, selection_members):
        if row["purpose"] != "selection_validation" or row["language"] != "en":
            raise AuthorityError("invalid selection row role")
        if row["source_family"] != member["source_family"] or row["source_git_blob_sha1"] != member["git_blob_sha1"]:
            raise AuthorityError("source identity mismatch")
        if hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != row["content_sha256"]:
            raise AuthorityError("selection payload hash mismatch")
    if len({r["content_sha256"] for r in rows}) != len(rows):
        raise AuthorityError("exact duplicate selection object")
    for i in range(len(rows)):
        for j in range(i+1, len(rows)):
            if jaccard(rows[i]["text"], rows[j]["text"]) >= THRESHOLD:
                raise AuthorityError("near-copy selection objects")
    final_manifest = _json(root / "data/evaluation/eval291/v2/final-test/manifest.json")
    _no_final_plaintext(final_manifest)
    if final_manifest.get("payload_committed") or final_manifest.get("outcomes_read") or final_manifest.get("outcomes_exposed"):
        raise AuthorityError("final-test firewall violated")
    evidence = _json(root / "evidence/eval291/en-selection-validation-v2-authority.json")
    if evidence.get("status") != "PASS" or evidence.get("reservation_commit_sha") != rsha:
        raise AuthorityError("exposure evidence not terminal")
    scan = evidence["exposure_scan"]
    if scan["source_identity_collisions"] or scan["near_copy_collisions"]:
        raise AuthorityError("training/tokenizer collision recorded")
    tok = scan["tokenizer_authority"]
    if tok["fit_may_start_now"] or tok["selection_validation_ingress_count"] or tok["final_test_ingress_count"]:
        raise AuthorityError("tokenizer ingress firewall not proven")
    for score in scan["materialized_training"]["new_object_max_jaccard"].values():
        if score >= THRESHOLD:
            raise AuthorityError("materialized training near-copy collision")
    return {"selection_records":len(rows),"final_test_records":len(final_members),"families":4,"reservation_commit_sha":rsha}
