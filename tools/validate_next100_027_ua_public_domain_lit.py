#!/usr/bin/env python3
"""Offline validator for NEXT100-027 Ukrainian public-domain literature authority."""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_027_ua_public_domain_lit_v1.json"

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical(obj: object) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def fail(msg: str) -> None:
    raise SystemExit(f"NEXT100-027 FAIL: {msg}")

def main() -> int:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected_identity = cfg.pop("authority_identity_sha256", None)
    actual_identity = sha256_bytes(canonical(cfg))
    if expected_identity != actual_identity:
        fail(f"authority identity mismatch: {expected_identity} != {actual_identity}")
    cfg["authority_identity_sha256"] = expected_identity

    if cfg["worker_id"] != "NEXT100-027-DATA-UA-PUBLIC-DOMAIN-LIT":
        fail("worker id drift")
    if cfg["verdict"] != "ADMIT" or cfg["execution_profile"] != "LOCAL_FREE":
        fail("verdict/execution profile drift")

    scope = cfg["scope"]
    required_true = ["training_admitted", "redistribution_admitted", "tokenizer_fit_admitted"]
    required_false = ["selection_validation_admitted", "final_test_admitted", "evaluation_admitted", "full_upstream_corpus_admitted"]
    if not all(scope[k] is True for k in required_true):
        fail("training/redistribution/tokenizer rights gate weakened")
    if not all(scope[k] is False for k in required_false):
        fail("purpose firewall weakened")

    if cfg["work"]["underlying_rights"] != "PUBLIC_DOMAIN":
        fail("underlying work is not bound public-domain")
    if cfg["work"]["rights_reasoning"]["public_domain_confirmed"] is not True:
        fail("public-domain proof not terminal")
    digital = cfg["digital_edition"]
    if digital["scan"]["reusable"] is not True:
        fail("scan reuse not admitted")
    if digital["transcription_compilation"]["reusable"] is not True:
        fail("digital compilation reuse not admitted")
    if digital["transcription_compilation"]["license"][:9] != "CC-BY-4.0":
        fail("digital-layer license drift")
    if digital["transcription_compilation"]["attribution_required"] is not True:
        fail("CC BY attribution obligation lost")

    family = cfg["family"]["source_family"]
    if family != "ua.verba.public-domain.nomis1864":
        fail("source-family identity drift")
    if family in set(cfg["family"]["independent_from_live_training_families"]):
        fail("candidate aliases an incumbent family")
    if cfg["registry_late_check"]["candidate_already_counted"] is not False:
        fail("candidate duplicate-family state changed")

    raw_path = ROOT / cfg["snapshot"]["raw_path"]
    norm_path = ROOT / cfg["snapshot"]["normalized_path"]
    raw = raw_path.read_bytes()
    norm = norm_path.read_bytes()
    if len(raw) != cfg["snapshot"]["raw_bytes"] or sha256_bytes(raw) != cfg["snapshot"]["raw_sha256"]:
        fail("raw snapshot hash/size mismatch")
    if len(norm) != cfg["snapshot"]["normalized_bytes"] or sha256_bytes(norm) != cfg["snapshot"]["normalized_sha256"]:
        fail("normalized snapshot hash/size mismatch")

    raw_lines = raw.decode("utf-8").splitlines()
    norm_lines = norm.decode("utf-8").splitlines()
    if len(raw_lines) != cfg["snapshot"]["records"] or len(norm_lines) != cfg["snapshot"]["records"]:
        fail("record count mismatch")

    ids = []
    rebuilt_norm = []
    raw_line_hashes = []
    norm_line_hashes = []
    for line in raw_lines:
        obj = json.loads(line)
        if list(obj.keys()) != ["id", "text", "sources"]:
            fail("raw schema/order drift")
        if obj["sources"] != ["Nomis1864"]:
            fail(f"non-Nomis source in admitted payload: {obj['id']}")
        if set(obj).intersection({"modern_text", "category", "explanation", "variant_group", "normalized_text"}):
            fail("LLM/enrichment field leaked into payload")
        canonical_line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        if canonical_line != line:
            fail(f"non-canonical JSONL line: {obj['id']}")
        ids.append(obj["id"])
        normalized = unicodedata.normalize("NFC", obj["text"].strip())
        rebuilt_norm.append(normalized)
        raw_line_hashes.append(sha256_bytes((line + "\n").encode("utf-8")))
        norm_line_hashes.append(sha256_bytes((normalized + "\n").encode("utf-8")))

    if ids != cfg["selection"]["selected_ids"]:
        fail("deterministic selection id order drift")
    expected_norm = ("\n".join(rebuilt_norm) + "\n").encode("utf-8")
    if expected_norm != norm:
        fail("normalization is not deterministic from raw snapshot")
    if len(ids) != len(set(ids)):
        fail("duplicate ids")
    if len(rebuilt_norm) != len(set(rebuilt_norm)):
        fail("duplicate normalized records")

    letters = [ch for ch in "\n".join(rebuilt_norm) if ch.isalpha()]
    cyr = [ch for ch in letters if "\u0400" <= ch <= "\u04ff"]
    ratio = (len(cyr) / len(letters)) if letters else 0.0
    uk_specific = sum("\n".join(rebuilt_norm).lower().count(ch) for ch in "іїєґ")
    lang = cfg["language"]
    if len(letters) != lang["alphabetic_codepoint_count"] or len(cyr) != lang["cyrillic_alphabetic_codepoint_count"]:
        fail("language character counts drift")
    if abs(ratio - lang["cyrillic_ratio"]) > 1e-12 or ratio < 0.98:
        fail("Ukrainian/Cyrillic language gate failed")
    if uk_specific != lang["ukrainian_specific_letter_occurrences"] or uk_specific <= 0:
        fail("Ukrainian-specific letter evidence failed")
    if lang["verification"] != "PASS":
        fail("language verdict drift")

    if cfg["quality"]["status"] != "PASS_BOUNDED_SNAPSHOT_ONLY":
        fail("quality scope drift")
    if cfg["quality"]["bounded_snapshot_checks"]["modern_text_llm_artifacts_included"] is not False:
        fail("LLM enrichment leakage")
    if cfg["privacy"]["modern_personal_records_included"] is not False:
        fail("privacy gate drift")

    dedup = cfg["dedup"]
    for key in (
        "within_snapshot_exact_duplicate_ids",
        "within_snapshot_exact_duplicate_normalized_texts",
    ):
        if dedup[key] != 0:
            fail(f"dedup regression: {key}")
    if dedup["live_training_family_overlap"] or dedup["evaluation_reserved_family_overlap"]:
        fail("source-family overlap with training/evaluation authority")
    if dedup["final_test_payload_read"] or dedup["selection_validation_payload_read"]:
        fail("evaluation firewall violated")

    truth = cfg["truth_boundary"]
    if not truth["admission_applies_only_to_committed_bounded_snapshot"]:
        fail("scope truth boundary weakened")
    if truth["full_verba_corpus_admitted"] or truth["modern_protected_collections_admitted"] or truth["evaluation_use_claimed"] or truth["model_training_executed"]:
        fail("forbidden broader claim detected")

    print(
        "NEXT100-027 PASS "
        f"authority={expected_identity} "
        f"records={len(ids)} raw_sha256={sha256_bytes(raw)} "
        f"normalized_sha256={sha256_bytes(norm)} family={family}"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
