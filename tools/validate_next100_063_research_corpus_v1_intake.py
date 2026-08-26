#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

DATA228_HEAD = "46a70c990dab6ff72bb84ddb54cff1156b491b40"
DATA228_REPORT_PATH = "reports/data228/source-probe.json"

EXPECTED_IDENTITY_PARTS = [
    "12-6.next100-063-research-corpus-v1-intake.v1",
    "8820ba1b255f6bb95c7db0531fd846078a1aae01",
    "939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81",
    "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c",
    "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9",
    "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
    "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d",
    "860b5bd9aed72d9bc754a4f73445d18ff3807408a0d6f5a18a83eca14b9f1712",
    "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd",
    "45/35/20|min2|no-replay",
]


def authority_identity() -> str:
    return hashlib.sha256("|".join(EXPECTED_IDENTITY_PARTS).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def validate(data: dict) -> None:
    _require(data["schema_version"] == EXPECTED_IDENTITY_PARTS[0], "schema drift")
    _require(data["execution_profile"] == "LOCAL_FREE", "execution profile must remain LOCAL_FREE")
    _require(data["authority_identity"]["sha256"] == authority_identity(), "authority identity mismatch")

    base = data["base_data301"]
    _require(base["head_sha"] == EXPECTED_IDENTITY_PARTS[1], "DATA-301 head drift")
    _require(base["evidence_identity_sha256"] == EXPECTED_IDENTITY_PARTS[2], "DATA-301 evidence drift")
    _require(base["corpus_identity"] is None and base["shard_identity"] is None, "base corpus identity must remain null")
    _require(base["authorized_balanced_no_replay_capacity"] == 0, "base capacity must remain zero")

    registry = data["base_registry"]
    _require(registry["registry_identity_sha256"] == EXPECTED_IDENTITY_PARTS[3], "DATA-287 registry drift")
    incumbent = set(registry["incumbent_families"])
    _require(len(incumbent) == 4, "incumbent family set changed")

    selection = data["selection_validation"]
    _require(selection["composite_identity_sha256"] == EXPECTED_IDENTITY_PARTS[8], "EVAL-303 identity drift")
    _require(selection["record_count"] == 10, "selection record count drift")
    _require(not selection["training_allowed"] and not selection["tokenizer_fit_allowed"], "selection data leaked into training")

    authorities = data["terminal_source_authorities"]
    _require(len(authorities) == 3, "expected exactly three additive terminal source authorities")
    expected = {
        "NEXT100-026-DATA-UA-CABINET-MINISTRY": (449, "40950a950b60921fd856af2719e1ae2486d9e892", "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9", "uk"),
        "NEXT100-022-DATA-UA-WIKISOURCE": (455, "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2", "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6", "uk"),
        "NEXT100-037-DATA-EN-PYTHON-DOCS": (467, "5a6a495a24bce449334cbc5126d0114f61a9f57c", "46a00dc70db690ae2b3c4495a75283e7e752bdccb1047d4318c2ebadfa392f0d", "en"),
    }
    seen_families: set[str] = set()
    for item in authorities:
        _require(item["worker"] in expected, "unexpected worker")
        pr, head, identity, stratum = expected[item["worker"]]
        _require(item["pr"] == pr and item["head_sha"] == head, "source head/PR drift")
        _require(item["authority_identity_sha256"] == identity, "source authority identity drift")
        _require(item["stratum"] == stratum, "source stratum drift")
        _require(item["dedicated_workflow_conclusion"] == "success", "nonterminal source workflow")
        _require(item["evaluation_rights"] == "NOT_SEPARATELY_ADMITTED", "training source gained evaluation rights")
        family = item["family"]
        _require(family not in incumbent and family not in seen_families, "family collision or double credit")
        seen_families.add(family)

    cpython = next(x for x in authorities if x["worker"] == "NEXT100-037-DATA-EN-PYTHON-DOCS")
    _require(cpython["data228_source_probe_report_sha256"] == EXPECTED_IDENTITY_PARTS[7], "DATA-228 probe identity drift")
    _require(cpython["accepted_chunk_count"] == 14 and cpython["rejected_chunk_count"] == 2, "CPython chunk gate drift")
    _require(cpython["accepted_chunk_utf8_bytes_sum_from_terminal_probe"] == 15540, "CPython accepted-byte sum drift")
    _require(cpython["accepted_chunk_utf8_bytes_min"] == 290 and cpython["accepted_chunk_utf8_bytes_max"] == 1196, "CPython accepted-byte range drift")
    _require(cpython["materialized_training_payload_bytes"] is None, "CPython accepted bytes must not be relabelled as materialized")

    inv = data["source_authority_inventory"]
    _require(inv["source_normalized_bytes_total"] == 183061 + 9153 + 1479 + 17901, "source byte accounting mismatch")
    _require(inv["cpython_accepted_chunk_bytes_bound_by_terminal_probe"] == 15540, "bound CPython byte accounting mismatch")
    _require(inv["known_accepted_or_incumbent_bytes_before_successor_global_gates"] == 183061 + 9153 + 1479 + 15540, "known accepted/incumbent byte accounting mismatch")
    _require(inv["exact_materialized_source_bytes_lower_bound_before_successor_gates"] == 183061 + 9153 + 1479, "materialized lower-bound mismatch")
    _require(inv["source_bytes_are_not_loss_positions"], "source bytes must not be relabelled as loss positions")
    _require(not inv["cpython_accepted_payload_bytes_materialized"], "CPython materialization state drift")

    card = data["family_cardinality"]
    _require(card["intake_total"] == {"uk": 3, "en": 2, "code": 2}, "family cardinality mismatch")
    _require(card["minimum_required_per_stratum"] == 2, "minimum family rule drift")
    _require(card["family_cardinality_prerequisite_satisfied_at_source_authority_layer"], "family cardinality prerequisite not recognized")
    _require(not card["replay_or_alias_credit_used"], "replay/alias family credit forbidden")

    mix = data["mixture_policy"]
    _require((mix["uk_fraction"], mix["en_fraction"], mix["code_fraction"]) == (0.45, 0.35, 0.20), "mixture drift")
    _require(not mix["replay_allowed"], "replay must remain forbidden")

    training = data["training_authority"]
    _require(training["corpus_identity"] is None and training["shard_identity"] is None, "corpus/shard identity fabricated")
    _require(training["postpack_unique_loss_positions"] is None, "post-pack ledger fabricated")
    _require(training["authorized_training_exposure"] == 0, "training exposure must stay zero")
    _require(not training["tokenizer_fit_authorized"], "tokenizer fit must remain blocked")
    _require(not training["long_training_authorized"] and not training["paid_compute_authorized"], "long/paid training not authorized")
    _require(training["optimizer_updates_executed"] == 0 and not training["base_weight_mutation_executed"], "weight mutation claim drift")

    truth = data["truth_boundary"]
    _require(not truth["research_corpus_v1_frozen"] and not truth["training_ready"] and not truth["learned_20m_claimed"], "truth-boundary overclaim")


def verify_git_authorities(data: dict) -> None:
    cpython_authority = None
    for item in data["terminal_source_authorities"]:
        raw = subprocess.check_output(["git", "show", f"{item['head_sha']}:{item['authority_config_path']}"])
        _require(_git_blob_sha1(raw) == item["authority_config_git_blob_sha1"], f"authority blob mismatch for {item['worker']}")
        source = json.loads(raw)
        observed = source.get("authority_identity_sha256", source.get("manifest_identity_sha256"))
        _require(observed == item["authority_identity_sha256"], f"authority payload identity mismatch for {item['worker']}")
        if item["worker"] == "NEXT100-037-DATA-EN-PYTHON-DOCS":
            cpython_authority = source
            _require(source["lineage"]["data228_probe_report_sha256"] == item["data228_source_probe_report_sha256"], "CPython/DATA-228 lineage mismatch")
            preview = source["quality_privacy"]
            _require(preview["accepted_chunk_count"] == 14 and preview["rejected_chunk_count"] == 2, "CPython preview count mismatch")

    _require(cpython_authority is not None, "CPython authority missing")
    report_raw = subprocess.check_output(["git", "show", f"{DATA228_HEAD}:{DATA228_REPORT_PATH}"])
    report = json.loads(report_raw)
    _require(report["report_sha256"] == EXPECTED_IDENTITY_PARTS[7], "terminal DATA-228 report SHA drift")
    cpython_probe = next(x for x in report["candidates"] if x["source_id"] == "en.python.docs.tutorial-introduction")
    preview = cpython_probe["privacy_quality_preview"]
    _require(preview["accepted_chunk_count"] == 14 and preview["rejected_chunk_count"] == 2, "DATA-228 CPython count drift")
    _require(preview["rejection_reasons"] == {"pii_phone": 2}, "DATA-228 CPython rejection drift")
    _require(preview["accepted_document_utf8_bytes"] == {"min": 290, "max": 1196, "mean": 1110.0}, "DATA-228 CPython byte statistics drift")
    _require(int(preview["accepted_document_utf8_bytes"]["mean"] * preview["accepted_chunk_count"]) == 15540, "DATA-228 CPython aggregate byte derivation drift")
    _require(preview["accepted_normalized_sha256"] == cpython_authority["quality_privacy"]["accepted_normalized_sha256"], "CPython accepted hash vector mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="configs/data/next100_063_research_corpus_v1_intake_v1.json")
    parser.add_argument("--verify-git-authorities", action="store_true")
    args = parser.parse_args()
    data = json.loads(Path(args.path).read_text(encoding="utf-8"))
    validate(data)
    if args.verify_git_authorities:
        verify_git_authorities(data)
    print(f"PASS next100-063 authority_identity={authority_identity()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
