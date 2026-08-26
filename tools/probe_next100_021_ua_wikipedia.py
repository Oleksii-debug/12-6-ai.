from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import re
import tempfile
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_021_ua_wikipedia_retest_v1.json")
CANDIDATES = Path("configs/data/external_source_candidates_ua_en_v1.json")
REGISTRY = Path("data/registry/external_snapshots.v2.json")
EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_config(config: dict[str, Any]) -> None:
    identity = config["evidence_identity_sha256"]
    core = {key: value for key, value in config.items() if key != "evidence_identity_sha256"}
    assert sha256_bytes(canonical_bytes(core)) == identity
    assert config["decision"] == "RETEST"
    assert config["local_free_only"] is True
    assert config["model_training_executed"] is False
    assert config["source"]["evaluation"] == "NOT_SEPARATELY_ADMITTED"
    assert config["rights"]["model_training"] == "RETEST_REQUIRED"
    assert config["rights"]["redistribution"] == "RETEST_REQUIRED"
    assert config["rights"]["evaluation"] == "NOT_SEPARATELY_ADMITTED"
    assert config["bounded_probe"]["retain_raw_payload"] is False
    assert config["bounded_probe"]["retain_normalized_payload"] is False
    assert config["bounded_probe"]["artifact_contains_text"] is False
    assert config["admission"] == {
        "training_snapshot_created": False,
        "registry_mutated": False,
        "corpus_contract_mutated": False,
        "evaluation_material_reserved": False,
    }
    expected = int(config["upstream"]["expected_size_bytes"])
    cap = int(config["bounded_probe"]["raw_download_cap_bytes"])
    assert expected == int(config["bounded_probe"]["expected_raw_bytes"])
    assert 0 < expected <= cap
    assert config["upstream"]["dated_url"].endswith(config["upstream"]["dump_object"])
    assert "/20260801/" in config["upstream"]["dated_url"]
    assert "/latest/" not in config["upstream"]["dated_url"]


def verify_incumbent_state(config: dict[str, Any]) -> dict[str, Any]:
    candidates = load_json(CANDIDATES)
    source_id = config["source"]["source_id"]
    rows = [item for item in candidates["sources"] if item["source_id"] == source_id]
    assert len(rows) == 1
    candidate = rows[0]
    assert candidate["eligibility_status"] == "BLOCKED_BY_RIGHTS"
    assert candidate["rights"]["status"] == "REVIEW_REQUIRED"
    assert candidate["rights"]["allows_model_training"] is False
    assert "ShareAlike" in candidate["block_reason"]

    registry = load_json(REGISTRY)
    admitted = [item["source_id"] for item in registry["sources"]]
    assert source_id not in admitted
    assert registry["registry_identity_sha256"] == config["registry_concurrency"][
        "base_registry_identity_sha256"
    ]
    return {
        "historical_candidate_status": candidate["rights"]["status"],
        "already_admitted": False,
        "registry_identity_sha256": registry["registry_identity_sha256"],
    }


def normalize_wikitext(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Cf")
    value = "\n".join(line.rstrip() for line in value.split("\n"))
    return value.strip()


def download_exact(config: dict[str, Any], target: Path) -> dict[str, Any]:
    upstream = config["upstream"]
    cap = int(config["bounded_probe"]["raw_download_cap_bytes"])
    expected_size = int(upstream["expected_size_bytes"])
    assert expected_size <= cap
    request = urllib.request.Request(
        upstream["dated_url"],
        headers={"User-Agent": "12-6-ai-NEXT100-021/1.0 bounded-source-validation"},
    )
    sha1 = hashlib.sha1()  # nosec B324 - required only to verify Wikimedia's published digest
    sha256 = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        length = response.headers.get("Content-Length")
        if length is not None:
            assert int(length) == expected_size
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            assert total <= cap, "bounded acquisition cap exceeded"
            sha1.update(chunk)
            sha256.update(chunk)
            out.write(chunk)
    assert total == expected_size
    assert sha1.hexdigest() == upstream["expected_sha1"]
    return {
        "raw_bytes": total,
        "raw_sha1": sha1.hexdigest(),
        "raw_sha256": sha256.hexdigest(),
    }


def extract_probe(config: dict[str, Any], path: Path) -> dict[str, Any]:
    record_cap = int(config["bounded_probe"]["normalized_record_cap"])
    byte_cap = int(config["bounded_probe"]["normalized_byte_cap"])
    aggregate = hashlib.sha256()
    aggregate_bytes = 0
    records: list[dict[str, Any]] = []
    text_hashes: set[str] = set()
    exact_duplicates = 0
    email_matches = 0
    phone_matches = 0

    with bz2.open(path, "rb") as stream:
        for _event, page in ET.iterparse(stream, events=("end",)):
            if not page.tag.endswith("}page") and page.tag != "page":
                continue
            ns = page.find("./{*}ns")
            redirect = page.find("./{*}redirect")
            if ns is None or ns.text != "0" or redirect is not None:
                page.clear()
                continue
            page_id_node = page.find("./{*}id")
            revision = page.find("./{*}revision")
            if page_id_node is None or revision is None:
                page.clear()
                continue
            revision_id_node = revision.find("./{*}id")
            text_node = revision.find("./{*}text")
            if revision_id_node is None or text_node is None or not text_node.text:
                page.clear()
                continue
            normalized = normalize_wikitext(text_node.text)
            if not normalized:
                page.clear()
                continue
            page_id = int(page_id_node.text or "0")
            revision_id = int(revision_id_node.text or "0")
            text_payload = normalized.encode("utf-8")
            text_sha256 = sha256_bytes(text_payload)
            record_payload = canonical_bytes(
                {"page_id": page_id, "revision_id": revision_id, "text": normalized}
            )
            if records and aggregate_bytes + len(record_payload) > byte_cap:
                page.clear()
                break
            if text_sha256 in text_hashes:
                exact_duplicates += 1
            text_hashes.add(text_sha256)
            aggregate.update(record_payload)
            aggregate_bytes += len(record_payload)
            email_matches += len(EMAIL_RE.findall(normalized))
            phone_matches += len(PHONE_RE.findall(normalized))
            records.append(
                {
                    "page_id": page_id,
                    "revision_id": revision_id,
                    "normalized_text_sha256": text_sha256,
                    "normalized_text_bytes": len(text_payload),
                    "canonical_record_sha256": sha256_bytes(record_payload),
                }
            )
            page.clear()
            if len(records) >= record_cap:
                break

    assert records, "bounded normalized probe produced no records"
    assert len(records) <= record_cap
    assert aggregate_bytes <= byte_cap
    return {
        "normalization_policy": config["normalization"]["policy_id"],
        "records": records,
        "record_count": len(records),
        "canonical_normalized_bytes": aggregate_bytes,
        "normalized_payload_sha256": aggregate.hexdigest(),
        "within_probe_exact_duplicate_records": exact_duplicates,
        "privacy_signal_counts": {
            "email_like_matches": email_matches,
            "phone_like_matches": phone_matches,
        },
        "text_retained_in_report": False,
    }


def build_report(config: dict[str, Any], source_sha: str) -> dict[str, Any]:
    incumbent = verify_incumbent_state(config)
    with tempfile.TemporaryDirectory(prefix="next100-021-ukwiki-") as temp_dir:
        raw_path = Path(temp_dir) / config["upstream"]["dump_object"]
        raw = download_exact(config, raw_path)
        normalized = extract_probe(config, raw_path)
    core = {
        "schema_version": "12-6.next100-021-ua-wikipedia-runtime-evidence.v1",
        "worker": config["worker"],
        "source_sha": source_sha,
        "decision": "RETEST",
        "local_free_only": True,
        "model_training_executed": False,
        "training_snapshot_created": False,
        "evaluation_material_reserved": False,
        "source": config["source"],
        "upstream": config["upstream"],
        "rights": {
            "license_id": config["rights"]["license_id"],
            "model_training": config["rights"]["model_training"],
            "redistribution": config["rights"]["redistribution"],
            "evaluation": config["rights"]["evaluation"],
            "policy_blocker": config["rights"]["policy_blocker"],
        },
        "incumbent_registry": incumbent,
        "bounded_acquisition": raw,
        "bounded_normalization": normalized,
        "privacy_status": config["privacy"]["status"],
        "dedup_status": config["dedup"]["status"],
        "terminal_blockers": config["terminal_blockers"],
        "payload_retention": {
            "raw": False,
            "normalized": False,
            "artifact_contains_text": False,
        },
    }
    return {**core, "report_identity_sha256": sha256_bytes(canonical_bytes(core))}


def verify_report(report: dict[str, Any]) -> None:
    identity = report["report_identity_sha256"]
    core = {key: value for key, value in report.items() if key != "report_identity_sha256"}
    assert sha256_bytes(canonical_bytes(core)) == identity
    assert report["decision"] == "RETEST"
    assert report["model_training_executed"] is False
    assert report["training_snapshot_created"] is False
    assert report["evaluation_material_reserved"] is False
    assert report["payload_retention"] == {
        "raw": False,
        "normalized": False,
        "artifact_contains_text": False,
    }
    assert report["bounded_acquisition"]["raw_bytes"] > 0
    assert report["bounded_normalization"]["record_count"] > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "probe", "verify-report"))
    parser.add_argument("--output", type=Path, default=Path("next100-021-ukwiki-report.json"))
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "LOCAL"))
    args = parser.parse_args()

    config = load_json(CONFIG)
    verify_config(config)
    verify_incumbent_state(config)
    if args.command == "validate":
        return
    if args.command == "verify-report":
        verify_report(load_json(args.output))
        return
    report = build_report(config, args.source_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_report(report)


if __name__ == "__main__":
    main()
