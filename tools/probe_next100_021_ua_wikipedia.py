from __future__ import annotations

import argparse
import hashlib
import json
import os
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any

CONFIG = Path("configs/data/next100_021_ua_wikipedia_retest_v1.json")
CANDIDATES = Path("configs/data/external_source_candidates_ua_en_v1.json")
REGISTRY = Path("data/registry/external_snapshots.v2.json")


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
    assert config["decision"] == "REJECT"
    assert config["local_free_only"] is True
    assert config["model_training_executed"] is False
    assert config["rights"]["model_training"] == "REJECT"
    assert config["rights"]["redistribution"] == "NOT_ADMITTED"
    assert config["rights"]["evaluation"] == "NOT_SEPARATELY_ADMITTED"
    assert config["bounded_acquisition"]["content_payload_executed"] is False
    assert config["bounded_acquisition"]["network_content_bytes_downloaded"] == 0
    assert config["bounded_acquisition"]["raw_sha256"] is None
    assert config["bounded_acquisition"]["normalized_sha256"] is None
    assert config["normalization"]["execution_status"] == "NOT_RUN_RIGHTS_GATE_FAILED"
    assert config["dedup"]["execution_status"] == "NOT_RUN_RIGHTS_GATE_FAILED"
    assert config["terminal"]["training_snapshot_created"] is False
    assert config["terminal"]["registry_mutated"] is False
    assert config["terminal"]["corpus_contract_mutated"] is False
    assert config["terminal"]["evaluation_material_reserved"] is False
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


def verify_normalization_fixture() -> dict[str, str]:
    source = "Ａ\r\nрядок  \u200b\rкінець  "
    expected = "A\nрядок\nкінець"
    first = normalize_wikitext(source)
    second = normalize_wikitext(source)
    assert first == expected
    assert first == second
    return {
        "fixture_input_sha256": sha256_bytes(source.encode("utf-8")),
        "fixture_output_sha256": sha256_bytes(first.encode("utf-8")),
        "repeat_output_sha256": sha256_bytes(second.encode("utf-8")),
    }


def fetch_manifest_row(config: dict[str, Any]) -> dict[str, Any]:
    probe = config["bounded_acquisition"]["metadata_probe"]
    cap = int(probe["max_bytes"])
    request = urllib.request.Request(
        config["upstream"]["sha1_manifest_url"],
        headers={"User-Agent": "12-6-ai-NEXT100-021/2.0 rights-gated-metadata-probe"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(cap + 1)
    assert len(payload) <= cap, "metadata acquisition cap exceeded"
    text = payload.decode("utf-8")
    expected = (
        f'{config["upstream"]["expected_sha1"]}  '
        f'{config["upstream"]["dump_object"]}'
    )
    assert expected in text.splitlines()
    return {
        "metadata_bytes_downloaded": len(payload),
        "metadata_sha256": sha256_bytes(payload),
        "verified_dump_object": config["upstream"]["dump_object"],
        "verified_upstream_sha1": config["upstream"]["expected_sha1"],
        "content_payload_bytes_downloaded": 0,
    }


def build_report(config: dict[str, Any], source_sha: str, *, network: bool) -> dict[str, Any]:
    incumbent = verify_incumbent_state(config)
    fixture = verify_normalization_fixture()
    metadata = (
        fetch_manifest_row(config)
        if network
        else {
            "metadata_bytes_downloaded": 0,
            "metadata_sha256": None,
            "verified_dump_object": config["upstream"]["dump_object"],
            "verified_upstream_sha1": config["upstream"]["expected_sha1"],
            "content_payload_bytes_downloaded": 0,
        }
    )
    core = {
        "schema_version": "12-6.next100-021-ua-wikipedia-runtime-evidence.v2",
        "worker": config["worker"],
        "source_sha": source_sha,
        "decision": "REJECT",
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
            "project_policy_status": config["rights"]["project_policy_status"],
            "rejection_root_cause": config["rights"]["rejection_root_cause"],
        },
        "incumbent_registry": incumbent,
        "bounded_metadata_acquisition": metadata,
        "content_materialization": {
            "executed": False,
            "network_content_bytes_downloaded": 0,
            "raw_sha256": None,
            "normalized_sha256": None,
        },
        "normalization": {
            "policy_id": config["normalization"]["policy_id"],
            "execution_status": config["normalization"]["execution_status"],
            "fixture_determinism": fixture,
        },
        "privacy_status": config["privacy"]["content_gate"],
        "dedup_status": config["dedup"]["execution_status"],
        "independent_family_credit": config["dedup"]["independent_family_credit"],
        "terminal_retest_condition": config["terminal"]["retest_condition"],
    }
    return {**core, "report_identity_sha256": sha256_bytes(canonical_bytes(core))}


def verify_report(report: dict[str, Any]) -> None:
    identity = report["report_identity_sha256"]
    core = {key: value for key, value in report.items() if key != "report_identity_sha256"}
    assert sha256_bytes(canonical_bytes(core)) == identity
    assert report["decision"] == "REJECT"
    assert report["model_training_executed"] is False
    assert report["training_snapshot_created"] is False
    assert report["evaluation_material_reserved"] is False
    assert report["content_materialization"]["executed"] is False
    assert report["content_materialization"]["network_content_bytes_downloaded"] == 0
    assert report["content_materialization"]["raw_sha256"] is None
    assert report["content_materialization"]["normalized_sha256"] is None
    assert report["independent_family_credit"] == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("validate", "probe-metadata", "verify-report"))
    parser.add_argument("--output", type=Path, default=Path("next100-021-ukwiki-report.json"))
    parser.add_argument("--source-sha", default=os.environ.get("GITHUB_SHA", "LOCAL"))
    args = parser.parse_args()

    config = load_json(CONFIG)
    verify_config(config)
    verify_incumbent_state(config)
    verify_normalization_fixture()
    if args.command == "validate":
        return
    if args.command == "verify-report":
        verify_report(load_json(args.output))
        return
    report = build_report(config, args.source_sha, network=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_report(report)


if __name__ == "__main__":
    main()
