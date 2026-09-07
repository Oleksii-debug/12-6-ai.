#!/usr/bin/env python3
"""Replay accepted Presidential decrees into an ephemeral global-dedup payload graph."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "ua.president.official-decrees"
PROOF_SCHEMA = "12-6.d03-ua-president-decrees-replay-proof.v1"
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ReplayError(RuntimeError):
    """Fail-closed replay/materialization error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _load_sibling(module_name: str, filename: str) -> Any:
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    _require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


producer = _load_sibling(
    "_d03_president_decrees_materializer",
    "materialize_d03_ua_president_decrees_v1.py",
)
validator = _load_sibling(
    "_d03_president_decrees_validator",
    "validate_d03_ua_president_decrees_v1.py",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _accepted_records(evidence: Mapping[str, Any]) -> list[dict[str, Any]]:
    mutable = dict(evidence)
    validator.validate(mutable)
    records = [
        dict(record)
        for record in mutable["records"]
        if isinstance(record, Mapping) and record.get("accepted") is True
    ]
    _require(records, "terminal evidence has no accepted records to replay")
    records.sort(key=lambda item: (str(item["normalized_sha256"]), str(item["url"])))
    return records


def _extract_payload(fetcher: Any, record: Mapping[str, Any]) -> bytes:
    url = str(record["url"])
    first = fetcher.fetch(url)
    second = fetcher.fetch(url)
    _require(first.final_url == url, f"first replay final URL drift: {url}")
    _require(second.final_url == url, f"second replay final URL drift: {url}")

    title_a, subject_a, lines_a = producer.extract_document(first.body)
    title_b, subject_b, lines_b = producer.extract_document(second.body)
    payload_a = producer.normalize_document(lines_a)
    payload_b = producer.normalize_document(lines_b)

    _require(title_a == title_b, f"replay title changed across double fetch: {url}")
    _require(subject_a == subject_b, f"replay subject changed across double fetch: {url}")
    _require(
        payload_a == payload_b,
        f"replay normalized payload changed across double fetch: {url}",
    )

    expected_sha = str(record["normalized_sha256"])
    expected_bytes = int(record["normalized_bytes"])
    _require(HEX64_RE.fullmatch(expected_sha) is not None, f"invalid expected SHA-256: {url}")
    _require(len(payload_a) == expected_bytes, f"replay normalized byte drift: {url}")
    _require(_sha256(payload_a) == expected_sha, f"replay normalized SHA-256 drift: {url}")

    title_match = re.search(r"№\s*(\d+/\d{4})", title_a)
    observed_number = title_match.group(1) if title_match else None
    _require(
        observed_number == record.get("document_number"),
        f"replay document-number drift: {url}",
    )
    reasons = producer.privacy_reasons(subject_a, payload_a.decode("utf-8"))
    reasons.extend(producer.quality_reasons(payload_a))
    _require(
        not reasons,
        f"replay policy now rejects accepted record {url}: {sorted(set(reasons))}",
    )
    return payload_a


def replay_payload_graph(
    evidence: Mapping[str, Any],
    *,
    fetcher: Any | None = None,
    request_delay_seconds: float = 1.05,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Return V3 inventory, ephemeral payload bytes, and a text-free replay proof."""
    _require(
        isinstance(request_delay_seconds, (int, float))
        and not isinstance(request_delay_seconds, bool)
        and request_delay_seconds >= 1.0,
        "replay request delay must be >= 1.0 seconds",
    )
    records = _accepted_records(evidence)
    active_fetcher = fetcher or producer.Fetcher(delay_seconds=float(request_delay_seconds))

    rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for record in records:
        payload = _extract_payload(active_fetcher, record)
        digest = str(record["normalized_sha256"])
        source_id = f"ua-president-decree:{digest}"
        _require(source_id not in payloads, f"duplicate replay source id: {source_id}")
        size = len(payload)
        rows.append(
            {
                "source_id": source_id,
                "source_family": SOURCE_ID,
                "stable_origin_id": SOURCE_ID,
                "stable_object_id": f"sha256:{digest}",
                "modality": "uk",
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": (
                    "D03 President decrees "
                    f"{evidence['materialization_identity_sha256']}"
                ),
                "declared_capacity_bytes": size,
                "expected_raw_bytes": size,
                "expected_raw_sha256": digest,
                "acquisition_url": str(record["url"]),
                "origin_key": f"president-decree:{record['url']}",
            }
        )
        payloads[source_id] = payload

    inventory = {
        "schema_version": "12-6.next100-065-cross-source-dedup.v3",
        "local_free_only": True,
        "model_training_executed": False,
        "sources": rows,
        "lineage_edges": [],
    }
    payload_rows = [
        {
            "source_id": row["source_id"],
            "normalized_sha256": row["expected_raw_sha256"],
            "normalized_bytes": row["expected_raw_bytes"],
        }
        for row in rows
    ]
    proof_core: dict[str, Any] = {
        "schema_version": PROOF_SCHEMA,
        "source_id": SOURCE_ID,
        "source_evidence_identity_sha256": evidence["evidence_identity_sha256"],
        "source_materialization_identity_sha256": evidence[
            "materialization_identity_sha256"
        ],
        "replayed_source_count": len(rows),
        "replayed_normalized_bytes": sum(len(payload) for payload in payloads.values()),
        "payload_vector_sha256": _sha256(_canonical_bytes(payload_rows)),
        "inventory_sha256": _sha256(_canonical_bytes(inventory)),
        "raw_text_emitted": False,
        "durable_payload_bytes_emitted": 0,
        "execution_profile": "LOCAL_FREE",
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "model_training_executed": False,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "next_gate": "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
    }
    proof = dict(proof_core)
    proof["proof_sha256"] = _sha256(_canonical_bytes(proof_core))
    return inventory, payloads, proof


def verify_proof(proof: Mapping[str, Any]) -> None:
    _require(proof.get("schema_version") == PROOF_SCHEMA, "replay proof schema drift")
    _require(proof.get("source_id") == SOURCE_ID, "replay proof source drift")
    for key in (
        "source_evidence_identity_sha256",
        "source_materialization_identity_sha256",
        "payload_vector_sha256",
        "inventory_sha256",
        "proof_sha256",
    ):
        value = proof.get(key)
        _require(
            isinstance(value, str) and HEX64_RE.fullmatch(value) is not None,
            f"invalid {key}",
        )
    _require(
        isinstance(proof.get("replayed_source_count"), int)
        and not isinstance(proof.get("replayed_source_count"), bool)
        and proof["replayed_source_count"] > 0,
        "replay proof source count invalid",
    )
    _require(
        isinstance(proof.get("replayed_normalized_bytes"), int)
        and not isinstance(proof.get("replayed_normalized_bytes"), bool)
        and proof["replayed_normalized_bytes"] > 0,
        "replay proof byte count invalid",
    )
    _require(
        proof.get("raw_text_emitted") is False,
        "replay proof raw-text boundary weakened",
    )
    _require(
        proof.get("durable_payload_bytes_emitted") == 0,
        "replay proof durable payload boundary weakened",
    )
    _require(proof.get("execution_profile") == "LOCAL_FREE", "replay execution drift")
    _require(proof.get("training_authorized_bytes") == 0, "replay grants training bytes")
    _require(
        proof.get("authorized_unique_loss_positions") == 0,
        "replay grants unique loss positions",
    )
    for key in ("model_training_executed", "final_test_accessed", "paid_compute_used"):
        _require(proof.get(key) is False, f"replay forbidden claim: {key}")
    _require(
        proof.get("next_gate") == "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
        "replay next gate drift",
    )
    core = dict(proof)
    observed = core.pop("proof_sha256")
    _require(
        observed == _sha256(_canonical_bytes(core)),
        "replay proof self-hash mismatch",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence")
    parser.add_argument("--output-proof", required=True)
    parser.add_argument("--request-delay", type=float, default=1.05)
    args = parser.parse_args()

    evidence = json.loads(Path(args.evidence).read_text(encoding="utf-8"))
    _inventory, _payloads, proof = replay_payload_graph(
        evidence,
        request_delay_seconds=args.request_delay,
    )
    verify_proof(proof)
    output = Path(args.output_proof)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(proof, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(proof, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
