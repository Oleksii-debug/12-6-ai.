from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(filename: str, module_name: str):
    path = ROOT / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


replay = load_tool(
    "replay_d03_ua_president_decrees_v1.py",
    "d03_president_decrees_replay",
)
producer = replay.producer
validator = replay.validator


def decree_html(
    *,
    number: str = "873/2026",
    subject: str = "Про Концепцію розвитку державної політики",
    body: str | None = None,
) -> bytes:
    body = body or (
        "Український нормативний текст про державну політику та розвиток. " * 60
    )
    slug = number.replace("/", "")
    return f"""<html><body>
<h1>УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №{number}</h1>
<div>{subject}</div>
<article>
<p>{body}</p>
<p>Президент України В.ЗЕЛЕНСЬКИЙ</p>
<p>4 вересня 2026 року</p>
</article>
<footer><h3>Новини</h3><p>неофіційний текст</p></footer>
<a href="/documents/{slug}-61461">УКАЗ ПРЕЗИДЕНТА УКРАЇНИ №{number}</a>
</body></html>""".encode()


def terminal_evidence(payload: bytes) -> dict:
    url = "https://www.president.gov.ua/documents/8732026-61461"
    record = {
        "url": url,
        "document_number": "873/2026",
        "raw_sha256_a": "a" * 64,
        "raw_sha256_b": "b" * 64,
        "raw_byte_identical": False,
        "normalized_sha256": producer.sha256(payload),
        "normalized_bytes": len(payload),
        "accepted": True,
    }
    evidence = {
        "schema": validator.SCHEMA,
        "status": "PASS_OBSERVED_YIELD",
        "generated_at_utc": "2026-09-07T20:00:00+00:00",
        "source": {
            "source_id": validator.SOURCE_ID,
            "family_id": validator.SOURCE_ID,
            "stratum": "uk",
            "catalog_url": "https://www.president.gov.ua/documents/decrees/",
            "allowed_origin": "https://www.president.gov.ua",
            "rights_scope": "OFFICIAL_DECREE_TEXT_ONLY",
            "website_blanket_license_not_used_as_training_authority": True,
            "ukraine_copyright_law": {
                "law": "2811-IX",
                "article": "8(1)(3)",
                "authority_url": "https://zakon.rada.gov.ua/laws/show/2811-20",
                "scope": "official acts only",
            },
        },
        "execution": {
            "class": "LOCAL_FREE",
            "max_catalog_pages": 4,
            "max_documents": 60,
            "request_delay_seconds": 1.05,
            "catalog_pages_observed": 1,
            "documents_probed": 1,
            "double_fetch_required": True,
            "final_test_accessed": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
        },
        "catalog_pages": [
            {
                "url": "https://www.president.gov.ua/documents/decrees/",
                "page": 1,
                "document_count": 1,
                "document_url_set_sha256": "d" * 64,
            }
        ],
        "observed_yield": {
            "accepted_documents": 1,
            "accepted_normalized_bytes": len(payload),
            "one_conservative_family": True,
            "rejected_documents": 0,
            "rejection_counts": {},
        },
        "records": [record],
        "downstream_required": [
            "GLOBAL_EXACT_NEAR_FRAGMENT_LINEAGE_DEDUP",
            "RESERVED_EVALUATION_DECONTAMINATION",
            "POST_COMPOSITION_QUALITY_PRIVACY_BALANCE_FAMILY_CAPS",
            "CLUSTER_SAFE_SPLIT",
            "DETERMINISTIC_TOKENIZER_PACKING_DOUBLE_BUILD",
            "POSITIVE_EXACT_UNIQUE_CAUSAL_LOSS_LEDGER",
        ],
        "claims": {
            "canonical_capacity_credit_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "tokenizer_fit_authorized": False,
            "research_corpus_released": False,
            "learned_20m_claim": False,
        },
    }
    materialization = json.dumps(
        validator.canonical_materialization_view(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence["materialization_identity_sha256"] = validator.sha256(materialization)
    identity_view = dict(evidence)
    identity_view.pop("generated_at_utc")
    identity_view.pop("materialization_identity_sha256")
    evidence["evidence_identity_sha256"] = validator.sha256(
        json.dumps(
            identity_view,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    validator.validate(evidence)
    return evidence


class FakeFetcher:
    def __init__(
        self,
        bodies: list[bytes],
        *,
        final_url: str = "https://www.president.gov.ua/documents/8732026-61461",
    ) -> None:
        self.bodies = list(bodies)
        self.final_url = final_url

    def fetch(self, url: str):
        assert self.bodies
        body = self.bodies.pop(0)
        return producer.FetchResult(
            requested_url=url,
            final_url=self.final_url,
            content_type="text/html",
            body=body,
        )


def normalized_payload(html: bytes) -> bytes:
    _title, _subject, lines = producer.extract_document(html)
    return producer.normalize_document(lines)


def test_replay_builds_v3_compatible_ephemeral_graph_and_text_free_proof():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    inventory, payloads, proof = replay.replay_payload_graph(
        evidence,
        fetcher=FakeFetcher([html, html]),
    )

    assert inventory["schema_version"] == "12-6.next100-065-cross-source-dedup.v3"
    assert inventory["model_training_executed"] is False
    assert inventory["lineage_edges"] == []
    assert len(inventory["sources"]) == 1
    row = inventory["sources"][0]
    assert row["source_family"] == replay.SOURCE_ID
    assert row["modality"] == "uk"
    assert row["evidence_status"] == "DEDICATED_TERMINAL"
    assert row["expected_raw_sha256"] == producer.sha256(payload)
    assert payloads == {row["source_id"]: payload}
    assert proof["raw_text_emitted"] is False
    assert proof["durable_payload_bytes_emitted"] == 0
    assert proof["training_authorized_bytes"] == 0
    replay.verify_proof(proof)


def test_replay_rejects_normalized_payload_mutation():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    mutated = decree_html(
        body="Інший український нормативний текст державної політики. " * 70
    )
    with pytest.raises(replay.ReplayError, match="normalized .* drift"):
        replay.replay_payload_graph(
            evidence,
            fetcher=FakeFetcher([mutated, mutated]),
        )


def test_replay_rejects_double_fetch_instability():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    second = decree_html(
        body="Змінений український нормативний текст державної політики. " * 70
    )
    with pytest.raises(replay.ReplayError, match="changed across double fetch"):
        replay.replay_payload_graph(
            evidence,
            fetcher=FakeFetcher([html, second]),
        )


def test_replay_rejects_same_origin_redirect_identity_drift():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    fetcher = FakeFetcher(
        [html, html],
        final_url="https://www.president.gov.ua/documents/other-61461",
    )
    with pytest.raises(replay.ReplayError, match="final URL drift"):
        replay.replay_payload_graph(evidence, fetcher=fetcher)


def test_replay_refuses_zero_yield_terminal_evidence():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    record = evidence["records"][0]
    record["accepted"] = False
    record["reason"] = "below_minimum_bytes"
    evidence["observed_yield"]["accepted_documents"] = 0
    evidence["observed_yield"]["accepted_normalized_bytes"] = 0
    evidence["observed_yield"]["rejected_documents"] = 1
    evidence["observed_yield"]["rejection_counts"] = {"below_minimum_bytes": 1}
    evidence["status"] = "PASS_ZERO_YIELD"
    materialization = json.dumps(
        validator.canonical_materialization_view(evidence),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence["materialization_identity_sha256"] = validator.sha256(materialization)
    identity_view = dict(evidence)
    identity_view.pop("generated_at_utc")
    identity_view.pop("materialization_identity_sha256")
    evidence["evidence_identity_sha256"] = validator.sha256(
        json.dumps(
            identity_view,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    validator.validate(evidence)
    with pytest.raises(replay.ReplayError, match="no accepted records"):
        replay.replay_payload_graph(evidence, fetcher=FakeFetcher([]))


@pytest.mark.parametrize("delay", [0, 0.99, True])
def test_replay_refuses_aggressive_or_boolean_request_delay(delay):
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    with pytest.raises(replay.ReplayError, match="delay"):
        replay.replay_payload_graph(
            evidence,
            fetcher=FakeFetcher([html, html]),
            request_delay_seconds=delay,
        )


def test_replay_proof_is_tamper_evident():
    html = decree_html()
    payload = normalized_payload(html)
    evidence = terminal_evidence(payload)
    _inventory, _payloads, proof = replay.replay_payload_graph(
        evidence,
        fetcher=FakeFetcher([html, html]),
    )
    proof["training_authorized_bytes"] = 1
    with pytest.raises(replay.ReplayError, match="training"):
        replay.verify_proof(proof)
