from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from twelve_six.data import cross_source_capacity_audit as v1
from twelve_six.data import cross_source_capacity_audit_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tools/run_next100_076_global_dedup_v4.py"
CONFIG = ROOT / "configs/data/next100_076_global_dedup_v4.json"

spec = importlib.util.spec_from_file_location("next100_076_runner", RUNNER)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _sha(payload: bytes) -> str:
    return v1._sha256(payload)


def test_current_config_preserves_fail_closed_boundaries() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["schema_version"] == module.CONFIG_SCHEMA
    assert config["worker_id"] == module.WORKER
    assert config["local_free_only"] is True
    assert config["model_training_executed"] is False
    assert config["tokenizer_fit_executed"] is False
    assert config["paid_compute_used"] is False
    assert config["final_test_payload_read"] is False
    assert config["convergence_authority"]["head_sha"] == "9a6b43849042a4c0dc60d6da5e341827ccf311e7"
    assert config["convergence_authority"]["successor_revalidation_required"] is True
    assert config["acceptance"]["expected_source_objects"] == 22
    assert config["acceptance"]["expected_declared_capacity_bytes_before"] == 320632
    assert config["acceptance"]["expected_modality_capacity_before"]["en"] == 150643
    assert config["late_authorities"]["mdn"]["expected_capacity_bytes"] == 6492
    assert config["downstream"]["next_gate_on_success"] == "BALANCE_DIVERSITY_RETEST"
    assert config["downstream"]["tokenizer_fit"] == "BLOCKED"
    assert config["downstream"]["learned_20m_campaign"] == "BLOCKED"


def test_kmu_normalization_reproduces_nfkc_whitespace_contract() -> None:
    raw = "  Уряд\tзатвердив  план  \n\n  тест   № 1  \n".encode("utf-8")
    assert module.normalize_kmu(raw) == "Уряд затвердив план\nтест No 1\n".encode("utf-8")


def test_mdn_normalization_removes_code_frontmatter_and_link_targets() -> None:
    raw = b"---\ntitle: Example\n---\n# Heading\n\nUse `gzip` with [HTTP](https://example.test).\n\n```js\nsecret();\n```\n\n![image](x.png)\n"
    assert module.normalize_mdn_prose(raw) == b"Heading\n\nUse with HTTP.\n"


def test_payload_row_binds_authority_approved_payload_identity() -> None:
    payload = b"approved normalized training payload\n"
    row = module._payload_row(
        source_id="x",
        family="f",
        origin="o",
        object_id="sha256:" + _sha(payload),
        modality="en",
        authority_ref="terminal authority",
        payload=payload,
        capacity=len(payload),
        provenance="authority-materialized://x",
    )
    assert row["declared_capacity_bytes"] == len(payload)
    assert row["expected_raw_bytes"] == len(payload)
    assert row["expected_raw_sha256"] == _sha(payload)
    assert row["stable_object_id"] == "sha256:" + _sha(payload)


def test_late_binding_status_drift_fails_closed() -> None:
    spec = {
        "head_sha": "a" * 40,
        "blob_sha1": "b" * 40,
        "family_id": "family",
        "terminal_status": "ADMIT_PROSE_ONLY",
    }
    row = {
        "head_sha": "a" * 40,
        "authority_blob_sha1": "b" * 40,
        "family_id": "family",
        "terminal_status": "ADMIT",
    }
    with pytest.raises(module.Next100076Error, match="terminal_status"):
        module._verify_late_binding(spec, row, "test")


def _synthetic_nested_report() -> dict[str, object]:
    payloads = {
        "uk-a": "український синтетичний текст альфа бета гамма дельта епсилон".encode(),
        "en-a": b"english synthetic text alpha beta gamma delta epsilon zeta eta theta",
        "code-a": b"def synthetic_value(x):\n    return x + 1\n",
    }
    rows = []
    for source_id, payload in payloads.items():
        modality = "code" if source_id.startswith("code") else ("uk" if source_id.startswith("uk") else "en")
        rows.append(
            {
                "source_id": source_id,
                "source_family": "family-" + source_id,
                "stable_origin_id": "origin-" + source_id,
                "stable_object_id": "sha256:" + _sha(payload),
                "modality": modality,
                "evidence_status": "DEDICATED_TERMINAL",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": _sha(payload),
                "acquisition_url": "synthetic://" + source_id,
                "origin_key": "origin-" + source_id,
            }
        )
    inventory = {
        "schema_version": "12-6.next100-065-cross-source-dedup.v3",
        "local_free_only": True,
        "model_training_executed": False,
        "sources": rows,
        "lineage_edges": [],
    }
    return v3.audit_payloads(inventory, payloads)


def _envelope(nested: dict[str, object], *, training: bool = False) -> dict[str, object]:
    core = {
        "schema_version": module.SCHEMA,
        "worker_id": module.WORKER,
        "local_free_only": True,
        "model_training_executed": training,
        "tokenizer_fit_executed": False,
        "paid_compute_used": False,
        "final_test_payload_read": False,
        "convergence_revalidated_by_successor": True,
        "dedup_report": nested,
        "next_gate": "BALANCE_DIVERSITY_RETEST",
        "corpus_materialization_claimed": False,
        "research_corpus_v1_released": False,
        "learned_20m_checkpoint_claimed": False,
    }
    return {**core, "report_sha256": _sha(v1._canonical_bytes(core))}


def test_verify_envelope_rejects_premature_training_claim() -> None:
    nested = _synthetic_nested_report()
    before = nested["terminal_candidates"]["declared_capacity_bytes_before"]
    config = {"acceptance": {"expected_source_objects": 3, "expected_declared_capacity_bytes_before": before}}
    with pytest.raises(module.Next100076Error, match="model_training_executed"):
        module.verify_envelope(_envelope(nested, training=True), config)


def test_verify_envelope_accepts_bounded_synthetic_report() -> None:
    nested = _synthetic_nested_report()
    before = nested["terminal_candidates"]["declared_capacity_bytes_before"]
    config = {"acceptance": {"expected_source_objects": 3, "expected_declared_capacity_bytes_before": before}}
    module.verify_envelope(_envelope(nested), config)


def test_capacity_inflation_is_never_accepted() -> None:
    broken = copy.deepcopy(_synthetic_nested_report())
    broken["terminal_candidates"]["conservative_unique_capacity_bytes_after"] = broken["terminal_candidates"]["declared_capacity_bytes_before"] + 1
    core = dict(broken)
    core.pop("report_sha256", None)
    broken["report_sha256"] = _sha(v1._canonical_bytes(core))
    with pytest.raises(v3.CrossSourceV3Error, match="inflated capacity"):
        v3.verify_report(broken)
