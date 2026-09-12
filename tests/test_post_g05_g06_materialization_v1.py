from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import twelve_six.data.post_g05_g06_materialization_v1 as materializer_module
from twelve_six.data.post_g05_g06_materialization_v1 import (
    PostG05G06MaterializationError,
    canonical_record_bytes,
    materialize_post_g05_g06,
    materialize_record_inventory,
)

ZERO = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
}
POLICY = "a" * 64
HEAD = "b" * 40


def cjson(value: object) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return text.encode()


def seal(value: dict[str, object], field: str) -> dict[str, object]:
    core = dict(value)
    core.pop(field, None)
    value[field] = hashlib.sha256(cjson(core)).hexdigest()
    return value


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw,
        usedforsecurity=False,
    ).hexdigest()


def running_materializer_blob() -> str:
    return blob_sha(Path(materializer_module.__file__).read_bytes())


def privacy_source(*, reject_redacted: bool = False) -> bytes:
    marker_rule = (
        "if '<redacted>' in text: "
        "out.append(Finding('marker','QUARANTINE',0,len(text)))"
        if reject_redacted
        else ""
    )
    source = f"""from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Finding:
    detector_id: str
    action: str
    start: int
    end: int

@dataclass(frozen=True)
class Scan:
    action: str

def policy_manifest():
    return {{"policy_sha256": "{POLICY}"}}

def detect(text):
    out = []
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}"
    for match in re.finditer(pattern, text):
        out.append(Finding("email", "REDACT", match.start(), match.end()))
    {marker_rule}
    return tuple(out)

def hash_safe_scan(data):
    text = data.decode("utf-8") if isinstance(data, bytes) else data
    findings = detect(text)
    rank = {{"ALLOW": 0, "REDACT": 1, "QUARANTINE": 2, "EXCLUDE": 3}}
    action = max((finding.action for finding in findings), key=rank.get, default="ALLOW")
    return Scan(action)
"""
    return source.encode()


def privacy_binding(source: bytes) -> dict[str, str]:
    return {
        "module": "twelve_six.data.privacy_filter_v3",
        "relative_path": "privacy_filter_v3.py",
        "implementation_git_blob_sha1": blob_sha(source),
        "policy_schema_version": "test-policy.v1",
        "policy_sha256": POLICY,
    }


def authorities(
    records: list[dict[str, str]],
    actions: dict[str, str],
    statuses: dict[str, str],
    binding: dict[str, str],
):
    g05_rows = []
    g06_rows = []
    for record in sorted(records, key=lambda row: row["record_id"]):
        raw = record["normalized_payload"].encode()
        common = {
            "record_id": record["record_id"],
            "mode": record["modality"],
            "payload_sha256": hashlib.sha256(raw).hexdigest(),
            "utf8_bytes": len(raw),
        }
        status = statuses[record["record_id"]]
        g05_rows.append(
            {
                **common,
                "status": status,
                "retained_utf8_bytes": 0 if status == "REJECT_DOCUMENT" else len(raw),
                "rejected_utf8_bytes": len(raw) if status == "REJECT_DOCUMENT" else 0,
            }
        )
        g06_rows.append({**common, "action": actions[record["record_id"]]})
    g05 = {
        "schema_version": "12-6.g05-quality-execution-authority.v1",
        "authority_class": "G05_QUALITY_EXECUTION_ZERO_CREDIT",
        "execution_rows_sha256": hashlib.sha256(cjson(g05_rows)).hexdigest(),
        "records": g05_rows,
        "truth_boundary": dict(ZERO),
    }
    seal(g05, "execution_identity_sha256")
    g06 = {
        "schema_version": "12-6.g06-privacy-execution-authority.v1",
        "authority_class": "G06_PRIVACY_EXECUTION_ZERO_CREDIT",
        "privacy_binding": dict(binding),
        "execution_rows_sha256": hashlib.sha256(cjson(g06_rows)).hexdigest(),
        "records": g06_rows,
        "truth_boundary": dict(ZERO),
    }
    seal(g06, "execution_identity_sha256")
    envelope = {
        "schema_version": "12-6.current-survivor-g06-dependency-bound-execution.v1",
        "execution_profile": "LOCAL_FREE",
        "privacy_execution_authority": g06,
        "truth_boundary": dict(ZERO),
    }
    seal(envelope, "evidence_identity_sha256")
    return g05, envelope


def preflight(records, g05, envelope, actions, statuses, *, terminal=True):
    inventory = materialize_record_inventory(records)
    drop = []
    redact = []
    unchanged = []
    unchanged_bytes = 0
    for record in records:
        rid = record["record_id"]
        if statuses[rid] == "REJECT_DOCUMENT" or actions[rid] in {"QUARANTINE", "EXCLUDE"}:
            drop.append(rid)
        elif statuses[rid] == "RETAIN_PARTIAL":
            pass
        elif actions[rid] == "REDACT":
            redact.append(rid)
        else:
            unchanged.append(rid)
            unchanged_bytes += len(record["normalized_payload"].encode())
    blockers = []
    if not terminal:
        blockers.append("G06_EXACT_BYTE_EXECUTION_AUTHORITY_NOT_TERMINAL")
    if drop:
        blockers.append("POST_G05_G06_EXCLUSION_MATERIALIZATION_REQUIRED")
    if redact:
        blockers.append("G06_REDACTION_MATERIALIZATION_REQUIRED")
    core = {
        "schema": "12-6.d03-current-g05-g06-composition-preflight.v1",
        "status": "BLOCKED_CURRENT_G05_G06_COMPOSITION",
        "g06_exact_byte_execution_terminal": terminal,
        "g05_execution_identity_sha256": g05["execution_identity_sha256"],
        "g06_envelope_identity_sha256": envelope["evidence_identity_sha256"],
        "g06_execution_identity_sha256": (
            envelope["privacy_execution_authority"]["execution_identity_sha256"]
        ),
        "input_record_count": len(records),
        "input_utf8_bytes": sum(len(r["normalized_payload"].encode()) for r in records),
        "survivor_jsonl_sha256": hashlib.sha256(canonical_record_bytes(records)).hexdigest(),
        "survivor_record_inventory_sha256": inventory["record_inventory_digest_sha256"],
        "survivor_payload_inventory_sha256": inventory["payload_inventory_digest_sha256"],
        "drop_record_count": len(drop),
        "g05_partial_record_count": sum(
            statuses[row["record_id"]] == "RETAIN_PARTIAL" for row in records
        ),
        "g06_redaction_record_count": len(redact),
        "unchanged_allow_record_count": len(unchanged),
        "unchanged_allow_input_utf8_bytes": unchanged_bytes,
        "drop_record_id_sha256": sorted(hashlib.sha256(v.encode()).hexdigest() for v in drop),
        "g06_redaction_record_id_sha256": sorted(
            hashlib.sha256(value.encode()).hexdigest() for value in redact
        ),
        "unchanged_allow_record_id_sha256": sorted(
            hashlib.sha256(value.encode()).hexdigest() for value in unchanged
        ),
        "blockers": sorted(blockers),
        "truth_boundary": dict(ZERO),
    }
    return seal(core, "composition_preflight_identity_sha256")


def setup_case(tmp_path: Path, *, reject_redacted: bool = False, terminal: bool = True):
    records = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f",
            "modality": "en",
            "normalized_payload": "clean text",
        },
        {
            "record_id": "r2",
            "source_id": "s2",
            "family": "f",
            "modality": "en",
            "normalized_payload": "mail person@example.org now",
        },
        {
            "record_id": "r3",
            "source_id": "s3",
            "family": "f",
            "modality": "en",
            "normalized_payload": "drop this",
        },
    ]
    actions = {"r1": "ALLOW", "r2": "REDACT", "r3": "QUARANTINE"}
    statuses = {rid: "RETAIN_ALL" for rid in actions}
    source = privacy_source(reject_redacted=reject_redacted)
    path = tmp_path / "privacy_filter_v3.py"
    path.write_bytes(source)
    binding = privacy_binding(source)
    g05, envelope = authorities(records, actions, statuses, binding)
    pf = preflight(records, g05, envelope, actions, statuses, terminal=terminal)
    kwargs = {
        "records": records,
        "composition_preflight": pf,
        "expected_composition_preflight_identity_sha256": (
            pf["composition_preflight_identity_sha256"]
        ),
        "g05_authority": g05,
        "expected_g05_execution_identity_sha256": g05["execution_identity_sha256"],
        "g06_execution_envelope": envelope,
        "expected_g06_envelope_identity_sha256": envelope["evidence_identity_sha256"],
        "expected_g06_execution_identity_sha256": (
            envelope["privacy_execution_authority"]["execution_identity_sha256"]
        ),
        "privacy_source_path": path,
        "execution_head_sha": HEAD,
        "expected_materializer_implementation_git_blob_sha1": running_materializer_blob(),
    }
    return kwargs


def test_materializes_drop_redact_and_unchanged_deterministically(tmp_path):
    kwargs = setup_case(tmp_path)
    out_a, inventory_a, evidence_a = materialize_post_g05_g06(**kwargs)
    out_b, inventory_b, evidence_b = materialize_post_g05_g06(**kwargs)
    assert out_a == out_b
    assert inventory_a == inventory_b
    assert evidence_a == evidence_b
    assert [row["record_id"] for row in out_a] == ["r1", "r2"]
    assert out_a[0]["normalized_payload"] == "clean text"
    assert out_a[1]["normalized_payload"] == "mail <redacted> now"
    assert evidence_a["transform"]["drop_record_count"] == 1
    assert evidence_a["transform"]["redact_record_count"] == 1
    assert (
        evidence_a["materializer_implementation_git_blob_sha1"]
        == kwargs["expected_materializer_implementation_git_blob_sha1"]
    )
    assert evidence_a["input"]["privacy_binding"] == (
        kwargs["g06_execution_envelope"]["privacy_execution_authority"]["privacy_binding"]
    )
    assert evidence_a["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert evidence_a["truth_boundary"]["training_executed"] is False
    assert evidence_a["result"]["record_payload_jsonl_sha256"] == hashlib.sha256(
        canonical_record_bytes(out_a)
    ).hexdigest()


def test_nonterminal_g06_fails_closed(tmp_path):
    kwargs = setup_case(tmp_path, terminal=False)
    with pytest.raises(PostG05G06MaterializationError, match="not terminal"):
        materialize_post_g05_g06(**kwargs)


def test_payload_substitution_fails_closed(tmp_path):
    kwargs = setup_case(tmp_path)
    kwargs["records"][0] = {**kwargs["records"][0], "normalized_payload": "dirty text"}
    with pytest.raises(PostG05G06MaterializationError, match="survivor JSONL identity drift"):
        materialize_post_g05_g06(**kwargs)


def test_g06_authenticated_privacy_blob_controls_runtime(tmp_path):
    kwargs = setup_case(tmp_path)
    alternate = privacy_source(reject_redacted=True)
    kwargs["privacy_source_path"].write_bytes(alternate)
    with pytest.raises(PostG05G06MaterializationError, match="privacy source Git blob drift"):
        materialize_post_g05_g06(**kwargs)


def test_materializer_implementation_substitution_fails_before_transform(tmp_path):
    kwargs = setup_case(tmp_path)
    kwargs["expected_materializer_implementation_git_blob_sha1"] = "c" * 40
    with pytest.raises(
        PostG05G06MaterializationError,
        match="materializer implementation Git blob drift",
    ):
        materialize_post_g05_g06(**kwargs)


def test_privacy_binding_unknown_field_fails_closed(tmp_path):
    kwargs = setup_case(tmp_path)
    envelope = kwargs["g06_execution_envelope"]
    g06 = envelope["privacy_execution_authority"]
    g06["privacy_binding"]["unexpected"] = "x"
    seal(g06, "execution_identity_sha256")
    seal(envelope, "evidence_identity_sha256")
    kwargs["expected_g06_execution_identity_sha256"] = g06["execution_identity_sha256"]
    kwargs["expected_g06_envelope_identity_sha256"] = envelope["evidence_identity_sha256"]
    actions = {"r1": "ALLOW", "r2": "REDACT", "r3": "QUARANTINE"}
    statuses = {rid: "RETAIN_ALL" for rid in actions}
    pf = preflight(kwargs["records"], kwargs["g05_authority"], envelope, actions, statuses)
    kwargs["composition_preflight"] = pf
    kwargs["expected_composition_preflight_identity_sha256"] = (
        pf["composition_preflight_identity_sha256"]
    )
    with pytest.raises(PostG05G06MaterializationError, match="privacy_binding schema drift"):
        materialize_post_g05_g06(**kwargs)


def test_redacted_output_must_rescan_allow(tmp_path):
    kwargs = setup_case(tmp_path, reject_redacted=True)
    with pytest.raises(PostG05G06MaterializationError, match="not privacy-clean"):
        materialize_post_g05_g06(**kwargs)


def test_partial_g05_has_no_materialization_authority(tmp_path):
    kwargs = setup_case(tmp_path)
    records = kwargs["records"]
    actions = {"r1": "ALLOW", "r2": "REDACT", "r3": "QUARANTINE"}
    statuses = {"r1": "RETAIN_PARTIAL", "r2": "RETAIN_ALL", "r3": "RETAIN_ALL"}
    binding = kwargs["g06_execution_envelope"]["privacy_execution_authority"][
        "privacy_binding"
    ]
    g05, envelope = authorities(records, actions, statuses, binding)
    pf = preflight(records, g05, envelope, actions, statuses)
    kwargs.update(
        {
            "composition_preflight": pf,
            "expected_composition_preflight_identity_sha256": (
                pf["composition_preflight_identity_sha256"]
            ),
            "g05_authority": g05,
            "expected_g05_execution_identity_sha256": g05["execution_identity_sha256"],
            "g06_execution_envelope": envelope,
            "expected_g06_envelope_identity_sha256": envelope["evidence_identity_sha256"],
            "expected_g06_execution_identity_sha256": (
                envelope["privacy_execution_authority"]["execution_identity_sha256"]
            ),
        }
    )
    with pytest.raises(PostG05G06MaterializationError, match="partial materialization"):
        materialize_post_g05_g06(**kwargs)
