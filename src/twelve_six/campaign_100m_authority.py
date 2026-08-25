"""Final CAMPAIGN-47 launch authority composition, including evaluation freeze identity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from twelve_six.campaign_100m import qualify_main_launch


def _seal(report: dict[str, Any]) -> dict[str, Any]:
    report.pop("report_sha256", None)
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def _evaluation_freeze_passes(
    *,
    source_sha: str,
    evaluation_freeze: Mapping[str, Any],
    tokenizer_freeze: Mapping[str, Any],
    corpus_freeze: Mapping[str, Any],
) -> bool:
    return (
        evaluation_freeze.get("schema") == "12-6.evaluation-freeze.v1"
        and evaluation_freeze.get("status") == "FROZEN"
        and evaluation_freeze.get("source_sha") == source_sha
        and evaluation_freeze.get("training_use") is False
        and evaluation_freeze.get("random_init_control") is True
        and bool(evaluation_freeze.get("validation_manifest_sha256"))
        and bool(evaluation_freeze.get("test_manifest_sha256"))
        and bool(evaluation_freeze.get("contamination_registry_sha256"))
        and bool(evaluation_freeze.get("protocol_sha256"))
        and bool(evaluation_freeze.get("capability_registry_sha256"))
        and evaluation_freeze.get("tokenizer_artifact_sha256")
        == tokenizer_freeze.get("artifact_sha256")
        and evaluation_freeze.get("corpus_manifest_sha256")
        == corpus_freeze.get("manifest_sha256")
    )


def qualify_campaign_main_launch(
    *,
    source_sha: str,
    variant_name: str,
    s2_evidence: Mapping[str, Any],
    s3_evidence: Mapping[str, Any],
    s4_preflight: Mapping[str, Any],
    gpu_pilot: Mapping[str, Any],
    tokenizer_freeze: Mapping[str, Any],
    corpus_freeze: Mapping[str, Any],
    evaluation_freeze: Mapping[str, Any],
    paid_compute_authorized: bool,
) -> dict[str, Any]:
    """Evaluate every technical identity gate; this function never launches compute."""

    report = qualify_main_launch(
        source_sha=source_sha,
        variant_name=variant_name,
        s2_evidence=s2_evidence,
        s3_evidence=s3_evidence,
        s4_preflight=s4_preflight,
        gpu_pilot=gpu_pilot,
        tokenizer_freeze=tokenizer_freeze,
        corpus_freeze=corpus_freeze,
        paid_compute_authorized=paid_compute_authorized,
    )
    checks = report["checks"]
    checks["evaluation_registry_frozen"] = _evaluation_freeze_passes(
        source_sha=source_sha,
        evaluation_freeze=evaluation_freeze,
        tokenizer_freeze=tokenizer_freeze,
        corpus_freeze=corpus_freeze,
    )
    technical_ready = all(
        value for key, value in checks.items() if key != "paid_compute_explicitly_authorized"
    )
    launch_ready = technical_ready and paid_compute_authorized
    report["technical_ready"] = technical_ready
    report["main_launch_ready"] = launch_ready
    report["action"] = (
        "PREPARED_FOR_EXPLICIT_PAYMENT_LAUNCH"
        if launch_ready
        else "BLOCKED_NO_PAYMENT_LAUNCH"
    )
    return _seal(report)
