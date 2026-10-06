#!/usr/bin/env python3
"""Fail-closed training-purpose rights/provenance recheck for current Rada data."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-rada-current-training-rights.v1"
EVIDENCE_SCHEMA = "12-6.d03-rada-current-training-rights-evidence.v1"
AUTHORITY_ID = "638a38a6185f280d3862da8fedf610a1c57801403b2bb7be042d66dce364e934"
REPLAY_BLOB = "9953940072fdd973cd6c83eb5899314d3484da30"
REPLAY_ID = "543f9cdd5a9aacaf2cc00b5d4057ad8b142aa685870545cff8bd518f8085055e"
RIGHTS_BLOB = "4a6cb0bd6b009ef36c9d2fb712967a4ae1cbfe0b"
RIGHTS_ID = "47e43afc87e798a52be1745d4313f353a349bc957e809126369a3923ffb68d0f"
DATA287_BLOB = "21fe6485aa9eaa0945837efce54c69cfc1098472"
DATA287_ID = "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
FAMILY = "ua.rada.open-data.laws-texts"
FAMILY_ID = "b8f1d2f99a3db71d894a3233e9417d6283d11768c41b1634bc8b096ab77aba4e"
ARCHIVE_SHA = "9f937b3283dd2d9508a0a92442fc007e7e6d373a17d3d50fa428e324cb2fa290"
ARCHIVE_BYTES = 46774786
ENTRY_ID = "1fcc222a959d1dfc24e2b23b71a5412b1050e22a5004cbd36f0dc79998898cc0"


class RightsRecheckError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def blob_sha(raw: bytes) -> str:
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw, usedforsecurity=False).hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RightsRecheckError(message)


def load(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RightsRecheckError(f"cannot load JSON: {path}") from exc
    require(isinstance(value, dict), f"JSON root must be object: {path}")
    return value, raw


def validate_config(config: Mapping[str, Any]) -> None:
    require(config.get("schema_version") == SCHEMA, "config schema drift")
    core = dict(config)
    identity = core.pop("authority_identity_sha256", None)
    require(identity == AUTHORITY_ID and digest(core) == AUTHORITY_ID, "authority identity drift")
    snapshot = config.get("current_snapshot")
    require(isinstance(snapshot, Mapping), "current snapshot missing")
    require(snapshot.get("parent_head_sha") == "fb49b7e212444547219df2bd2aa466db955d51e5", "parent head drift")
    require(snapshot.get("source_family") == FAMILY, "source family drift")
    require(snapshot.get("archive_sha256") == ARCHIVE_SHA and snapshot.get("archive_bytes") == ARCHIVE_BYTES, "snapshot archive drift")
    require(snapshot.get("entry_identity_sha256") == ENTRY_ID and snapshot.get("canonical_entry_count") == 3055, "snapshot inventory drift")
    prior = config.get("prior_family_authority")
    require(isinstance(prior, Mapping), "prior authority missing")
    require(prior.get("registry_git_blob_sha1") == DATA287_BLOB and prior.get("registry_identity_sha256") == DATA287_ID, "DATA-287 binding drift")
    require(prior.get("family_identity_sha256") == FAMILY_ID and prior.get("model_training_status") == "ALLOWED", "prior family training authority drift")
    incumbent = config.get("incumbent_rights_policy")
    require(isinstance(incumbent, Mapping), "incumbent policy missing")
    require(incumbent.get("git_blob_sha1") == RIGHTS_BLOB and incumbent.get("policy_identity_sha256") == RIGHTS_ID, "incumbent rights binding drift")
    decision = config.get("project_decision")
    require(isinstance(decision, Mapping), "project decision missing")
    require(decision.get("purpose_rights_recheck_complete_for_exact_snapshot") is True, "rights recheck decision drift")
    require(decision.get("model_training_purpose") == "ALLOWED_WITH_SOURCE_ATTRIBUTION_AFTER_REMAINING_DATA_GATES", "training-purpose decision drift")
    require(decision.get("evaluation") == "NOT_SEPARATELY_ADMITTED" and decision.get("final_test") == "PROHIBITED", "evaluation boundary drift")
    require(decision.get("legal_conclusion_claimed") is False and decision.get("attribution_required") is True, "legal/attribution boundary drift")
    require(decision.get("training_authorized_bytes") == 0 and decision.get("canonical_capacity_credited") == 0, "decision pre-credits data")
    truth = config.get("truth_boundary")
    require(isinstance(truth, Mapping), "truth boundary missing")
    zero = {
        "canonical_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    require(dict(truth) == zero, "truth boundary widened")


def validate_replay(value: Mapping[str, Any], raw: bytes) -> None:
    require(blob_sha(raw) == REPLAY_BLOB and value.get("authority_identity_sha256") == REPLAY_ID, "replay authority drift")
    source = value.get("source")
    require(isinstance(source, Mapping), "replay source missing")
    require(source.get("family") == FAMILY, "replay family drift")
    require(source.get("source_archive_sha256") == ARCHIVE_SHA and source.get("source_archive_bytes") == ARCHIVE_BYTES, "replay snapshot drift")
    require(source.get("entry_identity_sha256") == ENTRY_ID, "replay entry identity drift")
    require(source.get("rights_policy_git_blob_sha1") == RIGHTS_BLOB and source.get("rights_policy_identity_sha256") == RIGHTS_ID, "replay rights lineage drift")
    require(source.get("training_authority_granted") is False and source.get("downstream_rights_and_provenance_recheck_required") is True, "replay preclaim drift")


def validate_policy(value: Mapping[str, Any], raw: bytes) -> None:
    require(blob_sha(raw) == RIGHTS_BLOB and value.get("policy_identity_sha256") == RIGHTS_ID, "rights policy drift")
    source = value.get("source")
    require(isinstance(source, Mapping) and source.get("dataset_id") == "laws-texts" and source.get("source_family") == FAMILY, "rights source drift")
    evidence = value.get("primary_evidence")
    require(isinstance(evidence, list) and len(evidence) == 1 and isinstance(evidence[0], Mapping), "rights evidence drift")
    row = evidence[0]
    require(row.get("reuse_scope") == "FREE_USE_REUSE_REDISTRIBUTION_INCLUDING_COMMERCIAL", "reuse scope drift")
    require(row.get("condition") == "MANDATORY_SOURCE_ATTRIBUTION" and row.get("license") == "CC-BY-4.0_UNLESS_OTHERWISE_SPECIFIED", "license/attribution drift")


def validate_registry(value: Mapping[str, Any], raw: bytes) -> None:
    require(blob_sha(raw) == DATA287_BLOB and value.get("registry_identity_sha256") == DATA287_ID, "DATA-287 registry drift")
    rows = [r for r in value.get("sources", []) if isinstance(r, Mapping) and r.get("source_id") == "ua.rada.open-data.laws-texts.d23314"]
    require(len(rows) == 1, "DATA-287 Rada cardinality drift")
    row = rows[0]
    family = row.get("independent_source_family")
    rights = row.get("rights")
    require(isinstance(family, Mapping) and family.get("family_id") == FAMILY and family.get("family_identity_sha256") == FAMILY_ID, "DATA-287 family drift")
    require(isinstance(rights, Mapping), "DATA-287 rights missing")
    require(rights.get("model_training", {}).get("status") == "ALLOWED", "DATA-287 training authority not ALLOWED")
    require(rights.get("evaluation", {}).get("status") == "NOT_SEPARATELY_ADMITTED", "DATA-287 evaluation boundary drift")


def page_text(raw: bytes) -> str:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RightsRecheckError("portal page is not strict UTF-8") from exc
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def evaluate_portal(raw: bytes, markers: Mapping[str, Any]) -> dict[str, Any]:
    text = page_text(raw)
    checks = {}
    for key, marker in markers.items():
        require(isinstance(key, str) and isinstance(marker, str) and marker, "invalid portal marker")
        checks[key] = marker in text
    missing = sorted(key for key, ok in checks.items() if not ok)
    require(not missing, "official portal evidence missing: " + ",".join(missing))
    projection = {
        "dataset_id": "laws-texts",
        "source_family": FAMILY,
        "semantic_checks": checks,
        "training_purpose_interpretation": "ALLOWED_WITH_SOURCE_ATTRIBUTION_AFTER_REMAINING_DATA_GATES",
        "evaluation": "NOT_SEPARATELY_ADMITTED",
        "legal_conclusion_claimed": False,
    }
    return {"page_sha256": hashlib.sha256(raw).hexdigest(), "semantic_projection": projection, "semantic_identity_sha256": digest(projection)}


def fetch_portal(url: str, host: str, limit: int) -> bytes:
    parsed = urllib.parse.urlsplit(url)
    require(parsed.scheme == "https" and parsed.hostname == host, "portal URL outside allowed HTTPS host")
    request = urllib.request.Request(url, headers={"User-Agent": "12-6-ai-rights-recheck/1.0", "Accept": "text/html"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            final = urllib.parse.urlsplit(response.geturl())
            require(final.scheme == "https" and final.hostname == host, "portal redirect left allowed HTTPS host")
            raw = response.read(limit + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise RightsRecheckError("cannot fetch official portal evidence") from exc
    require(0 < len(raw) <= limit, "portal response size invalid")
    return raw


def build_evidence(config: Mapping[str, Any], portal: Mapping[str, Any] | None) -> dict[str, Any]:
    core = {
        "schema_version": EVIDENCE_SCHEMA,
        "authority_identity_sha256": AUTHORITY_ID,
        "exact_snapshot": {"source_family": FAMILY, "archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE_BYTES, "entry_identity_sha256": ENTRY_ID},
        "prior_family_authority": {"registry_identity_sha256": DATA287_ID, "family_identity_sha256": FAMILY_ID, "model_training_status": "ALLOWED", "evaluation_status": "NOT_SEPARATELY_ADMITTED"},
        "official_portal_semantic_evidence": portal,
        "project_decision": config["project_decision"],
        "downstream_required": config["downstream_required"],
        "truth_boundary": config["truth_boundary"],
    }
    return {**core, "evidence_identity_sha256": digest(core)}


def write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--replay-authority", type=Path, required=True)
    parser.add_argument("--rights-policy", type=Path, required=True)
    parser.add_argument("--prior-registry", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config, _ = load(args.config)
        replay, replay_raw = load(args.replay_authority)
        policy, policy_raw = load(args.rights_policy)
        registry, registry_raw = load(args.prior_registry)
        validate_config(config)
        validate_replay(replay, replay_raw)
        validate_policy(policy, policy_raw)
        validate_registry(registry, registry_raw)
        portal = None
        if args.live:
            spec = config["official_portal_recheck"]
            portal = evaluate_portal(fetch_portal(str(spec["url"]), str(spec["allowed_host"]), int(spec["max_response_bytes"])), spec["required_semantics"])
        evidence = build_evidence(config, portal)
        if args.output:
            write(args.output, evidence)
    except (RightsRecheckError, OSError, ValueError, TypeError) as exc:
        print(f"BLOCKED: {exc}")
        return 2
    print("D03_RADA_CURRENT_RIGHTS_RECHECK=PASS" if args.live else "D03_RADA_CURRENT_RIGHTS_RECHECK=STATIC_PASS")
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("TOKENIZER_FIT_AUTHORIZED=false")
    print("TRAINING_EXECUTED=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
