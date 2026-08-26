#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/next100_062_license_compliance_manifest_v1.json"
BASE_REGISTRY = ROOT / "data/registry/external_snapshots.v2.json"

EXPECTED_BASE = {
    "code.encode.httpx._content": ("BSD-3-Clause", "2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28"),
    "code.psf.requests._internal_utils": ("Apache-2.0", "4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff"),
    "en.standardebooks.manual.8-typography": ("CC0-1.0", "21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860"),
    "en.standardebooks.manual.9-metadata": ("CC0-1.0", "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509"),
    "ua.rada.open-data.laws-texts.d23314": ("RADA-OPEN-DATA-REUSE", "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4"),
}
EXPECTED_SUCCESSOR = {
    "source_id": "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13",
    "raw_sha256": "65e570c3cd954b595b586554b89a90da6efad0deca6a84d2316937745db17ef2",
    "head_sha": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
    "workflow_run": 32998002424,
    "authority_identity_sha256": "6b443faa7fef777214022028d5fdb356dae0ab1a9b71822b4e16bea8f92cd0d6",
}


class ComplianceError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def canonical_digest(manifest: dict) -> str:
    data = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def validate_manifest() -> dict:
    m = _load(MANIFEST)
    if m.get("schema_version") != "12-6.next100-062-license-compliance-manifest.v1":
        raise ComplianceError("schema drift")
    if m.get("worker_id") != "NEXT100-062-LICENSE-COMPLIANCE-MANIFEST":
        raise ComplianceError("worker drift")
    if m.get("local_free_only") is not True:
        raise ComplianceError("LOCAL_FREE boundary weakened")
    if m.get("terminality_rule") != "ONLY_EXACT_HEAD_DEDICATED_TERMINAL_SUCCESS_ADMIT_AUTHORITIES_ENTER_SOURCE_SET":
        raise ComplianceError("terminality rule weakened")

    vector = m.get("terminal_authority_vector", {})
    data287 = vector.get("DATA-287", {})
    if data287.get("head_sha") != "b0523ccbc4b957615aac849d476cfa851be87578":
        raise ComplianceError("DATA-287 head drift")
    if data287.get("registry_identity_sha256") != "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c":
        raise ComplianceError("DATA-287 identity drift")
    data293 = vector.get("DATA-293", {})
    if data293.get("head_sha") != "2665a4cd86186ba44024334719d30dcc35d222d8" or data293.get("dedicated_workflow_run") != 32967466790:
        raise ComplianceError("DATA-293 terminal authority drift")
    succ = vector.get("NEXT100-022", {})
    if succ.get("head_sha") != EXPECTED_SUCCESSOR["head_sha"] or succ.get("dedicated_workflow_run") != EXPECTED_SUCCESSOR["workflow_run"]:
        raise ComplianceError("NEXT100-022 terminal binding drift")
    if succ.get("authority_identity_sha256") != EXPECTED_SUCCESSOR["authority_identity_sha256"]:
        raise ComplianceError("NEXT100-022 authority identity drift")

    sources = m.get("sources")
    if not isinstance(sources, list) or m.get("source_count") != len(sources):
        raise ComplianceError("source count mismatch")
    ids = [s.get("source_id") for s in sources]
    if ids != sorted(ids) or len(ids) != len(set(ids)):
        raise ComplianceError("sources must be unique and deterministically sorted")
    expected_ids = sorted([*EXPECTED_BASE, EXPECTED_SUCCESSOR["source_id"]])
    if ids != expected_ids:
        raise ComplianceError("terminal source set drift")

    base = _load(BASE_REGISTRY)
    if base.get("registry_identity_sha256") != "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c":
        raise ComplianceError("bound base registry identity changed")
    base_by_id = {s["source_id"]: s for s in base.get("sources", [])}
    if set(base_by_id) != set(EXPECTED_BASE):
        raise ComplianceError("DATA-287 source set changed")

    by_id = {s["source_id"]: s for s in sources}
    for sid, (license_id, raw_sha) in EXPECTED_BASE.items():
        src = by_id[sid]
        upstream = base_by_id[sid]
        if src.get("license") != license_id or src.get("raw_sha256") != raw_sha:
            raise ComplianceError(f"license/source identity drift: {sid}")
        if upstream["license"]["license_id"] != license_id or upstream["snapshot"]["raw_sha256"] != raw_sha:
            raise ComplianceError(f"base registry mismatch: {sid}")
        if upstream["rights"]["model_training"]["status"] != "ALLOWED":
            raise ComplianceError(f"training verdict no longer allowed: {sid}")
        if upstream["rights"]["evaluation"]["status"] != "NOT_SEPARATELY_ADMITTED":
            raise ComplianceError(f"evaluation boundary changed: {sid}")

    ws = by_id[EXPECTED_SUCCESSOR["source_id"]]
    if ws.get("raw_sha256") != EXPECTED_SUCCESSOR["raw_sha256"]:
        raise ComplianceError("Wikisource exact source identity drift")

    required_fields = {
        "license", "copyright_notice", "attribution", "notice", "share_alike",
        "redistribution_requirements", "project_purpose_authorization",
        "required_accompanying_files_text", "compliance_status", "missing_obligations",
    }
    blocked_sources = 0
    for src in sources:
        missing_fields = sorted(required_fields - src.keys())
        if missing_fields:
            raise ComplianceError(f"{src['source_id']}: missing fields {missing_fields}")
        purposes = src["project_purpose_authorization"]
        if not str(purposes.get("model_training", "")).startswith("ALLOWED"):
            raise ComplianceError(f"{src['source_id']}: training verdict weakened or missing")
        if purposes.get("evaluation") != "NOT_SEPARATELY_ADMITTED":
            raise ComplianceError(f"{src['source_id']}: evaluation permission inferred")
        for p in ("tokenizer_fitting", "selection_validation", "final_test"):
            if purposes.get(p) != "NOT_SEPARATELY_AUTHORIZED_BY_THIS_MANIFEST":
                raise ComplianceError(f"{src['source_id']}: {p} inferred without authority")
        for sidecar in src["required_accompanying_files_text"]:
            path = ROOT / sidecar["path"]
            if not path.is_file():
                raise ComplianceError(f"{src['source_id']}: required sidecar missing: {sidecar['path']}")
            if _git_blob_sha1(path) != sidecar["git_blob_sha1"]:
                raise ComplianceError(f"{src['source_id']}: required sidecar drift: {sidecar['path']}")
        obligations = src["missing_obligations"]
        if obligations:
            blocked_sources += 1
            if "BLOCKED" not in src["compliance_status"]:
                raise ComplianceError(f"{src['source_id']}: missing obligations not fail-closed")
        elif "BLOCKED" in src["compliance_status"]:
            raise ComplianceError(f"{src['source_id']}: blocked without named missing obligation")
        if "REQUIRED" in src["share_alike"] and "UNRESOLVED" in src["share_alike"]:
            if not obligations:
                raise ComplianceError(f"{src['source_id']}: unresolved ShareAlike path not blocked")

    if blocked_sources:
        if not str(m.get("redistribution_release_status", "")).startswith("BLOCKED"):
            raise ComplianceError("manifest has blocked sources but redistribution release is not blocked")
    elif m.get("redistribution_release_status") != "READY":
        raise ComplianceError("no blocked source but release is not READY")

    # Retained evidence is part of the fail-closed compliance chain.
    evidence = {
        "data/external/rights-evidence/data181/rada-open-data-terms-20260826.txt": "8b965544ad25538806f8293b4a3bb499314114e4",
        "data/external/rights-evidence/data181/standardebooks-manual-license-d1143a9.txt": "ecc3ab7a2a7d726cc225b51a0c85809a7b0274cb",
    }
    for rel, blob in evidence.items():
        path = ROOT / rel
        if not path.is_file() or _git_blob_sha1(path) != blob:
            raise ComplianceError(f"retained rights evidence drift: {rel}")
    return m


def assert_training_authorized() -> None:
    m = validate_manifest()
    for src in m["sources"]:
        if not src["project_purpose_authorization"]["model_training"].startswith("ALLOWED"):
            raise ComplianceError(f"training blocked: {src['source_id']}")


def assert_redistributable() -> None:
    m = validate_manifest()
    if m["redistribution_release_status"] != "READY":
        blocked = [s["source_id"] for s in m["sources"] if s["missing_obligations"]]
        raise ComplianceError("redistribution fail-closed; unresolved obligations: " + ",".join(blocked))


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "validate-manifest"
    try:
        m = validate_manifest()
        if cmd == "validate-manifest":
            print("PASS", canonical_digest(m))
        elif cmd == "check-training":
            assert_training_authorized()
            print("PASS training-purpose source compliance")
        elif cmd == "check-redistribution":
            assert_redistributable()
            print("PASS redistribution compliance")
        elif cmd == "digest":
            print(canonical_digest(m))
        else:
            raise ComplianceError(f"unknown command: {cmd}")
    except ComplianceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
