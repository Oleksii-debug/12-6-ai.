#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/data/next100_062_license_compliance_manifest_v1.json"
BASE_REGISTRY = ROOT / "data/registry/external_snapshots.v2.json"

EXPECTED_KEYS = [
    "code.encode.httpx._content",
    "code.psf.requests._internal_utils",
    "en.mdn.webdocs.prose.http-compression",
    "en.python.docs.tutorial-introduction",
    "en.rust-lang.book.prose.ch10-ch16",
    "en.standardebooks.manual.8-typography",
    "en.standardebooks.manual.9-metadata",
    "en.usgov.nist.technical-series",
    "en.wikisource.varieties-1902.pages20-22",
    "github:Kludex/starlette",
    "github:pydantic/pydantic",
    "ua.kmu.portal.secretariat-news",
    "ua.rada.open-data.laws-texts.d23314",
    "ua.verba.public-domain.nomis1864",
    "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13",
]

BASE_EXPECTED = {
    "code.encode.httpx._content": ("BSD-3-Clause", "2c61b3ac94d1dcebcde0c6f519554d2d7917247fbaa0a97002db4ef69e70ff28"),
    "code.psf.requests._internal_utils": ("Apache-2.0", "4c7d8d132c9898fc7d715e473f3ac74785ddc4ab96d2c9240f87835dc6d981ff"),
    "en.standardebooks.manual.8-typography": ("CC0-1.0", "21582c7f0e4ad39f2b0ed97bbc2c082d275e898b7a63c28e6d9badb8ee0f7860"),
    "en.standardebooks.manual.9-metadata": ("CC0-1.0", "7ac53dfb4bf6f73f178560e09f33160d0250c69fb679802f3254dc0eb4c9f509"),
    "ua.rada.open-data.laws-texts.d23314": ("RADA-OPEN-DATA-REUSE", "36eae31c3b0676ea7c02236fa05bd695c240c9a8eade5febc00457b8103ee1a4"),
}

SUCCESSOR_HEADS = {
    "NEXT100-022": "84c51e42b6daa51796fd20d793b5ef1ff01cc9d2",
    "NEXT100-026": "40950a950b60921fd856af2719e1ae2486d9e892",
    "NEXT100-027": "d75edd497c7fb1054e86d892c9462f059c1f4aa9",
    "NEXT100-032": "838a7687712fb3ed0c2c41bd259f77e4e0a451c9",
    "NEXT100-034": "b7491745b34ac8679baaf69cb96cd609dcbe0a16",
    "NEXT100-037": "5a6a495a24bce449334cbc5126d0114f61a9f57c",
    "NEXT100-038": "902eccc0b3efff09a38dc89cda789180b6c6e754",
    "NEXT100-039": "4de173535bacda145a6c1b598b80715d119175f6",
    "NEXT100-045": "c6756b5ebb6eb1d3bf3de2499167833d99d99a72",
    "NEXT100-048": "ca1755886f052d272029d6d68b2f1b7f02187936",
}

REQUIRED_FIELDS = {
    "source", "authority", "license", "copyright_notice", "attribution", "NOTICE",
    "share_alike", "redistribution_requirements", "project_purpose_authorization",
    "required_accompanying_files_text", "compliance_status", "missing_obligations",
}


class ComplianceError(RuntimeError):
    pass


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _key(source: str) -> str:
    return source.split("@", 1)[0]


def canonical_digest(manifest: dict) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_manifest() -> dict:
    m = _load(MANIFEST)
    if m.get("schema_version") != "12-6.next100-062-license-compliance-manifest.v2":
        raise ComplianceError("schema drift")
    if m.get("worker_id") != "NEXT100-062-LICENSE-COMPLIANCE-MANIFEST":
        raise ComplianceError("worker drift")
    if m.get("local_free_only") is not True:
        raise ComplianceError("LOCAL_FREE boundary weakened")
    if m.get("authority_boundary") != "LICENSE_COMPLIANCE_ONLY_NO_RIGHTS_REINTERPRETATION_NO_CORPUS_ADMISSION":
        raise ComplianceError("authority boundary weakened")
    if m.get("terminality_rule") != "INCLUDE_SOURCE_AUTHORITIES_THAT_PUBLISH_AN_UNCONDITIONAL_TERMINAL_OR_ADMIT_TRAINING_VERDICT; EXCLUDE_CANDIDATES_WHOSE_OWN_STATED_TERMINAL_PRECONDITION_REMAINS_UNSATISFIED":
        raise ComplianceError("terminality rule drift")

    baseline = m.get("baseline", {})
    if baseline.get("DATA-287", {}).get("head") != "b0523ccbc4b957615aac849d476cfa851be87578":
        raise ComplianceError("DATA-287 head drift")
    if baseline.get("DATA-287", {}).get("registry_identity_sha256") != "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c":
        raise ComplianceError("DATA-287 registry identity drift")
    if baseline.get("DATA-293", {}).get("head") != "2665a4cd86186ba44024334719d30dcc35d222d8":
        raise ComplianceError("DATA-293 head drift")

    successors = m.get("terminal_successors", {})
    if set(successors) != set(SUCCESSOR_HEADS):
        raise ComplianceError("terminal successor set drift")
    for worker, head in SUCCESSOR_HEADS.items():
        if successors[worker].get("head") != head or not successors[worker].get("basis"):
            raise ComplianceError(f"terminal successor binding drift: {worker}")

    sources = m.get("sources")
    if not isinstance(sources, list) or m.get("source_count") != len(sources):
        raise ComplianceError("source count mismatch")
    keys = [_key(str(row.get("source", ""))) for row in sources]
    if keys != EXPECTED_KEYS or len(keys) != len(set(keys)):
        raise ComplianceError("terminal source set/order drift")

    base = _load(BASE_REGISTRY)
    if base.get("registry_identity_sha256") != "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c":
        raise ComplianceError("bound base registry identity changed")
    base_by_id = {row["source_id"]: row for row in base.get("sources", [])}
    if set(base_by_id) != set(BASE_EXPECTED):
        raise ComplianceError("DATA-287 source set changed")
    by_key = {key: row for key, row in zip(keys, sources)}
    for sid, (license_id, raw_sha) in BASE_EXPECTED.items():
        row = by_key[sid]
        upstream = base_by_id[sid]
        if license_id not in row["license"] or raw_sha not in row["source"]:
            raise ComplianceError(f"baseline license/source identity drift: {sid}")
        if upstream["license"]["license_id"] != license_id or upstream["snapshot"]["raw_sha256"] != raw_sha:
            raise ComplianceError(f"DATA-287 mismatch: {sid}")
        if upstream["rights"]["model_training"]["status"] != "ALLOWED":
            raise ComplianceError(f"baseline training verdict changed: {sid}")
        if upstream["rights"]["evaluation"]["status"] != "NOT_SEPARATELY_ADMITTED":
            raise ComplianceError(f"baseline evaluation boundary changed: {sid}")

    blocked = []
    for key, row in zip(keys, sources):
        miss = sorted(REQUIRED_FIELDS - row.keys())
        if miss:
            raise ComplianceError(f"{key}: missing compliance fields {miss}")
        purposes = row["project_purpose_authorization"]
        for required_purpose in ("training", "redistribution", "evaluation", "tokenizer_fitting", "selection_validation", "final_test"):
            if required_purpose not in purposes:
                raise ComplianceError(f"{key}: missing purpose decision {required_purpose}")
        if not str(purposes["training"]).startswith("ALLOWED"):
            raise ComplianceError(f"{key}: terminal training verdict not preserved")
        if str(purposes["evaluation"]).startswith("ALLOWED"):
            raise ComplianceError(f"{key}: training license leaked into evaluation")
        if key == "ua.verba.public-domain.nomis1864":
            if purposes["tokenizer_fitting"] != "ALLOWED":
                raise ComplianceError("Nomis explicit tokenizer authority was lost")
        elif str(purposes["tokenizer_fitting"]).startswith("ALLOWED"):
            raise ComplianceError(f"{key}: tokenizer permission inferred without source authority")
        if str(purposes["selection_validation"]).startswith("ALLOWED") or str(purposes["final_test"]).startswith("ALLOWED"):
            raise ComplianceError(f"{key}: protected evaluation purpose was inferred")

        for item in row["required_accompanying_files_text"]:
            if item.get("materialized") is True:
                path = ROOT / item["path"]
                if not path.is_file():
                    raise ComplianceError(f"{key}: required sidecar missing: {item['path']}")
                if _git_blob_sha1(path) != item["git_blob_sha1"]:
                    raise ComplianceError(f"{key}: required sidecar drift: {item['path']}")
            elif item.get("materialized") is False:
                for field in ("repository", "commit", "path", "git_blob_sha1"):
                    if not item.get(field):
                        raise ComplianceError(f"{key}: unmaterialized obligation lacks immutable {field}")
                if not row["missing_obligations"]:
                    raise ComplianceError(f"{key}: unmaterialized required file did not fail closed")
            else:
                raise ComplianceError(f"{key}: sidecar materialization state missing")

        obligations = row["missing_obligations"]
        if obligations:
            blocked.append(key)
            if "BLOCKED" not in row["compliance_status"]:
                raise ComplianceError(f"{key}: missing obligation not fail-closed")
        elif "BLOCKED" in row["compliance_status"]:
            raise ComplianceError(f"{key}: blocked without named obligation")
        if "REQUIRED" in row["share_alike"] and row["share_alike"] != "NOT_REQUIRED" and not obligations:
            raise ComplianceError(f"{key}: ShareAlike obligation lacks an explicit packaging decision")

    expected_blocked = {
        "en.mdn.webdocs.prose.http-compression",
        "en.python.docs.tutorial-introduction",
        "en.usgov.nist.technical-series",
        "en.wikisource.varieties-1902.pages20-22",
        "ua.wikisource.lesia-ukrainka.na-krylah-pisen.1892.page13",
    }
    if set(blocked) != expected_blocked:
        raise ComplianceError(f"blocked-source set drift: {sorted(blocked)}")
    if m.get("redistribution_release_status") != "BLOCKED_UNRESOLVED_SOURCE_OBLIGATIONS":
        raise ComplianceError("redistribution release must remain fail-closed")

    retained = {
        "data/external/rights-evidence/data181/rada-open-data-terms-20260826.txt": "8b965544ad25538806f8293b4a3bb499314114e4",
        "data/external/rights-evidence/data181/standardebooks-manual-license-d1143a9.txt": "ecc3ab7a2a7d726cc225b51a0c85809a7b0274cb",
    }
    for rel, blob in retained.items():
        path = ROOT / rel
        if not path.is_file() or _git_blob_sha1(path) != blob:
            raise ComplianceError(f"retained baseline rights evidence drift: {rel}")
    return m


def assert_training_authorized() -> None:
    m = validate_manifest()
    for row in m["sources"]:
        if not row["project_purpose_authorization"]["training"].startswith("ALLOWED"):
            raise ComplianceError(f"training blocked: {_key(row['source'])}")


def assert_redistributable() -> None:
    m = validate_manifest()
    blocked = [_key(row["source"]) for row in m["sources"] if row["missing_obligations"]]
    if blocked or m["redistribution_release_status"] != "READY":
        raise ComplianceError("redistribution fail-closed; unresolved obligations: " + ",".join(blocked))


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "validate-manifest"
    try:
        manifest = validate_manifest()
        if cmd == "validate-manifest":
            print("PASS", canonical_digest(manifest))
        elif cmd == "check-training":
            assert_training_authorized()
            print("PASS training-purpose source compliance")
        elif cmd == "check-redistribution":
            assert_redistributable()
            print("PASS redistribution compliance")
        elif cmd == "digest":
            print(canonical_digest(manifest))
        else:
            raise ComplianceError(f"unknown command: {cmd}")
    except ComplianceError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
