#!/usr/bin/env python3
"""Rematerialize the exact terminal Project Gutenberg payload from merged authority.

The durable terminal seal is authority. This tool only reconstructs the three exact
normalized bodies named by that seal so they can enter the incumbent global-dedup
pipeline. It creates no corpus, tokenizer, unique-loss, or training authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEAL = ROOT / "configs/data/next100_107_gutenberg_terminal_seal_v1.json"
SEAL_VALIDATOR = ROOT / "tools/validate_next100_107_gutenberg_terminal_seal.py"

SCHEMA = "12-6.d03-gutenberg-terminal-payload-rematerialization.v1"
WORKER_ID = "SWARM-1778-D03-GUTENBERG-TERMINAL-PAYLOAD"
EXPECTED_SEAL_BLOB_SHA1 = "a4b2a2d20653fe91ea8520cd947e2f0bf54268b2"
EXPECTED_AUTHORITY = (
    "1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b"
)
EXPECTED_FAMILY = "en.project-gutenberg.public-domain-books"
EXPECTED_FAMILY_IDENTITY = (
    "5798a9625d3b7f111de28851f7ef57d2a4f60d7b508987d24ce5905f0093fa12"
)
EXPECTED_NORMALIZED_TOTAL = 1_672_110
NORMALIZER_ID = "NEXT100_033_PG_BODY_NFC_LF_V1"

EXPECTED_RECORDS: dict[str, dict[str, Any]] = {
    "en.project-gutenberg.37177": {
        "ebook_id": 37177,
        "encoding": "ascii",
        "raw_bytes": 87_742,
        "raw_sha256": "cb88e93e292a743cdc7476f9cd8ea12ac85b0896023d5b6814bd82779f3faf96",
        "normalized_utf8_bytes": 66_612,
        "normalized_sha256": (
            "533f768fa90bd6fac90951760f12b1b70777ae3c147dbb053deb86ddf04860ba"
        ),
        "source_manifest_sha256": (
            "d543a637dc2db4143b9da7dbbbcd1eb70b9ab1ca0c5b3dcd7e8210bf12747daa"
        ),
        "transport_repo": (
            "GITenberg/"
            "Ludvig-Holberg-The-Founder-of-Norwegian-Literature-and-an-Oxford-Student_37177"
        ),
        "transport_commit": "6ba1ee491cfba2aa56729967b48886bf71b20ac7",
        "transport_path": "37177.txt",
        "transport_git_blob_sha1": "cc35bab195c6d55b14568cd3da8fecfd0499f868",
    },
    "en.project-gutenberg.37985": {
        "ebook_id": 37985,
        "encoding": "utf-8-sig",
        "raw_bytes": 1_203_657,
        "raw_sha256": "e356cd7d2e57677d48413768956fa9b517ed8f28d9ff8ff63b3651cfa0893066",
        "normalized_utf8_bytes": 1_158_509,
        "normalized_sha256": (
            "6c4cd7d1fa4ef2340fe94a3ec737c72a55658ea3fb2bb19e5f5859eb8d8e3810"
        ),
        "source_manifest_sha256": (
            "137e9330f1562a7da99f2530dc5f1f0736ab7a16e25d854d24784225d16db504"
        ),
        "transport_repo": "GITenberg/A-Literary-History-of-the-Arabs_37985",
        "transport_commit": "b8477090720ab89858bfed937f799760cb4f23e3",
        "transport_path": "37985-0.txt",
        "transport_git_blob_sha1": "cd62e73adf6d502823c4b59de304212cc3d63361",
    },
    "en.project-gutenberg.40652": {
        "ebook_id": 40652,
        "encoding": "iso-8859-1",
        "raw_bytes": 482_371,
        "raw_sha256": "654ca36825a6bc0c80f7126983b53501d1087047b28d889affefb710eb2820c8",
        "normalized_utf8_bytes": 446_989,
        "normalized_sha256": (
            "e0e7095ba7ac0f1478d18feb4d87be1ed21be37e472fa4203bb92b43f9e593b4"
        ),
        "source_manifest_sha256": (
            "919c07512ba94d7f8c13bff0434dd5c98e3560ce1d52a03fabc041c90510c91c"
        ),
        "transport_repo": (
            "GITenberg/A-Guide-to-the-Scientific-Knowledge-of-Things-Familiar_40652"
        ),
        "transport_commit": "4d69c9e1daad0951f2baecbae2709e72d8d7d53f",
        "transport_path": "40652-8.txt",
        "transport_git_blob_sha1": "52c517db02e5e5a7f5bfef806cf98dd8c92e06e1",
    },
}

START_RE = re.compile(r"^\*\*\* START OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")
END_RE = re.compile(r"^\*\*\* END OF .*PROJECT GUTENBERG EBOOK.*\*\*\*$")


class MaterializationError(RuntimeError):
    """Fail-closed terminal payload materialization error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MaterializationError(message)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    framed = b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    return hashlib.sha1(framed).hexdigest()


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _load_terminal_validator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "_gutenberg_terminal_seal_validator",
        SEAL_VALIDATOR,
    )
    _require(spec is not None and spec.loader is not None, "cannot load terminal seal validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_terminal_seal(path: Path) -> dict[str, Any]:
    """Load and independently pin the exact merged Gutenberg terminal authority."""
    raw = path.read_bytes()
    _require(
        _git_blob_sha1(raw) == EXPECTED_SEAL_BLOB_SHA1,
        "terminal seal Git blob identity drift",
    )
    validator = _load_terminal_validator()
    try:
        result = validator.validate(path)
    except SystemExit as exc:
        raise MaterializationError(str(exc)) from exc
    _require(result["status"] == "PASS", "terminal seal validator did not PASS")

    seal = json.loads(raw)
    _require(
        seal.get("authority_identity_sha256") == EXPECTED_AUTHORITY,
        "terminal authority identity drift",
    )
    family = seal.get("source_family", {})
    _require(family.get("family_id") == EXPECTED_FAMILY, "source family drift")
    _require(
        family.get("family_identity_sha256") == EXPECTED_FAMILY_IDENTITY,
        "source family identity drift",
    )
    _require(family.get("language") == "en", "source language drift")
    _require(family.get("modality") == "text", "source modality drift")
    _require(family.get("independent_family_credit") == 1, "family credit drift")

    rights = seal.get("rights", {})
    _require(
        rights.get("model_training") == "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
        "training-purpose rights drift",
    )
    _require(rights.get("evaluation") == "NOT_AUTHORIZED", "evaluation grant drift")
    _require(
        rights.get("redistribution") == "RIGHTS_CLEARED_JURISDICTIONS_ONLY",
        "redistribution boundary drift",
    )
    _require(
        rights.get("worldwide_public_domain_claim") is False,
        "worldwide public-domain overclaim",
    )

    records = seal.get("records", [])
    _require(len(records) == len(EXPECTED_RECORDS), "record count drift")
    seen: set[str] = set()
    for record in records:
        source_id = record.get("source_id")
        _require(source_id in EXPECTED_RECORDS, f"unexpected source_id: {source_id}")
        _require(source_id not in seen, f"duplicate source_id: {source_id}")
        seen.add(source_id)
        expected = EXPECTED_RECORDS[source_id]
        for key, expected_value in expected.items():
            if key == "encoding":
                continue
            _require(
                record.get(key) == expected_value,
                f"{source_id} terminal identity drift: {key}",
            )
    _require(seen == set(EXPECTED_RECORDS), "terminal source set drift")
    return seal


def normalize_pg_body(raw: bytes, encoding: str) -> bytes:
    """Reproduce preregistered NEXT100_033_PG_BODY_NFC_LF_V1 exactly."""
    try:
        text = raw.decode(encoding)
    except (LookupError, UnicodeDecodeError) as exc:
        raise MaterializationError(
            f"decode failure under preregistered encoding {encoding}: {exc}"
        ) from exc

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    starts = [i for i, line in enumerate(lines) if START_RE.match(line.strip())]
    ends = [i for i, line in enumerate(lines) if END_RE.match(line.strip())]
    _require(len(starts) == 1, f"expected exactly one Gutenberg START marker, got {len(starts)}")
    _require(len(ends) == 1, f"expected exactly one Gutenberg END marker, got {len(ends)}")
    _require(ends[0] > starts[0], "Gutenberg END marker does not follow START marker")

    body_lines = lines[starts[0] + 1 : ends[0]]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and not body_lines[-1].strip():
        body_lines.pop()
    body = "\n".join(body_lines)
    if body.startswith("\ufeff"):
        body = body[1:]
    body = unicodedata.normalize("NFC", body)
    return (body + "\n").encode("utf-8")


def verify_and_normalize_record(record: Mapping[str, Any], raw: bytes) -> bytes:
    """Verify one pinned transport object and its exact normalized identity."""
    source_id = str(record["source_id"])
    _require(len(raw) == int(record["raw_bytes"]), f"{source_id} raw byte count drift")
    _require(_sha256(raw) == record["raw_sha256"], f"{source_id} raw SHA-256 drift")
    _require(
        _git_blob_sha1(raw) == record["transport_git_blob_sha1"],
        f"{source_id} transport Git blob drift",
    )
    normalized = normalize_pg_body(raw, str(record["encoding"]))
    _require(
        len(normalized) == int(record["normalized_utf8_bytes"]),
        f"{source_id} normalized byte count drift",
    )
    _require(
        _sha256(normalized) == record["normalized_sha256"],
        f"{source_id} normalized SHA-256 drift",
    )
    return normalized


def _transport_url(record: Mapping[str, Any]) -> str:
    repo = str(record["transport_repo"])
    commit = str(record["transport_commit"])
    path = str(record["transport_path"])
    _require(repo.startswith("GITenberg/"), "transport repository is not GITenberg")
    _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "transport commit drift")
    _require(path and not path.startswith("/") and ".." not in Path(path).parts, "unsafe path")
    quoted = urllib.parse.quote(path, safe="/")
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{quoted}"


def fetch_bytes(url: str, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "12-6-ai-SWARM-1778-gutenberg-rematerializer/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(2)
    raise MaterializationError(f"transport acquisition failed for {url}: {last_error}")


def _runtime_record(
    record: Mapping[str, Any], normalized: bytes, output_name: str
) -> dict[str, Any]:
    return {
        "source_id": record["source_id"],
        "source_family": EXPECTED_FAMILY,
        "stable_origin_id": EXPECTED_FAMILY,
        "stable_object_id": f"sha256:{record['normalized_sha256']}",
        "modality": "en",
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            "NEXT100-107 / PR#1130; historical real run 32998859164 "
            f"source={record['source_id']}"
        ),
        # These names intentionally match the incumbent DATA-298 / NEXT100-065
        # matcher row contract. They describe the exact bytes passed to the
        # matcher (the terminal normalized body), not the pre-normalization
        # GITenberg transport object retained separately below.
        "declared_capacity_bytes": len(normalized),
        "expected_raw_bytes": len(normalized),
        "expected_raw_sha256": _sha256(normalized),
        "acquisition_url": f"terminal-rematerialization://next100-107/{record['source_id']}",
        "origin_key": (
            f"project-gutenberg:{record['ebook_id']}:"
            f"normalized-sha256:{record['normalized_sha256']}"
        ),
        "normalizer_id": NORMALIZER_ID,
        "transport_repo": record["transport_repo"],
        "transport_commit": record["transport_commit"],
        "transport_path": record["transport_path"],
        "transport_git_blob_sha1": record["transport_git_blob_sha1"],
        "raw_bytes": record["raw_bytes"],
        "raw_sha256": record["raw_sha256"],
        "payload_file": output_name,
        "payload_utf8_bytes": len(normalized),
        "payload_sha256": _sha256(normalized),
    }


def _build_receipt(seal: Mapping[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    inventory = {
        "record_count": len(rows),
        "normalized_utf8_bytes": sum(int(row["payload_utf8_bytes"]) for row in rows),
        "records": rows,
    }
    inventory["inventory_identity_sha256"] = _sha256(_canonical(inventory))
    _require(inventory["record_count"] == 3, "runtime record count drift")
    _require(
        inventory["normalized_utf8_bytes"] == EXPECTED_NORMALIZED_TOTAL,
        "runtime normalized byte total drift",
    )

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": WORKER_ID,
        "execution_profile": "LOCAL_FREE",
        "decision": "EXACT_TERMINAL_SOURCE_PAYLOAD_REMATERIALIZED",
        "source_authority": {
            "seal_git_blob_sha1": EXPECTED_SEAL_BLOB_SHA1,
            "authority_identity_sha256": EXPECTED_AUTHORITY,
            "family_identity_sha256": EXPECTED_FAMILY_IDENTITY,
            "parent_execution_head": seal["parent_authority"]["head_sha"],
            "historical_workflow_run_id": seal["exact_head_execution"]["workflow_run_id"],
            "historical_artifact_id": seal["exact_head_execution"]["artifact_id"],
            "historical_artifact_digest": seal["exact_head_execution"]["artifact_digest"],
        },
        "source_family": {
            "family_id": EXPECTED_FAMILY,
            "independent_family_credit": 1,
            "language": "en",
            "modality": "text",
        },
        "payload_inventory": inventory,
        "rights_boundary": {
            "model_training": "ALLOWED_FOR_EXACT_ADMITTED_NORMALIZED_BODIES",
            "evaluation": "NOT_AUTHORIZED",
            "redistribution": "RIGHTS_CLEARED_JURISDICTIONS_ONLY",
            "worldwide_public_domain_claim": False,
        },
        "claim_boundary": {
            "canonical_corpus_capacity_credited_bytes": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    receipt = dict(core)
    receipt["receipt_sha256"] = _sha256(_canonical(core))
    return receipt


def materialize(
    seal_path: Path,
    out_root: Path,
    *,
    fetcher: Callable[[str], bytes] = fetch_bytes,
) -> dict[str, Any]:
    """Rematerialize exact payload and return a text-free deterministic receipt."""
    seal = load_terminal_seal(seal_path)
    _require(not out_root.exists(), f"refusing to overwrite existing output: {out_root}")
    payload_dir = out_root / "payload"
    payload_dir.mkdir(parents=True)

    encoding_by_source = {
        source_id: expected["encoding"] for source_id, expected in EXPECTED_RECORDS.items()
    }
    rows: list[dict[str, Any]] = []
    for record in seal["records"]:
        source_id = str(record["source_id"])
        runtime_record = dict(record)
        runtime_record["encoding"] = encoding_by_source[source_id]
        raw = fetcher(_transport_url(runtime_record))
        normalized = verify_and_normalize_record(runtime_record, raw)
        output_name = source_id.replace(".", "_") + ".txt"
        (payload_dir / output_name).write_bytes(normalized)
        rows.append(_runtime_record(runtime_record, normalized, f"payload/{output_name}"))

    rows.sort(key=lambda row: str(row["source_id"]))
    receipt = _build_receipt(seal, rows)
    (out_root / "receipt.json").write_bytes(_canonical(receipt))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", type=Path, default=DEFAULT_SEAL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    receipt = materialize(args.seal, args.out)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
