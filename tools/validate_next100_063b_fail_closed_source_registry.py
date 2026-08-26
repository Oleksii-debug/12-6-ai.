#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "configs/data/next100_063b_fail_closed_source_registry_v1.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_SCHEMA = "12-6.next100-063b-fail-closed-source-registry.v1"
EXPECTED_WORKER = "NEXT100-063B-FAIL-CLOSED-SOURCE-REGISTRY"
EXPECTED_IDENTITY = "d7c67df78325bc2640db27def5dd30695a93f97766141fca1a968b79fcad0eec"
EXPECTED_BASE_HEAD = "efc278cec0e4773eb4ff405bf4b4d24ee63b5d13"
EXPECTED_BASE_BLOB = "c1e05f09490e25f6fed765dfb70d900717528f4b"
EXPECTED_BASE_RUN = 32999969398
EXPECTED_BASE_CAPACITY = {"uk": 90044, "en": 84793, "code": 69133, "total": 243970}
EXPECTED_BASE_FAMILIES = {"uk": 2, "en": 1, "code": 4, "total": 7}
EXPECTED_LATE = {
    449: ("40950a950b60921fd856af2719e1ae2486d9e892", "1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9", 32997970539, "ua.kmu.portal.secretariat-news", "uk", 9153),
    462: ("d75edd497c7fb1054e86d892c9462f059c1f4aa9", "85f596e79b0ec6479d2ef815e2a6a9bdbfaa55993c797309c1ea4d93b1d9b0e7", 32998503672, "ua.verba.public-domain.nomis1864", "uk", 1659),
    472: ("b7491745b34ac8679baaf69cb96cd609dcbe0a16", "3ffba0fcd08ab42e940b2db12ffafb6f7234ad0bae6f7fe523071497485b9d1c", 32998703545, "en.usgov.nist.technical-series", "en", 59358),
    445: ("902eccc0b3efff09a38dc89cda789180b6c6e754", "0f5dbd5313f8196811e2a99f77eb8698c6bc69f69648d76a7e240ee9757ecc47", 32998544359, "en.mdn.webdocs.prose", "en", 6492),
}
EXPECTED_ZERO = {
    467: ("5a6a495a24bce449334cbc5126d0114f61a9f57c", 32998356906, "success", "python.cpython.documentation", "SOURCE_NORMALIZED_BYTES_NOT_ELIGIBLE_CAPACITY"),
    465: ("ca1755886f052d272029d6d68b2f1b7f02187936", 32999061340, "failure", "github:pydantic/pydantic", "DEDICATED_EXACT_HEAD_ADMISSION_WORKFLOW_FAILED"),
    475: ("78cada1d69b3f0c438012c4e6cf79143aae2f603", 32999511493, "failure", "github:Textualize/rich", "DEDICATED_EXACT_HEAD_ADMISSION_WORKFLOW_FAILED"),
}
EXPECTED_VECTOR = {"uk": 100856, "en": 150643, "code": 69133, "total": 320632}
EXPECTED_FAMILIES = {"uk": 4, "en": 3, "code": 4, "total": 11}
TARGET = 20_000_000


def fail(message: str) -> None:
    raise SystemExit(f"NEXT100-063B FAIL: {message}")


def main() -> None:
    data = json.loads(PATH.read_text(encoding="utf-8"))
    identity = data.pop("registry_identity_sha256", None)
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    actual = hashlib.sha256(canonical).hexdigest()
    if identity != actual or identity != EXPECTED_IDENTITY:
        fail("registry identity mismatch")
    if data.get("schema_version") != EXPECTED_SCHEMA or data.get("worker_id") != EXPECTED_WORKER:
        fail("schema/worker drift")
    if data.get("decision") != "CONSERVATIVE_TERMINAL_SOURCE_VECTOR_PRE_SUCCESSOR_GLOBAL_DEDUP":
        fail("decision boundary drift")
    for key in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        if data.get(key) is not False:
            fail(f"{key} must remain false")
    if data.get("local_free_only") is not True:
        fail("LOCAL_FREE boundary weakened")

    base = data["base_dedup_authority"]
    if base.get("head_sha") != EXPECTED_BASE_HEAD or base.get("inventory_blob_sha1") != EXPECTED_BASE_BLOB:
        fail("NEXT100-065 base identity drift")
    if base.get("dedicated_workflow_run") != EXPECTED_BASE_RUN or base.get("dedicated_workflow_conclusion") != "success":
        fail("NEXT100-065 exact-head workflow is not terminal-success bound")
    if base.get("capacity_bytes") != EXPECTED_BASE_CAPACITY or base.get("family_counts") != EXPECTED_BASE_FAMILIES:
        fail("NEXT100-065 base accounting drift")

    rows = data["credited_late_terminal_authorities"]
    if {row.get("pr") for row in rows} != set(EXPECTED_LATE):
        fail("credited late PR set drift")
    families = set()
    computed_capacity = dict(EXPECTED_BASE_CAPACITY)
    computed_families = dict(EXPECTED_BASE_FAMILIES)
    for row in rows:
        pr = row["pr"]
        expected = EXPECTED_LATE[pr]
        observed = (
            row.get("head_sha"), row.get("authority_identity"), row.get("dedicated_workflow_run"),
            row.get("family"), row.get("stratum"), row.get("eligible_capacity_bytes")
        )
        if observed != expected:
            fail(f"credited authority provenance drift for PR {pr}")
        if row.get("dedicated_workflow_conclusion") != "success":
            fail(f"credited PR {pr} lacks exact-head success")
        if row.get("family_credit") != 1:
            fail(f"credited PR {pr} must contribute exactly one family")
        if not HEX40.fullmatch(row["head_sha"]) or not HEX64.fullmatch(row["authority_identity"]):
            fail(f"invalid immutable identity for PR {pr}")
        if row["family"] in families:
            fail(f"duplicate late family {row['family']}")
        families.add(row["family"])
        stratum = row["stratum"]
        computed_capacity[stratum] += row["eligible_capacity_bytes"]
        computed_capacity["total"] += row["eligible_capacity_bytes"]
        computed_families[stratum] += 1
        computed_families["total"] += 1

    zero = data["zero_credit_authorities"]
    if {row.get("pr") for row in zero} != set(EXPECTED_ZERO):
        fail("zero-credit authority set drift")
    for row in zero:
        pr = row["pr"]
        head, run, conclusion, family, reason_prefix = EXPECTED_ZERO[pr]
        if row.get("head_sha") != head or row.get("dedicated_workflow_run") != run:
            fail(f"zero-credit provenance drift for PR {pr}")
        if row.get("dedicated_workflow_conclusion") != conclusion or row.get("family") != family:
            fail(f"zero-credit status drift for PR {pr}")
        if not str(row.get("reason", "")).startswith(reason_prefix):
            fail(f"zero-credit reason weakened for PR {pr}")
        if row.get("eligible_capacity_bytes") != 0 or row.get("family_credit") != 0:
            fail(f"zero-credit PR {pr} gained unauthorized capacity")

    vector = data["conservative_pre_successor_dedup_vector"]
    if computed_capacity != EXPECTED_VECTOR or vector.get("capacity_bytes") != EXPECTED_VECTOR:
        fail(f"capacity arithmetic drift: {computed_capacity}")
    if computed_families != EXPECTED_FAMILIES or vector.get("family_counts") != EXPECTED_FAMILIES:
        fail(f"family arithmetic drift: {computed_families}")
    if min(vector["family_counts"][key] for key in ("uk", "en", "code")) < 2:
        fail("family minimum does not pass")
    if vector.get("family_minimum_status") != "PASS_PRE_SUCCESSOR_GLOBAL_DEDUP":
        fail("family minimum status drift")
    if vector.get("research_corpus_v1_target_normalized_bytes") != TARGET:
        fail("Research Corpus V1 target drift")
    if vector.get("target_gap_normalized_bytes") != TARGET - EXPECTED_VECTOR["total"]:
        fail("target gap arithmetic drift")
    if not math.isclose(vector.get("target_fraction"), EXPECTED_VECTOR["total"] / TARGET, rel_tol=0, abs_tol=1e-9):
        fail("target fraction drift")

    gates = data["downstream_gates"]
    if gates.get("successor_global_cross_source_dedup") != "REQUIRED_NEXT":
        fail("successor dedup gate weakened")
    if gates.get("authorized_training_exposure") != 0 or gates.get("long_training") != "BLOCKED":
        fail("training gate weakened")
    if gates.get("paid_compute") != "NOT_AUTHORIZED":
        fail("paid compute boundary weakened")
    claim = data["claim_boundary"]
    if claim.get("training_authorized") is not False or claim.get("research_corpus_v1_released") is not False:
        fail("claim boundary weakened")
    if claim.get("post_successor_dedup_capacity_claimed") is not False:
        fail("post-dedup capacity fabricated")

    print(
        "NEXT100-063B PASS "
        f"identity={actual} candidate_bytes={EXPECTED_VECTOR['total']} families={EXPECTED_FAMILIES['total']} "
        f"gap={TARGET - EXPECTED_VECTOR['total']}"
    )


if __name__ == "__main__":
    main()
