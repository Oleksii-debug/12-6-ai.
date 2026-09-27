from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "tools/run_eval647_real_overlap_ci_v1.sh"
EXPECTED_MATERIALIZER_V2_BLOB = "a4de33f1e225ab8598197e4c1cba28480180d575"
EXPECTED_RUN_ID = "36026689718"
EXPECTED_ARTIFACT_ID = "10820342689"
EXPECTED_ARTIFACT_SHA = "99069ce2183abbbc374749cca5c538efa259df0c64658a9b88cc96b25c0fbba0"
EXPECTED_RECORDS_SHA = "bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8"
EXPECTED_MATERIALIZATION_ID = "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"


def test_real_overlap_carrier_binds_clean_v2_physical_authority() -> None:
    carrier = CARRIER.read_text(encoding="utf-8")
    assert f'EXPECTED_MATERIALIZER_V2_BLOB="{EXPECTED_MATERIALIZER_V2_BLOB}"' in carrier
    assert 'git hash-object src/twelve_six/data/post_g05_g06_materialization_v2.py' in carrier
    assert f'PHYSICAL_RUN_ID="{EXPECTED_RUN_ID}"' in carrier
    assert f'PHYSICAL_ARTIFACT_ID="{EXPECTED_ARTIFACT_ID}"' in carrier
    assert f'EXPECTED_ARTIFACT_ZIP_SHA="{EXPECTED_ARTIFACT_SHA}"' in carrier
    assert f'EXPECTED_CURRENT_SHA="{EXPECTED_RECORDS_SHA}"' in carrier
    assert f'EXPECTED_MATERIALIZATION_ID="{EXPECTED_MATERIALIZATION_ID}"' in carrier
    assert 'EXPECTED_MATERIALIZATION_SCHEMA="12-6.d03-post-g05-g06-materialization.v2"' in carrier
    assert 'EXPECTED_RECORD_COUNT="257"' in carrier
    assert 'EXPECTED_PAYLOAD_BYTES="5601716"' in carrier
    assert 'EXPECTED_DISTINCT_SOURCE_IDS="244"' in carrier
    assert 'artifact["digest"] == f"sha256:{zip_sha}"' in carrier
    assert 'evidence["schema_version"] == expected_schema' in carrier
    assert 'result["record_payload_jsonl_sha256"] == expected_jsonl' in carrier


def test_real_overlap_carrier_rejects_historical_dirty_authority() -> None:
    carrier = CARRIER.read_text(encoding="utf-8")
    assert "b1ec0433fbd9675645b7e29c1e32b406e638fa081868cad7a56afec8f9a601cc" not in carrier
    assert "3f60cfe55435daf53908c492be358f36d7ebbc2ebee532c921a69ba92b2f6b25" not in carrier
    assert "10032510626" not in carrier
    assert "10292063266" not in carrier
    assert "10297428350" not in carrier
    assert '--expected-survivor-record-count 272' not in carrier
    assert 'record_count"] == 255' not in carrier
