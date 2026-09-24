from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CARRIER = ROOT / "tools/run_eval647_real_overlap_ci_v1.sh"
EXPECTED_MATERIALIZER_V2_BLOB = "a4de33f1e225ab8598197e4c1cba28480180d575"


def test_real_overlap_carrier_binds_materializer_v2_implementation() -> None:
    carrier = CARRIER.read_text(encoding="utf-8")
    assert (
        f'EXPECTED_MATERIALIZER_V2_BLOB="{EXPECTED_MATERIALIZER_V2_BLOB}"'
        in carrier
    )
    assert (
        'git hash-object src/twelve_six/data/post_g05_g06_materialization_v2.py'
        in carrier
    )
    assert (
        '--expected-materializer-v2-implementation-git-blob-sha1 '
        '"$EXPECTED_MATERIALIZER_V2_BLOB"'
        in carrier
    )
