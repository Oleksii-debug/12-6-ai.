import json
from pathlib import Path
import pytest

from twelve_six.eval291_en_selection_validation_v2 import (
    AuthorityError, assert_no_exposure, materialize_selection_validation, validate
)

def test_terminal_authority():
    root = Path(__file__).resolve().parents[1]
    result = validate(root)
    assert result["selection_records"] == 3
    assert result["final_test_records"] == 0
    assert result["families"] == 3

def test_materializer_only_reads_selection_validation():
    root = Path(__file__).resolve().parents[1]
    rows = materialize_selection_validation(root)
    assert all(r["purpose"] == "selection_validation" for r in rows)

def test_exact_exposure_rejected():
    with pytest.raises(AuthorityError):
        assert_no_exposure(["alpha beta gamma delta epsilon zeta"], ["alpha beta gamma delta epsilon zeta"])

def test_near_copy_exposure_rejected():
    a = "one two three four five six seven eight nine ten eleven twelve"
    b = "one two three four five six seven eight nine ten eleven changed"
    with pytest.raises(AuthorityError):
        assert_no_exposure([a], [b], threshold=0.5)

def test_unrelated_consumer_passes():
    assert_no_exposure(["request checksum stream api"], ["astronomy telescope galaxy orbit"])

def test_final_manifest_has_no_plaintext_and_no_members():
    root = Path(__file__).resolve().parents[1]
    obj = json.loads((root / "data/evaluation/eval291/v2/final-test/manifest.json").read_text())
    raw = json.dumps(obj)
    assert '"text"' not in raw and '"content"' not in raw and '"outcome"' not in raw
    assert obj["members"] == []

def test_reservation_epoch_is_fixed():
    root = Path(__file__).resolve().parents[1]
    obj = json.loads((root / "data/evaluation/eval291/v2/reservation.json").read_text())
    assert obj["state"] == "SEALED"
    assert obj["reservation_commit_sha"] == "d3f7fead8c04cafd535d1e574a7203523b54464d"
