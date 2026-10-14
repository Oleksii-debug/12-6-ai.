"""W3 synthetic candidate-only regression: subclass values fail closed."""
from __future__ import annotations

import hashlib
import pytest
from tools.plan2_s16_structural_preflight_v1 import Section16PreflightDenied, inspect_structure


def _cohort():
    rows = []
    for split in ("train", "validation", "test"):
        for lang in ("uk", "en"):
            key = f"{lang}.{split}"
            h = lambda suffix: hashlib.sha256((key + suffix).encode()).hexdigest()
            rows.append({
                "record_id": key, "source_id": "source." + key,
                "source_family": "family." + key,
                "document_family": "document." + key,
                "source_revision": "v1", "language": lang, "split": split,
                "payload_sha256": h("raw"), "normalized_sha256": h("normalized"),
                "rights_receipt_sha256": h("rights"),
                "privacy_receipt_sha256": h("privacy"),
            })
    return rows


class SubclassedString(str):
    pass


@pytest.mark.parametrize("field", (
    "record_id", "source_id", "source_family", "document_family",
    "source_revision", "payload_sha256", "normalized_sha256",
    "rights_receipt_sha256", "privacy_receipt_sha256",
))
def test_subclassed_identity_and_hash_denied(field):
    rows = _cohort()
    rows[0][field] = SubclassedString(rows[0][field])
    with pytest.raises(Section16PreflightDenied):
        inspect_structure(rows)


def test_structural_pass_is_never_training_authority():
    receipt = inspect_structure(_cohort())
    assert receipt["training_corpus_authorized"] is False
    assert receipt["production_release_authorized"] is False
