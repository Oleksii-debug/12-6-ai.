"""W3 S16: bounded enumeration of hostile Mapping metadata, candidate-only.

All fixtures are synthetic metadata. No source/license/training authority is granted.
"""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from tools.plan2_s16_structural_preflight_v1 import (
    RECORD_KEYS,
    Section16PreflightDenied,
    inspect_structure,
)
from test_plan2_s16_structural_preflight_v1 import cohort as _cohort


class OversizedIterator(Mapping):
    """A hostile mapping that never terminates key enumeration voluntarily."""

    def __init__(self, record):
        self.record = record
        self.yields = 0

    def __iter__(self):
        while True:
            self.yields += 1
            if self.yields > len(RECORD_KEYS) + 5:
                raise RuntimeError("UNBOUNDED_ITERATOR_EXPOSED")
            yield "source_id"

    def __len__(self):
        return len(RECORD_KEYS)

    def __getitem__(self, key):
        return self.record[key]


def test_hostile_iterator_cannot_enumerate_unboundedly():
    rows = _cohort()
    hostile = OversizedIterator(rows[0])
    rows[0] = hostile
    with pytest.raises(Section16PreflightDenied):
        inspect_structure(rows)
    assert hostile.yields <= len(RECORD_KEYS) + 1


class LengthMismatch(Mapping):
    """A mapping that conceals a physical key from its enumeration."""

    def __init__(self, record):
        self.record = record

    def __iter__(self):
        return iter(self.record)

    def __len__(self):
        return len(self.record) + 1

    def __getitem__(self, key):
        return self.record[key]


def test_mapping_length_mismatch_cannot_issue_candidate_receipt():
    rows = _cohort()
    rows[0] = LengthMismatch(rows[0])
    with pytest.raises(Section16PreflightDenied, match="mapping length mismatch"):
        inspect_structure(rows)


class ExcessUniqueKeys(Mapping):
    def __init__(self, record):
        self.record = record
        self.yields = 0

    def __iter__(self):
        for key in self.record:
            self.yields += 1
            yield key
        self.yields += 1
        yield "extra_unapproved_grant"

    def __len__(self):
        return len(self.record) + 1

    def __getitem__(self, key):
        return self.record[key]


def test_extra_unique_key_is_bounded_and_rejected():
    rows = _cohort()
    hostile = ExcessUniqueKeys(rows[0])
    rows[0] = hostile
    with pytest.raises(Section16PreflightDenied, match="enumeration overflow"):
        inspect_structure(rows)
    assert hostile.yields == len(RECORD_KEYS) + 1


class ExactMapping(Mapping):
    def __init__(self, record):
        self.record = record

    def __iter__(self):
        return iter(self.record)

    def __len__(self):
        return len(self.record)

    def __getitem__(self, key):
        return self.record[key]


def test_valid_mapping_remains_candidate_only():
    rows = _cohort()
    rows[0] = ExactMapping(rows[0])
    result = inspect_structure(rows)
    assert result["physical_record_count"] == 6
    assert result["rights_independently_verified"] is False
    assert result["training_corpus_authorized"] is False
    assert result["production_release_authorized"] is False
