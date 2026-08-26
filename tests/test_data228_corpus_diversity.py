from __future__ import annotations

import hashlib
import math

import pytest

from twelve_six.data.corpus_diversity import (
    CorpusDiversityError,
    SourceFamilyIdentity,
    measure_diversity,
    validate_family_identities,
)
from twelve_six.data.multilingual_pretraining import AdmittedRecord, LanguageEvidence, script_profile


DUMMY_MANIFEST = "0" * 64


def _record(record_id: str, source_id: str, language: str, text: str) -> AdmittedRecord:
    normalized_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    profile = script_profile(text)
    evidence = LanguageEvidence(
        label=language,  # type: ignore[arg-type]
        confidence=1.0,
        script=profile,
        ukrainian_lexical_hits=1 if language == "uk" else 0,
        english_lexical_hits=1 if language == "en" else 0,
        reason="test",
    )
    return AdmittedRecord(
        record_id=record_id,
        source_id=source_id,
        source_version="v1",
        source_manifest_sha256=DUMMY_MANIFEST,
        split="train",
        modality="natural",
        language=language,
        normalized_text=text,
        normalized_sha256=normalized_sha,
        language_evidence=evidence,
    )


def _family(source_id: str, family: str, publisher: str, corpus: str) -> SourceFamilyIdentity:
    return SourceFamilyIdentity(
        source_id=source_id,
        family_id=family,
        publisher_identity=publisher,
        underlying_corpus_identity=corpus,
    )


def test_baseline_one_family_per_language_is_measured_as_single_source() -> None:
    records = (
        _record("ua-1", "ua.rada.d23314", "uk", "українські дані " * 20),
        _record("en-1", "en.standard.manual.8", "en", "the english data " * 20),
        _record("en-2", "en.standard.manual.9", "en", "the metadata rules " * 19),
    )
    families = (
        _family("ua.rada.d23314", "rada.open-data", "rada.gov.ua", "rada-laws"),
        _family("en.standard.manual.8", "standardebooks.manual", "standardebooks.org", "standardebooks-manual"),
        _family("en.standard.manual.9", "standardebooks.manual", "standardebooks.org", "standardebooks-manual"),
    )

    report = measure_diversity(records, family_identities=families)
    for language in ("uk", "en"):
        metrics = report["languages"][language]
        assert metrics["family_count"] == 1
        assert metrics["top_family_token_share"] == 1.0
        assert metrics["effective_source_count"] == 1.0
        assert metrics["entropy_nats"] == 0.0
        assert metrics["meets_minimum_family_count"] is False
    assert report["all_required_languages_meet_family_floor"] is False


def test_two_independent_families_per_language_pass_family_floor() -> None:
    uk_a = "українські дані " * 20
    uk_b = "українська документація " * 13
    en_a = "the english data " * 20
    en_b = "the python tutorial " * 17
    records = (
        _record("ua-a", "ua.rada", "uk", uk_a),
        _record("ua-b", "ua.kubernetes", "uk", uk_b),
        _record("en-a", "en.standard", "en", en_a),
        _record("en-b", "en.python", "en", en_b),
    )
    families = (
        _family("ua.rada", "rada.open-data", "rada.gov.ua", "rada-laws"),
        _family("ua.kubernetes", "kubernetes.docs", "cncf.io", "kubernetes-website"),
        _family("en.standard", "standardebooks.manual", "standardebooks.org", "standardebooks-manual"),
        _family("en.python", "python.docs", "python.org", "cpython-documentation"),
    )

    report = measure_diversity(records, family_identities=families)
    for language in ("uk", "en"):
        metrics = report["languages"][language]
        assert metrics["family_count"] == 2
        assert 0.5 <= metrics["top_family_token_share"] < 1.0
        assert 1.0 < metrics["effective_source_count"] <= 2.0
        assert 0.0 < metrics["entropy_nats"] <= math.log(2)
        assert 0.0 < metrics["normalized_entropy"] <= 1.0
        assert metrics["meets_minimum_family_count"] is True
    assert report["all_required_languages_meet_family_floor"] is True
    assert math.isclose(sum(report["language_balance"].values()), 1.0)


def test_same_publisher_cannot_be_renamed_into_two_families() -> None:
    with pytest.raises(CorpusDiversityError, match="same publisher"):
        validate_family_identities(
            (
                _family("a", "publisher.family.a", "publisher.example", "corpus-a"),
                _family("b", "publisher.family.b", "publisher.example", "corpus-b"),
            )
        )


def test_same_underlying_corpus_cannot_be_split_into_two_families() -> None:
    with pytest.raises(CorpusDiversityError, match="same underlying corpus"):
        validate_family_identities(
            (
                _family("a", "family.a", "publisher-a", "same-corpus"),
                _family("b", "family.b", "publisher-b", "same-corpus"),
            )
        )


def test_duplicate_normalized_document_is_rejected_not_counted_twice() -> None:
    text = "the same normalized document " * 20
    records = (
        _record("a", "en.a", "en", text),
        _record("b", "en.b", "en", text),
    )
    families = (
        _family("en.a", "family.a", "publisher-a", "corpus-a"),
        _family("en.b", "family.b", "publisher-b", "corpus-b"),
    )
    with pytest.raises(CorpusDiversityError, match="duplicate normalized document"):
        measure_diversity(records, family_identities=families)


def test_missing_family_classification_fails_closed() -> None:
    records = (_record("a", "en.unknown", "en", "the english data " * 20),)
    families = (_family("en.other", "family.a", "publisher-a", "corpus-a"),)
    with pytest.raises(CorpusDiversityError, match="missing source-family classification"):
        measure_diversity(records, family_identities=families)
