import json

import pytest

from twelve_six.translation_evaluation import TranslationExample, corpus_chrf, evaluate_translation


def test_chrf_effective_order_counts_repetitions_within_segments():
    assert corpus_chrf([" a b\n"], ["ab"]) == 100
    assert corpus_chrf(["і\u0308"], ["ї"]) == 100
    assert corpus_chrf(["A"], ["a"]) == 0
    assert corpus_chrf([""], ["blue"]) == 0
    assert corpus_chrf(["a", "b"], ["b", "a"]) == 0
    assert corpus_chrf(["aa"], ["aaa"]) == pytest.approx(100 * 35 / 55)
    assert corpus_chrf(["aa", "bb"], ["a", "bb"]) == pytest.approx(100 * 35 / 36)
    with pytest.raises(ValueError):
        corpus_chrf(["one", "two"], ["one"])


def test_evaluation_uses_source_only_and_reports_both_directions():
    examples = (
        TranslationExample("1", "en-uk", "Blue", "Синій"),
        TranslationExample("2", "en-uk", "Rain", "Дощ"),
        TranslationExample("1", "uk-en", "Синій", "Blue"),
        TranslationExample("2", "uk-en", "Дощ", "Rain"),
        TranslationExample("3", "uk-en", "Зима", "Winter"),
    )
    responses = iter((" Синій\n", "Rain", "Blue", " ", "Winter"))
    calls = []

    def translate(direction, source):
        calls.append((direction, source))
        return next(responses)

    report = evaluate_translation(iter(examples), translate)
    assert calls == [(row.direction, row.source) for row in examples]
    assert report["example_count"] == 5
    forward = report["by_direction"]["en-uk"]
    reverse = report["by_direction"]["uk-en"]
    assert (forward["count"], reverse["count"]) == (2, 3)
    assert (forward["exact_match_count"], reverse["exact_match_count"]) == (1, 2)
    assert (forward["source_copy_count"], reverse["source_copy_count"]) == (1, 0)
    assert (forward["empty_output_count"], reverse["empty_output_count"]) == (0, 1)
    assert report["macro_average"]["exact_match_rate"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert report["predictions"][0]["prediction"] == " Синій\n"
    assert json.loads(json.dumps(report)) == report


def test_invalid_holdout_is_rejected_before_generation():
    forward = TranslationExample("1", "en-uk", "Blue", "Синій")
    reverse = TranslationExample("1", "uk-en", "Синій", "Blue")

    def unexpected_generation(*_):
        pytest.fail("invalid evaluation should not call the translator")

    with pytest.raises(ValueError, match="both translation directions"):
        evaluate_translation((forward,), unexpected_generation)
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_translation((forward, reverse, forward), unexpected_generation)
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_translation(
            (forward, TranslationExample("2", "uk-en", "Синій", " ")), unexpected_generation,
        )
    with pytest.raises(TypeError, match="non-text"):
        evaluate_translation((forward, reverse), lambda *_: None)
