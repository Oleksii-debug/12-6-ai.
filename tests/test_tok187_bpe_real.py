from __future__ import annotations

from twelve_six import tok187_bpe_real as t


def _row(scale: str, vocab: int, seed: int, aggregate: float, macro: float):
    per = {"uk": macro, "en": macro, "code": macro}
    return {
        "scale": scale,
        "actual_vocab_size": vocab,
        "seed": seed,
        "run": {
            "parameter_count": 500_000,
            "model": {"d_ff": 123},
            "evaluation": {
                "final": {
                    "aggregate_bits_per_byte": aggregate,
                    "macro_bits_per_byte": macro,
                    "by_stratum": {
                        key: {"bits_per_byte": value} for key, value in per.items()
                    },
                }
            },
        },
    }


def test_grid_brackets_model116_promising_actual_vocabularies():
    assert t.REQUESTED_GRID == (320, 384, 437, 512)
    assert {320, 384, 437}.issubset(t.REQUESTED_GRID)
    assert max(t.REQUESTED_GRID) > 437


def test_three_paired_model_seeds_are_preregistered():
    assert len(t.MODEL_SEEDS) == 3
    assert len(set(t.MODEL_SEEDS)) == 3


def test_matched_geometry_never_gives_larger_vocab_extra_capacity():
    solved = t._solve_geometries(t.REQUESTED_GRID)
    for label in t.SCALE_LABELS:
        counts = [solved[label][vocab].parameter_count() for vocab in t.REQUESTED_GRID]
        assert counts == sorted(counts, reverse=True)
        target = t._anchors()[label].parameter_count()
        assert all(abs(count - target) / target <= t.PARAMETER_TOLERANCE for count in counts)


def test_scale_ranking_uses_held_out_aggregate_bpb_before_macro():
    rows = []
    for seed in t.MODEL_SEEDS:
        rows.append(_row("500K", 320, seed, aggregate=1.0, macro=9.0))
        rows.append(_row("500K", 384, seed, aggregate=1.1, macro=0.1))
    summary = t._scale_summary(rows, "500K")
    assert summary["primary_metric"].startswith("mean final selection-validation aggregate")
    assert summary["ranked_candidates"][0]["actual_vocab_size"] == 320
    assert summary["ranked_candidates"][0]["rank_primary_held_out_bpb"] == 1


def test_promotion_fails_closed_without_external_code_and_representativeness():
    summaries = {}
    for label in t.SCALE_LABELS:
        rows = []
        for seed in t.MODEL_SEEDS:
            rows.append(_row(label, 320, seed, aggregate=1.0, macro=1.0))
            rows.append(_row(label, 384, seed, aggregate=1.2, macro=1.2))
        summaries[label] = t._scale_summary(rows, label)
    data183 = {
        "truth_boundary": {"external_real_code_present": False},
        "representativeness": {"full_v0_2_claim": False},
    }
    status = t._promotion_status(data183, summaries)
    assert status["tokenizer_promoted"] is False
    assert status["tokenizer_frozen"] is False
    assert status["promotion_allowed"] is False
    assert "EXTERNAL_REAL_CODE_UNAVAILABLE" in status["blockers"]
    assert "FULL_V0_2_REPRESENTATIVENESS_NOT_ESTABLISHED" in status["blockers"]


def test_selection_manifest_binds_text_and_source_identity():
    rows = [
        {
            "record_id": "a",
            "split": "validation",
            "stratum": "uk",
            "source_id": "family-a",
            "origin": "external_real",
            "text": "Україна",
        }
    ]
    one = t._selection_manifest(
        rows,
        purpose="selection_validation_only_not_final_test",
        corpus_identity_sha256="a" * 64,
    )
    changed = [dict(rows[0], text="Україна!")]
    two = t._selection_manifest(
        changed,
        purpose="selection_validation_only_not_final_test",
        corpus_identity_sha256="a" * 64,
    )
    assert one["identity_sha256"] != two["identity_sha256"]
    assert one["records"][0]["source_id"] == "family-a"
