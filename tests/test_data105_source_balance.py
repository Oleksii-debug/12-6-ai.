from __future__ import annotations

from twelve_six.data.source_balance import (
    SourcePolicy,
    analyze_records,
    build_source_balance_plan,
    concentration,
    policy_is_effective,
    provenance_family,
    source_weight_units,
)


def _row(
    record_id: str,
    source_id: str,
    stratum: str,
    byte_tokens: int,
    *,
    external: bool = True,
    project_authored: bool = False,
    **provenance,
):
    return {
        "record_id": record_id,
        "source_id": source_id,
        "stratum": stratum,
        "modality": "code" if stratum == "code" else "natural",
        "byte_tokens": byte_tokens,
        "external": external,
        "project_authored": project_authored,
        **provenance,
    }


def test_taxonomy_uses_only_provenance_and_reports_requested_mass_axes():
    rows = [
        _row(
            "a",
            "github:pallets/itsdangerous",
            "code",
            100,
            repository_url="https://github.com/pallets/itsdangerous",
        ),
        _row(
            "b",
            "github:pallets/flask",
            "code",
            300,
            repository_url="https://github.com/pallets/flask",
        ),
        _row(
            "c",
            "rada:law",
            "uk",
            500,
            source_url="https://data.rada.gov.ua/example",
        ),
        _row(
            "d",
            "project-authored:en:corpus-v01",
            "en",
            700,
            external=False,
            project_authored=True,
        ),
    ]
    analysis = analyze_records(rows)

    assert provenance_family(rows[0]) == "github.com/pallets"
    assert provenance_family(rows[2]) == "data.rada.gov.ua"
    assert provenance_family(rows[3]) == "project-authored:en"
    assert analysis["mass"]["source"]["github:pallets/itsdangerous"]["byte_tokens"] == 100
    assert analysis["mass"]["source_family_domain"]["github.com/pallets"]["byte_tokens"] == 400
    assert analysis["mass"]["origin"]["real_external"]["byte_tokens"] == 900
    assert analysis["mass"]["origin"]["project_authored"]["byte_tokens"] == 700
    assert set(analysis["mass"]) == {
        "language_modality",
        "source",
        "source_family_domain",
        "document_length_bucket",
        "origin",
    }


def test_concentration_metrics_are_interpretable():
    metrics = concentration({"dominant": 75, "minor": 25})
    assert metrics["top_member"] == "dominant"
    assert metrics["top_share"] == 0.75
    assert metrics["effective_count_hill_q2"] == 1.6
    assert 0.8 < metrics["token_entropy_bits"] < 0.82


def test_source_policies_change_only_within_selected_stratum():
    rows = [
        _row("uk-a", "uk:a", "uk", 900),
        _row("uk-b", "uk:b", "uk", 100),
        _row("en-a", "en:a", "en", 500),
        _row("en-b", "en:b", "en", 500),
        _row("code-a", "code:a", "code", 800),
        _row("code-b", "code:b", "code", 200),
    ]
    analysis = analyze_records(rows)
    common = {
        "corpus_identity_sha256": "1" * 64,
        "top_level_mixture_sha256": "2" * 64,
        "analysis": analysis,
        "seed": 105,
    }
    raw = build_source_balance_plan(
        **common,
        policy=SourcePolicy("raw_proportional"),
    )
    cap = build_source_balance_plan(
        **common,
        policy=SourcePolicy("bounded_source_cap", cap_basis_points=3500),
    )
    tempered = build_source_balance_plan(
        **common,
        policy=SourcePolicy("tempered_source_sqrt", temper_exponent="1/2"),
    )

    assert policy_is_effective(cap, raw)
    assert policy_is_effective(tempered, raw)
    for index in range(100):
        for stratum in ("uk", "en", "code"):
            raw_sources = {source for source, _ in dict(raw.source_weights_by_stratum)[stratum]}
            cap_sources = {source for source, _ in dict(cap.source_weights_by_stratum)[stratum]}
            assert raw.source_for_draw(stratum, index) in raw_sources
            assert cap.source_for_draw(stratum, index) in cap_sources


def test_single_source_per_stratum_makes_policies_distribution_equivalent():
    rows = [
        _row(
            "uk",
            "project-authored:uk:corpus-v01",
            "uk",
            900,
            external=False,
            project_authored=True,
        ),
        _row(
            "en",
            "project-authored:en:corpus-v01",
            "en",
            700,
            external=False,
            project_authored=True,
        ),
        _row(
            "code",
            "project-authored:code:corpus-v01",
            "code",
            400,
            external=False,
            project_authored=True,
        ),
    ]
    analysis = analyze_records(rows)
    common = {
        "corpus_identity_sha256": "a" * 64,
        "top_level_mixture_sha256": "b" * 64,
        "analysis": analysis,
        "seed": 105,
    }
    raw = build_source_balance_plan(
        **common,
        policy=SourcePolicy("raw_proportional"),
    )
    candidates = (
        build_source_balance_plan(
            **common,
            policy=SourcePolicy("bounded_source_cap", cap_basis_points=3500),
        ),
        build_source_balance_plan(
            **common,
            policy=SourcePolicy("tempered_source_sqrt", temper_exponent="1/2"),
        ),
    )
    for plan in candidates:
        assert not policy_is_effective(plan, raw)
        for stratum in ("uk", "en", "code"):
            assert [plan.source_for_draw(stratum, i) for i in range(20)] == [
                raw.source_for_draw(stratum, i) for i in range(20)
            ]


def test_exact_integer_policy_weights():
    masses = {"big": 900, "small": 100}
    assert source_weight_units(
        masses,
        SourcePolicy("raw_proportional"),
    ) == {"big": 900, "small": 100}
    assert source_weight_units(
        masses,
        SourcePolicy("bounded_source_cap", cap_basis_points=3500),
    ) == {"big": 350, "small": 100}
    assert source_weight_units(
        masses,
        SourcePolicy("tempered_source_sqrt", temper_exponent="1/2"),
    ) == {"big": 30000, "small": 10000}
