from twelve_six.swarm_protocol_v2 import (
    Claim,
    canonical_lane_key,
    ci_pressure,
    distribution_counts,
    package_is_large,
    routing_slot,
    winning_claim,
)


def test_any_120_consecutive_issue_numbers_cover_all_coarse_slots_once():
    counts = distribution_counts(range(731, 851))
    assert len(counts) == 120
    assert set(counts.values()) == {1}


def test_two_hundred_workers_are_balanced_over_120_coarse_slots():
    counts = distribution_counts(range(1000, 1200))
    assert len(counts) == 120
    assert max(counts.values()) - min(counts.values()) <= 1


def test_one_thousand_workers_are_balanced_over_120_coarse_slots():
    counts = distribution_counts(range(1000, 2000))
    assert len(counts) == 120
    assert max(counts.values()) - min(counts.values()) <= 1


def test_routing_is_deterministic_and_rejects_bad_ids():
    assert routing_slot(732) == routing_slot(732)
    for value in (0, -1, True, 1.5, "732"):
        try:
            routing_slot(value)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid issue number: {value!r}")


def test_lane_key_normalization_prevents_formatting_aliases():
    first = canonical_lane_key("d03", "Common Pile", "source rights audit", "v1")
    second = canonical_lane_key("D03", "common-pile", "SOURCE_RIGHTS_AUDIT", "V1")
    assert first == second == "D03|COMMON-PILE|SOURCE-RIGHTS-AUDIT|V1"


def test_claim_arbitration_uses_earliest_created_at_then_issue_number():
    key = "D03|COMMON-PILE|SOURCE-RIGHTS-AUDIT|V1"
    claims = [
        Claim(810, key, "2026-08-27T06:40:00Z"),
        Claim(809, key, "2026-08-27T06:40:00Z"),
        Claim(808, key, "2026-08-27T06:40:01Z"),
        Claim(700, key, "2026-08-27T06:30:00Z", status="TERMINAL"),
    ]
    assert winning_claim(claims, key).issue_number == 809  # type: ignore[union-attr]


def test_tiny_puzzle_does_not_pass_large_package_gate():
    assert not package_is_large(
        {
            "implementation_or_primary_research",
            "focused_tests",
            "documentation_or_operator_handoff",
        }
    )


def test_vertical_package_passes_large_package_gate():
    assert package_is_large(
        {
            "implementation_or_primary_research",
            "focused_tests",
            "adversarial_or_negative_tests",
            "machine_readable_evidence_or_validator",
            "live_authority_binding",
            "documentation_or_operator_handoff",
        }
    )


def test_ci_pressure_thresholds():
    assert ci_pressure(0, 0) == "GREEN"
    assert ci_pressure(20, 5) == "GREEN"
    assert ci_pressure(26, 0) == "AMBER"
    assert ci_pressure(90, 10) == "AMBER"
    assert ci_pressure(101, 0) == "RED"


def test_ci_pressure_rejects_invalid_counts():
    for queued, running in ((-1, 0), (0, -1), (True, 0), (0, 1.5)):
        try:
            ci_pressure(queued, running)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:
            raise AssertionError("invalid CI pressure count was accepted")
