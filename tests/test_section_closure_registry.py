from __future__ import annotations

import json
from pathlib import Path


_REGISTRY = Path(__file__).parents[1] / "coordination" / "SECTION_CLOSURE_REGISTRY.json"


def _load_registry() -> dict[str, object]:
    return json.loads(_REGISTRY.read_text(encoding="utf-8"))


def test_section_registry_is_structurally_self_consistent() -> None:
    registry = _load_registry()
    plan = registry["canonical_plan"]
    sections = registry["sections"]

    assert isinstance(plan, dict)
    assert isinstance(sections, list)
    assert plan["section_count"] == len(sections)
    assert [section["number"] for section in sections] == list(range(len(sections)))
    assert all(isinstance(section["title"], str) and section["title"] for section in sections)


def test_section_registry_frontier_is_first_non_done_section() -> None:
    registry = _load_registry()
    sections = registry["sections"]
    frontier = registry["frontier"]

    assert isinstance(sections, list)
    assert isinstance(frontier, dict)
    first_non_done = next(
        (section["number"] for section in sections if section["status"] != "DONE"),
        None,
    )
    assert frontier["earliest_unfinished_section"] == first_non_done


def test_section_registry_never_marks_later_section_done_across_open_frontier() -> None:
    registry = _load_registry()
    sections = registry["sections"]

    assert isinstance(sections, list)
    seen_unfinished = False
    for section in sections:
        if section["status"] != "DONE":
            seen_unfinished = True
        if seen_unfinished:
            assert section["status"] != "DONE"


def test_section_registry_status_values_are_declared() -> None:
    registry = _load_registry()
    allowed = set(registry["status_model"])
    sections = registry["sections"]

    assert isinstance(sections, list)
    assert all(section["status"] in allowed for section in sections)


def test_done_sections_require_durable_integration_evidence() -> None:
    registry = _load_registry()
    sections = registry["sections"]

    assert isinstance(sections, list)
    for section in sections:
        if section["status"] != "DONE":
            continue
        evidence = section["evidence"]
        assert isinstance(evidence, dict)
        assert isinstance(evidence.get("integrated_main_sha"), str)
        assert len(evidence["integrated_main_sha"]) == 40
        assert isinstance(evidence.get("pull_request"), int)
        terminal_ci = evidence.get("terminal_ci")
        assert isinstance(terminal_ci, dict)
        assert terminal_ci.get("conclusion") == "success"


def test_registry_records_actual_repository_identity() -> None:
    registry = _load_registry()
    observed = registry["observed_live_state"]

    assert isinstance(observed, dict)
    assert observed["canonical_repository"] == "Oleksii-debug/12-6-ai."
