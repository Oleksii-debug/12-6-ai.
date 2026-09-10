from __future__ import annotations

import pytest

from twelve_six.data.wikisource_pd_contract import (
    APPROVED_CATEGORY,
    INDEX_REVISION_ID,
    PAGE_PREFIX,
    WikisourceIntakeError,
)
from twelve_six.data.wikisource_pd_edition import materialize_live


def test_materialize_live_spaces_every_mediawiki_request() -> None:
    clock = [0.0]
    sleeps: list[float] = []
    calls: list[tuple[float, dict[str, str]]] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock[0] += seconds

    def fake_get_json(params: dict[str, str]) -> dict[str, object]:
        calls.append((clock[0], params))
        if params == {"action": "parse", "oldid": str(INDEX_REVISION_ID), "prop": "links"}:
            return {"parse": {"links": [{"title": f"{PAGE_PREFIX}13"}]}}
        if params == {
            "action": "query",
            "titles": f"{PAGE_PREFIX}13",
            "prop": "revisions",
            "rvprop": "ids",
        }:
            return {"query": {"pages": [{"revisions": [{"revid": 560107}]}]}}
        if params == {"action": "parse", "oldid": "560107", "prop": "text|categories"}:
            return {
                "parse": {
                    "revid": 560107,
                    "categories": [{"category": APPROVED_CATEGORY}],
                    "text": (
                        "<p>Український літературний текст достатньої довжини для "
                        "перевіреної сторінки та контрольованого мережевого запиту.</p>"
                    ),
                }
            }
        raise AssertionError(f"unexpected request: {params!r}")

    result = materialize_live(
        max_pages=1,
        cadence_seconds=0.5,
        get_json=fake_get_json,
        sleep_fn=fake_sleep,
        monotonic_fn=lambda: clock[0],
    )

    assert result.report["candidate"]["page_count"] == 1
    assert len(calls) == 3
    assert [called_at for called_at, _ in calls] == [0.0, 0.5, 1.0]
    assert sleeps == [0.5, 0.5]


def test_materialize_live_rejects_boolean_cadence() -> None:
    with pytest.raises(WikisourceIntakeError, match="cadence"):
        materialize_live(cadence_seconds=True)
