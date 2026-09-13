"""Repository-wide pytest compatibility for legacy bounded-pilot core tests."""

from __future__ import annotations

import pytest

from twelve_six.training.bounded_pilot_core import (
    BoundedPilotStepRunner as CoreBoundedPilotStepRunner,
)


@pytest.fixture(autouse=True)
def _legacy_bounded_pilot_core_only(request, monkeypatch):
    """Keep the pre-facade regression corpus bound to its exact qualified core.

    Dedicated durable/recipe tests instantiate the production facade explicitly.
    This test-only monkeypatch never changes package/runtime symbols.
    """

    if request.module.__name__ == "test_bounded_pilot":
        monkeypatch.setattr(
            request.module,
            "BoundedPilotStepRunner",
            CoreBoundedPilotStepRunner,
        )
