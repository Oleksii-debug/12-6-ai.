from __future__ import annotations

import os
from pathlib import Path

from twelve_six.postbase_research_fixture import canonical_json, run_fixture


def main() -> None:
    raw = os.environ.get("NEXT100_094_BASE_CHECKPOINT")
    checkpoint = Path(raw) if raw else None
    print(canonical_json(run_fixture(checkpoint=checkpoint)), end="")


if __name__ == "__main__":
    main()
