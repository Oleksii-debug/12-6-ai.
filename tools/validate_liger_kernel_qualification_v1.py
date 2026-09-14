from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from twelve_six.liger_kernel_qualification import canonical_sha256, validate_manifest


def main() -> int:
    path = ROOT / "configs/research/liger_kernel_qualification_v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    print("VALID_CANDIDATE_MANIFEST")
    print(canonical_sha256(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
