"""Authoritative MODEL-119 runner using exact DATA-25 record-id semantics.

The first research runner accidentally used source_version ``0.1`` instead of
DATA-25's exact ``0.1.0`` while deriving deterministic record IDs. This wrapper
corrects that function before delegating to the retained experiment machinery.
The earlier local evidence is rejected and superseded by v2 evidence.
"""
from __future__ import annotations

import run_model119_qk_norm as experiment
from twelve_six.data.corpus_v01 import sha


def _data25_record_id(stratum: str, index: int, raw: str) -> str:
    source_id = f"project-authored:{stratum}:corpus-v01"
    source_version = "0.1.0"
    digest = sha(f"{source_id}\0{source_version}\0{index}\0{raw}".encode())[:24]
    return f"{source_id}:{digest}"


experiment._record_id = _data25_record_id


if __name__ == "__main__":
    experiment.main()
