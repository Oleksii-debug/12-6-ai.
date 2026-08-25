"""Exact DATA-10-normalized entrypoint for the LEARN-04 campaign.

DATA-10 admission strips terminal newlines from code records before its manifested
training snapshot is formed.  The experiment module retains the source literals for
traceability; this entrypoint applies that incumbent normalization before execution.
"""

from __future__ import annotations

from . import learn04_1m_experiment as experiment


def bind_data10_normalized_records() -> None:
    experiment.TRAIN_RECORDS = tuple(
        (stratum, record_id, text.rstrip("\n") if stratum == "code" else text)
        for stratum, record_id, text in experiment.TRAIN_RECORDS
    )


bind_data10_normalized_records()


def main(argv: list[str] | None = None) -> int:
    return experiment.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
