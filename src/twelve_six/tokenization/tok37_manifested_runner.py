"""Manifest-authoritative entry point for the incumbent TOK-37 sweep.

DATA-10's committed corpus file is the byte authority. Historical helper
constants in ``bpe_sweep`` retained terminal newlines inside code records,
which makes their joined text differ from the manifested 1,454-byte file.
This entry point reconstructs the same nine logical records from the exact
committed bytes and then delegates to the existing TOK-37 implementation.
"""

from __future__ import annotations

from . import bpe_sweep


def manifested_training_records() -> tuple[tuple[str, str], ...]:
    raw = bpe_sweep.CORPUS_PATH.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if len(lines) != 17:
        raise bpe_sweep.SweepError(
            f"unexpected DATA-10 manifested corpus layout: {len(lines)} lines"
        )

    records: list[tuple[str, str]] = [
        ("uk-1", lines[0]),
        ("uk-2", lines[1]),
        ("uk-3", lines[2]),
        ("en-1", lines[3]),
        ("en-2", lines[4]),
        ("en-3", lines[5]),
        ("code-1", "\n".join(lines[6:8])),
        ("code-2", "\n".join(lines[8:14])),
        ("code-3", "\n".join(lines[14:17])),
    ]

    reconstructed = "\n".join(text for _record_id, text in records) + "\n"
    if reconstructed != raw:
        raise bpe_sweep.SweepError(
            "manifested DATA-10 logical-record reconstruction changed corpus bytes"
        )
    return tuple(records)


def main() -> int:
    # Keep the existing sweep/model-probe implementation as the single authority;
    # only replace its stale duplicate record literals with records reconstructed
    # from the exact manifested corpus bytes before any contract/hash/training work.
    bpe_sweep.TRAIN_RECORDS = manifested_training_records()
    return bpe_sweep.main()


if __name__ == "__main__":
    raise SystemExit(main())
