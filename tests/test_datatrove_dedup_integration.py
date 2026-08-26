from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from twelve_six.data.datatrove_dedup_runtime import (
    DataTroveMinhashSpec,
    run_datatrove_minhash,
)

H = "a" * 64
R = "b" * 64


def _read_jsonl_outputs(folder: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(folder.rglob("*.jsonl*")):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def test_datatrove_0100_four_stage_ukrainian_minhash_smoke(tmp_path: Path) -> None:
    pytest.importorskip("datatrove")
    pytest.importorskip("spacy")

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    work_dir = tmp_path / "work"
    logs_dir = tmp_path / "logs"
    input_dir.mkdir()

    duplicate = (
        "це український тестовий документ для перевірки масштабованого мінхеш "
        "дедуплікування корпусу мовної моделі з достатньою кількістю слів"
    )
    distinct = (
        "інший незалежний український текст описує навчання трансформера якість "
        "даних валідацію та відтворюваний дослідницький процес"
    )
    records = [
        {"id": "doc-a", "text": duplicate, "source_id": "ua-family-a"},
        {"id": "doc-b", "text": duplicate, "source_id": "ua-family-b"},
        {"id": "doc-c", "text": distinct, "source_id": "ua-family-c"},
    ]
    with (input_dir / "00000.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    spec = DataTroveMinhashSpec(
        source_registry_sha256=H,
        reserved_registry_sha256=R,
        input_uri=input_dir.as_uri(),
        output_uri=output_dir.as_uri(),
        work_uri=work_dir.as_uri(),
        logging_uri=logs_dir.as_uri(),
        language="uk",
        signature_tasks=1,
        workers=1,
    )
    run_datatrove_minhash(spec)

    rows = _read_jsonl_outputs(output_dir)
    ids = {row["id"] for row in rows}
    assert len(rows) == 2
    assert "doc-c" in ids
    assert len({"doc-a", "doc-b"} & ids) == 1

    removed_rows = _read_jsonl_outputs(work_dir / "removed")
    assert len(removed_rows) == 1
    assert removed_rows[0]["id"] in {"doc-a", "doc-b"}
