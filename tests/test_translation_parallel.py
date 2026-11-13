import bz2
import io
import tarfile
from pathlib import Path

import pytest

from twelve_six.data import translation_parallel as parallel


@pytest.fixture
def source_exports(tmp_path, monkeypatch):
    english = []
    ukrainian = []
    links = []
    for index in range(100):
        en_id, uk_id = index + 1, index + 101
        english.append(f"{en_id}\teng\tEnglish sentence {index}.\tEnContributor\tcreated\tupdated\n")
        ukrainian.append(f"{uk_id}\tukr\tУкраїнське речення {index}.\tUkContributor\tcreated\tupdated\n")
        links.append(f"{en_id}\t{uk_id}\n")
    payloads = {
        parallel.EN_EXPORT: english,
        parallel.UK_EXPORT: ukrainian,
        parallel.LINK_EXPORT: links,
    }
    sources = tmp_path / "sources"
    sources.mkdir()
    expected = {}
    for name, lines in payloads.items():
        raw = bz2.compress("".join(lines).encode())
        (sources / name).write_bytes(raw)
        expected[name] = {**parallel.EXPORTS[name], "sha256": parallel._digest(raw), "size": len(raw)}
    monkeypatch.setattr(parallel, "EXPORTS", expected)
    return sources


def test_rebuild_and_verify_each_split_without_opening_holdouts(tmp_path, source_exports, monkeypatch):
    first, second = tmp_path / "first", tmp_path / "second"
    counts = {"train": 10, "validation": 3, "test": 3}
    built = parallel.materialize_translation_dataset(source_exports, first, counts=counts)
    assert parallel.materialize_translation_dataset(source_exports, second, counts=counts) == built
    digest = built["manifest_sha256"]
    assert parallel.verify_translation_dataset(first, expected_manifest_sha256=digest) == built
    read = parallel._read

    def train_only(directory: Path, name: str, limit: int) -> bytes:
        assert name not in {"validation.jsonl", "test.jsonl"}
        return read(directory, name, limit)

    monkeypatch.setattr(parallel, "_read", train_only)
    train = parallel.load_translation_split(first, expected_manifest_sha256=digest, split="train")
    assert len(train) == 10
    assert all(row["pair_id"].startswith("tatoeba:") and row["uk_source"]["contributor"] == "UkContributor"
               for row in train)


def test_rejects_source_manifest_and_member_changes(tmp_path, source_exports):
    bundle = tmp_path / "bundle"
    result = parallel.materialize_translation_dataset(
        source_exports, bundle, counts={"train": 2, "validation": 2, "test": 2},
    )
    with pytest.raises(parallel.TranslationDatasetError, match="manifest SHA-256 mismatch"):
        parallel.load_translation_split(bundle, expected_manifest_sha256="0" * 64, split="train")
    (bundle / "train.jsonl").write_bytes(b"{}\n")
    with pytest.raises(parallel.TranslationDatasetError, match="split SHA-256 mismatch"):
        parallel.load_translation_split(bundle, expected_manifest_sha256=result["manifest_sha256"], split="train")
    file = source_exports / parallel.LINK_EXPORT
    changed = bytearray(file.read_bytes())
    changed[-1] ^= 1
    file.write_bytes(changed)
    with pytest.raises(parallel.TranslationDatasetError, match="source snapshot changed"):
        parallel.materialize_translation_dataset(
            source_exports, tmp_path / "changed", counts={"train": 2, "validation": 2, "test": 2},
        )


def test_cross_split_audit_rejects_normalized_endpoints_even_with_new_ids():
    def row(split, en_id, uk_id, en, uk):
        digest = next(
            f"{number:064x}" for number in (n << 224 for n in range(100))
            if parallel._component_split(f"{number:064x}") == split
        )
        return {
            "pair_id": parallel._pair_id(en_id, uk_id),
            "en": en,
            "uk": uk,
            "component_sha256": digest,
            "en_source": parallel._source(en_id, "EnContributor"),
            "uk_source": parallel._source(uk_id, "UkContributor"),
        }

    train = row("train", 1, 2, "The same line.", "Перший переклад.")
    heldout = row("test", 3, 4, "THE SAME LINE!", "Другий переклад.")
    with pytest.raises(parallel.TranslationDatasetError, match="split contamination"):
        parallel._audit_splits({"train": [train], "test": [heldout]})


def test_flores_builder_screens_both_languages_and_keeps_attribution(tmp_path, monkeypatch):
    en = [f"English evaluation {index}." for index in range(1012)]
    uk = [f"Український тест {index}." for index in range(1012)]
    metadata = ["URL\tdomain\ttopic\thas_image\thas_hyperlink"] + [
        f"https://example.org/{index}\ttest\ttopic\t0\t0" for index in range(1012)
    ]
    original_names = list(parallel.FLORES_MEMBERS)
    output = io.BytesIO()
    members = {}
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for name, lines in zip(original_names, (en, uk, metadata), strict=True):
            raw = ("\n".join(lines) + "\n").encode()
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
            members[name] = (len(raw), parallel._digest(raw))
    raw = output.getvalue()
    (tmp_path / parallel.FLORES_EXPORT).write_bytes(raw)
    monkeypatch.setattr(parallel, "FLORES_MEMBERS", members)
    monkeypatch.setattr(parallel, "FLORES_ARCHIVE", {"size": len(raw), "sha256": parallel._digest(raw)})
    monkeypatch.setattr(parallel, "load_translation_split", lambda *_, **kwargs: (
        {"en": en[0].upper(), "uk": uk[1].upper()},
    ))
    bundle = tmp_path / "evaluation"
    receipt = parallel.materialize_flores_devtest(
        tmp_path, tmp_path, bundle, expected_train_manifest_sha256="f" * 64,
    )
    assert receipt["train_overlap_rows_excluded"] == 2
    assert receipt["license"] == "CC-BY-SA-4.0"
    assert receipt["evaluation_only"] is True
    selected = (bundle / "devtest.jsonl").read_text(encoding="utf-8")
    assert len(selected.splitlines()) == 100
    assert "flores200:devtest:0001" not in selected and "flores200:devtest:0002" not in selected
