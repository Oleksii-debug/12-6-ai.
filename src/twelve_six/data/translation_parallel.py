"""Materialize and read the pinned Tatoeba English–Ukrainian text export."""

from __future__ import annotations

import argparse
import bz2
import hashlib
import io
import json
import re
import tarfile
import unicodedata
from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

SCHEMA = "12-6.translation-parallel.v1"
SPLITS = ("train", "validation", "test")
DEFAULT_COUNTS = {"train": 20_000, "validation": 1_000, "test": 1_000}
SNAPSHOT = "tatoeba-en-uk-2026-11-07"
BASE = "https://downloads.tatoeba.org/exports/per_language/"
EN_EXPORT = "eng_sentences_detailed.tsv.bz2"
UK_EXPORT = "ukr_sentences_detailed.tsv.bz2"
LINK_EXPORT = "eng-ukr_links.tsv.bz2"
FLORES_EXPORT = "flores200_dataset.tar.gz"
FLORES_ARCHIVE = {
    "url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
    "sha256": "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6",
    "size": 25_585_843,
}
FLORES_MEMBERS = {
    "./flores200_dataset/devtest/eng_Latn.devtest": (
        133_108, "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88",
    ),
    "./flores200_dataset/devtest/ukr_Cyrl.devtest": (
        246_954, "7bb8f160a455fca27032bdd292dd65838b7aa5a8324c6aedd03ccdf92e20dbc4",
    ),
    "./flores200_dataset/metadata_devtest.tsv": (
        105_761, "8edac47f861fcf6b54dad2068c10ba8f741da8b8676861c6bc27fcd9f5483b36",
    ),
}
EXPORTS = {
    EN_EXPORT: {
        "url": BASE + "eng/" + EN_EXPORT,
        "sha256": "afec98e4c44a72f8fc709ce4d672fd3ab7f9ca731402f5d35e384b0b21b207c5",
        "size": 34_983_679,
    },
    UK_EXPORT: {
        "url": BASE + "ukr/" + UK_EXPORT,
        "sha256": "c9b457b147b2db94526a56f2d2543f100c7fe8656cdf0ba2dda4d7cad572f505",
        "size": 3_088_100,
    },
    LINK_EXPORT: {
        "url": BASE + "eng/" + LINK_EXPORT,
        "sha256": "452ed45b8217932b681120620cbcf6d36f2fe2376c365578c33c07e49cda14d6",
        "size": 1_259_190,
    },
}
RIGHTS = {
    "export_license": "CC-BY-2.0-FR",
    "license_url": "https://creativecommons.org/licenses/by/2.0/fr/",
    "project_url": "https://tatoeba.org/",
    "downloads_url": "https://tatoeba.org/en/downloads",
    "terms_url": "https://tatoeba.org/en/terms_of_use",
    "links_url": "https://en.wiki.tatoeba.org/articles/show/how-to-be-a-good-contributor-in-tatoeba",
    "attribution": "Tatoeba contributors identified for each sentence in the rows.",
}
NOTICE = (
    "English–Ukrainian sentence pairs from Tatoeba contributors, https://tatoeba.org/.\n"
    "Tatoeba publishes the selected text and link exports under CC BY 2.0 France.\n"
    "https://creativecommons.org/licenses/by/2.0/fr/\n"
    "Each row names both sentence contributors and links to both original sentences.\n"
    "Changes: selected linked sentences, filtered and deduplicated pairs, and grouped "
    "them into separate splits. The selected sentence text is preserved.\n"
).encode()
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_POLICY = "normalized-shared-sentence-connected-components-v1"
_ROW_KEYS = {"pair_id", "en", "uk", "component_sha256", "en_source", "uk_source"}
_MAX_TEXT = 4_096
_MAX_MEMBER = 100_000_000


class TranslationDatasetError(ValueError):
    """The exact translation dataset or its source bytes failed verification."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise TranslationDatasetError(message)


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n").encode()


def _normalized(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(ch for ch in folded if ch.isalnum())


def _read(directory: Path, name: str, limit: int) -> bytes:
    path = directory / name
    _require(path.is_file() and not path.is_symlink(), f"missing or linked dataset member: {name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    _require(len(raw) <= limit, f"oversized dataset member: {name}")
    return raw


def _source_fields(directory: Path, name: str) -> Iterator[list[str]]:
    expected = EXPORTS[name]
    raw = _read(directory, name, expected["size"])
    _require(len(raw) == expected["size"] and _digest(raw) == expected["sha256"],
             f"Tatoeba source snapshot changed: {name}")
    with io.TextIOWrapper(bz2.BZ2File(io.BytesIO(raw)), encoding="utf-8", newline="") as stream:
        for line in stream:
            yield line.removesuffix("\n").removesuffix("\r").split("\t")


def _sentences(directory: Path, name: str, language: str, wanted: set[int]) -> dict[int, tuple[str, str]]:
    result = {}
    for fields in _source_fields(directory, name):
        _require(len(fields) == 6 and fields[1] == language, f"invalid Tatoeba sentence row in {name}")
        sentence_id = int(fields[0])
        text, contributor = fields[2:4]
        if sentence_id in wanted and contributor.strip() and 0 < len(text) <= _MAX_TEXT and _normalized(text):
            _require(sentence_id not in result, f"duplicate Tatoeba sentence ID in {name}")
            result[sentence_id] = (text, contributor)
    return result


def _component_root(parent: dict[str, str], node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _component_split(component_sha256: str) -> str:
    bucket = int(component_sha256[:8], 16) % 100
    if bucket < 80:
        return "train"
    return "validation" if bucket < 90 else "test"


def _source(sentence_id: int, contributor: str) -> dict[str, Any]:
    return {
        "sentence_id": sentence_id,
        "contributor": contributor,
        "url": f"https://tatoeba.org/en/sentences/show/{sentence_id}",
    }


def _pair_id(en_id: int, uk_id: int) -> str:
    return f"tatoeba:{en_id}:{uk_id}"


def _select(directory: Path, counts: dict[str, int]) -> tuple[dict[str, list[dict[str, Any]]], int, int]:
    links = sorted({tuple(map(int, fields)) for fields in _source_fields(directory, LINK_EXPORT)})
    _require(bool(links) and all(len(pair) == 2 and min(pair) > 0 for pair in links), "invalid direct link rows")
    english = _sentences(directory, EN_EXPORT, "eng", {pair[0] for pair in links})
    ukrainian = _sentences(directory, UK_EXPORT, "ukr", {pair[1] for pair in links})
    en_keys = {key: _normalized(value[0]) for key, value in english.items()}
    uk_keys = {key: _normalized(value[0]) for key, value in ukrainian.items()}
    parent = {key: key for key in set(en_keys.values()) | set(uk_keys.values())}
    seen: set[tuple[str, str]] = set()
    pairs = []
    for en_id, uk_id in links:
        if en_id not in en_keys or uk_id not in uk_keys:
            continue
        en_key, uk_key = en_keys[en_id], uk_keys[uk_id]
        if (en_key, uk_key) in seen:
            continue
        seen.add((en_key, uk_key))
        left, right = sorted((_component_root(parent, en_key), _component_root(parent, uk_key)))
        parent[right] = left
        pairs.append((en_id, uk_id))

    selected: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    pairs.sort(key=lambda pair: _digest(_pair_id(*pair).encode()))
    for en_id, uk_id in pairs:
        component = _digest(_component_root(parent, en_keys[en_id]).encode())
        split = _component_split(component)
        rows = selected[split]
        if len(rows) < counts[split]:
            en, en_author = english[en_id]
            uk, uk_author = ukrainian[uk_id]
            rows.append({
                "pair_id": _pair_id(en_id, uk_id),
                "en": en,
                "uk": uk,
                "component_sha256": component,
                "en_source": _source(en_id, en_author),
                "uk_source": _source(uk_id, uk_author),
            })
    _require({name: len(rows) for name, rows in selected.items()} == counts,
             "pinned corpus does not supply the requested split sizes")
    return selected, len(links), len(pairs)


def _validate_row(row: Any, split: str) -> None:
    _require(type(row) is dict and row.keys() == _ROW_KEYS, "invalid translation row")
    component = row["component_sha256"]
    _require(isinstance(component, str) and _HEX.fullmatch(component) is not None
             and _component_split(component) == split, "translation row in the wrong split")
    sources = []
    for lang in ("en", "uk"):
        text, ref = row[lang], row[f"{lang}_source"]
        _require(isinstance(text, str) and 0 < len(text) <= _MAX_TEXT and bool(_normalized(text)),
                 "invalid translation text")
        _require(type(ref) is dict and ref.keys() == {"sentence_id", "contributor", "url"},
                 "missing sentence attribution")
        sentence_id, author = ref["sentence_id"], ref["contributor"]
        _require(type(sentence_id) is int and sentence_id > 0 and isinstance(author, str) and bool(author.strip())
                 and ref == _source(sentence_id, author), "invalid sentence attribution")
        sources.append(sentence_id)
    _require(row["pair_id"] == _pair_id(*sources), "translation pair identity changed")


def _audit_splits(splits: dict[str, tuple[dict[str, Any], ...] | list[dict[str, Any]]]) -> None:
    seen_text: dict[str, str] = {}
    seen_id: dict[int, str] = {}
    seen_pairs: set[str] = set()
    for split, rows in splits.items():
        for row in rows:
            _validate_row(row, split)
            _require(row["pair_id"] not in seen_pairs, "duplicate translation pair")
            seen_pairs.add(row["pair_id"])
            for lang in ("en", "uk"):
                text_key, source_id = _normalized(row[lang]), row[f"{lang}_source"]["sentence_id"]
                _require(seen_text.setdefault(text_key, split) == split
                         and seen_id.setdefault(source_id, split) == split,
                         "translation split contamination")


def materialize_translation_dataset(
    sources: Path, destination: Path, *, counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Rebuild a fresh split bundle from the exact original compressed exports."""
    counts = dict(DEFAULT_COUNTS) if counts is None else counts
    _require(type(counts) is dict and counts.keys() == set(SPLITS)
             and all(type(value) is int and value > 0 for value in counts.values()), "invalid split counts")
    _require(not destination.exists() and not destination.is_symlink(), "dataset destination already exists")
    splits, direct_links, eligible_pairs = _select(sources, counts)
    _audit_splits(splits)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "bundle"
        staged.mkdir()
        members = {}
        for name, rows in splits.items():
            filename = name + ".jsonl"
            raw = b"".join(map(_canonical, rows))
            (staged / filename).write_bytes(raw)
            members[name] = {"file": filename, "sha256": _digest(raw), "rows": len(rows)}
        (staged / "NOTICE.txt").write_bytes(NOTICE)
        manifest = {
            "schema": SCHEMA,
            "snapshot": SNAPSHOT,
            "sources": EXPORTS,
            "rights": RIGHTS,
            "notice_sha256": _digest(NOTICE),
            "split_policy": _POLICY,
            "direct_links": direct_links,
            "eligible_deduplicated_pairs": eligible_pairs,
            "splits": members,
        }
        raw = _canonical(manifest)
        (staged / "manifest.json").write_bytes(raw)
        staged.rename(destination)
    return {"manifest_sha256": _digest(raw), **manifest}


def _open_manifest(directory: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    _require(isinstance(expected_manifest_sha256, str) and _HEX.fullmatch(expected_manifest_sha256) is not None,
             "an independently supplied translation manifest SHA-256 is required")
    raw = _read(directory, "manifest.json", 100_000)
    _require(_digest(raw) == expected_manifest_sha256, "translation manifest SHA-256 mismatch")
    manifest = json.loads(raw)
    _require(type(manifest) is dict and raw == _canonical(manifest) and manifest.get("schema") == SCHEMA
             and manifest.get("snapshot") == SNAPSHOT and manifest.get("sources") == EXPORTS
             and manifest.get("rights") == RIGHTS and manifest.get("split_policy") == _POLICY,
             "unsupported translation dataset manifest")
    _require(_read(directory, "NOTICE.txt", 10_000) == NOTICE and manifest.get("notice_sha256") == _digest(NOTICE),
             "translation attribution notice changed")
    _require(type(manifest.get("splits")) is dict and manifest["splits"].keys() == set(SPLITS),
             "invalid translation split inventory")
    return manifest


def _load_split(directory: Path, manifest: dict[str, Any], split: str) -> tuple[dict[str, Any], ...]:
    _require(split in SPLITS, f"unknown translation split: {split}")
    member = manifest["splits"][split]
    filename = split + ".jsonl"
    _require(type(member) is dict and member.keys() == {"file", "sha256", "rows"}
             and member["file"] == filename and type(member["rows"]) is int and member["rows"] > 0,
             "invalid translation split manifest")
    raw = _read(directory, filename, _MAX_MEMBER)
    _require(_digest(raw) == member["sha256"], f"translation split SHA-256 mismatch: {split}")
    lines = raw.splitlines(keepends=True)
    _require(len(lines) == member["rows"], f"translation split row count mismatch: {split}")
    rows = []
    for line in lines:
        row = json.loads(line)
        _require(_canonical(row) == line, "translation row is not canonical JSONL")
        _validate_row(row, split)
        rows.append(row)
    _require(len({row["pair_id"] for row in rows}) == len(rows), "duplicate translation pair")
    return tuple(rows)


def load_translation_split(
    directory: Path, *, expected_manifest_sha256: str, split: str,
) -> tuple[dict[str, Any], ...]:
    """Verify and read one requested split without opening the other split payloads."""
    return _load_split(directory, _open_manifest(directory, expected_manifest_sha256), split)


def verify_translation_dataset(directory: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
    """Check the complete frozen bundle, including the cross-split firewall."""
    manifest = _open_manifest(directory, expected_manifest_sha256)
    _audit_splits({name: _load_split(directory, manifest, name) for name in SPLITS})
    return {"manifest_sha256": expected_manifest_sha256, **manifest}


def materialize_flores_devtest(
    archive_directory: Path, train_directory: Path, destination: Path, *,
    expected_train_manifest_sha256: str,
) -> dict[str, Any]:
    """Keep a deterministic independent FLORES-200 evaluation sample outside training."""
    train = load_translation_split(
        train_directory, expected_manifest_sha256=expected_train_manifest_sha256, split="train",
    )
    seen = {_normalized(row[lang]) for row in train for lang in ("en", "uk")}
    raw = _read(archive_directory, FLORES_EXPORT, FLORES_ARCHIVE["size"])
    _require(len(raw) == FLORES_ARCHIVE["size"] and _digest(raw) == FLORES_ARCHIVE["sha256"],
             "FLORES-200 archive SHA-256 mismatch")
    texts = []
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for path, (size, digest) in FLORES_MEMBERS.items():
            member = archive.getmember(path)
            stream = archive.extractfile(member)
            _require(member.isfile() and member.size == size and stream is not None,
                     "invalid FLORES-200 archive member")
            assert stream is not None
            with stream:
                payload = stream.read(size + 1)
            _require(_digest(payload) == digest, "FLORES-200 archive member SHA-256 mismatch")
            texts.append(payload.decode("utf-8").splitlines())
    english, ukrainian, metadata = texts
    _require(len(english) == len(ukrainian) == len(metadata) - 1 == 1_012
             and metadata[0].split("\t") == ["URL", "domain", "topic", "has_image", "has_hyperlink"],
             "FLORES-200 devtest alignment changed")
    eligible = []
    overlap = 0
    for line, (en, uk, meta) in enumerate(zip(english, ukrainian, metadata[1:], strict=True), start=1):
        if _normalized(en) in seen or _normalized(uk) in seen:
            overlap += 1
            continue
        url, domain, topic, _, _ = meta.split("\t")
        eligible.append({
            "pair_id": f"flores200:devtest:{line:04d}",
            "en": en,
            "uk": uk,
            "source_url": url,
            "domain": domain,
            "topic": topic,
        })
    eligible.sort(key=lambda row: _digest(row["pair_id"].encode()))
    _require(len(eligible) >= 100, "insufficient independent FLORES-200 references")
    selected = b"".join(map(_canonical, eligible[:100]))
    notice = (
        "FLORES-200 English–Ukrainian devtest, Meta AI / NLLB Team and the linked source articles.\n"
        "https://github.com/facebookresearch/flores/tree/main/flores200\n"
        "CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/.\n"
        "Changes: selected 100 unchanged aligned text pairs after exact normalized overlap screening.\n"
        "Every row carries the original article URL. This bundle is reserved for evaluation.\n"
    ).encode()
    receipt = {
        "schema": "12-6.translation-flores-devtest.v1",
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "archive": FLORES_ARCHIVE,
        "source_members": {name: {"size": size, "sha256": digest}
                           for name, (size, digest) in FLORES_MEMBERS.items()},
        "tatoeba_train_manifest_sha256": expected_train_manifest_sha256,
        "source_rows": len(english),
        "train_overlap_rows_excluded": overlap,
        "selection": "first-100-by-sha256-of-original-line-pair-id",
        "normalization": "NFKC-casefold-alphanumeric",
        "evaluation_only": True,
        "file": "devtest.jsonl",
        "rows": 100,
        "sha256": _digest(selected),
        "notice_sha256": _digest(notice),
    }
    _require(not destination.exists() and not destination.is_symlink(), "evaluation destination already exists")
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / "evaluation"
        staged.mkdir()
        (staged / "devtest.jsonl").write_bytes(selected)
        (staged / "NOTICE.txt").write_bytes(notice)
        raw_receipt = _canonical(receipt)
        (staged / "receipt.json").write_bytes(raw_receipt)
        staged.rename(destination)
    return {"receipt_sha256": _digest(raw_receipt), **receipt}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build", help="Build from the pinned original exports")
    build.add_argument("--sources", required=True, type=Path)
    build.add_argument("--out", required=True, type=Path)
    build.add_argument("--train-count", type=int, default=DEFAULT_COUNTS["train"])
    verify = modes.add_parser("verify", help="Verify all three splits against an external digest")
    verify.add_argument("--directory", required=True, type=Path)
    verify.add_argument("--manifest-sha256", required=True)
    evaluation = modes.add_parser("flores-devtest", help="Create an isolated evaluation-only FLORES-200 sample")
    evaluation.add_argument("--archive-directory", required=True, type=Path)
    evaluation.add_argument("--train-directory", required=True, type=Path)
    evaluation.add_argument("--train-manifest-sha256", required=True)
    evaluation.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "build":
        result = materialize_translation_dataset(
            args.sources, args.out, counts={**DEFAULT_COUNTS, "train": args.train_count},
        )
    elif args.mode == "verify":
        result = verify_translation_dataset(args.directory, expected_manifest_sha256=args.manifest_sha256)
    else:
        result = materialize_flores_devtest(
            args.archive_directory, args.train_directory, args.out,
            expected_train_manifest_sha256=args.train_manifest_sha256,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
