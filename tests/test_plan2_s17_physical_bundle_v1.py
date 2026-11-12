"""Frozen tokenizer and admission authority regressions."""
from gzip import GzipFile, compress
from pathlib import Path

import pytest

from tools import plan2_s17_clean_rebuild_witness_v1 as witness
from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s17_physical_bundle_v1 as bundle
from tools.plan2_s15_real_32k_bpe_v1 import Real32kBpeDenied

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_tokenizer_reproduces_published_identity():
    tokenizer, receipt = frozen.load_frozen(ROOT)
    assert tokenizer.vocab_size == 32768
    assert receipt["source_manifest_sha256"] == frozen.S15_MANIFEST_SHA256
    sample = "Дані, dáta, data. print(42)"
    assert tokenizer.decode(tokenizer.encode(sample, add_bos=True, add_eos=True)) == sample


def test_changed_tokenizer_bytes_are_refused(tmp_path):
    target = tmp_path / frozen.ARTIFACT
    target.parent.mkdir(parents=True)
    with GzipFile(ROOT / frozen.ARTIFACT) as stream:
        raw = stream.read()
    target.write_bytes(compress(raw + b"\n", mtime=0))
    with pytest.raises(Real32kBpeDenied, match="tokenizer bytes differ"):
        frozen.load_frozen(tmp_path)


def test_modified_source_authority_is_denied_before_training_is_read(tmp_path, monkeypatch):
    read = bundle.books.read_checked
    changed_path = next(iter(bundle.SOURCE_AUTHORITY_BLOBS))

    def altered(root, path):
        raw = read(root, path)
        return raw + b"\n" if path == changed_path else raw

    monkeypatch.setattr(bundle.books, "read_checked", altered)
    with pytest.raises(bundle.BundleDenied, match="source authority changed"):
        bundle.stage(ROOT, tmp_path / "missing.jsonl", tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_modified_training_payload_is_denied_before_any_bundle_files(tmp_path):
    manifest = bundle.json.loads(bundle.books.read_checked(ROOT, bundle.S16_MANIFEST))
    train = tmp_path / "changed.jsonl"
    with train.open("wb") as stream:
        stream.truncate(manifest["payload_bytes"])
    with pytest.raises(bundle.BundleDenied, match="training payload differs"):
        bundle.stage(ROOT, train, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_reserved_roles_bind_both_languages_without_payload():
    admission = bundle.json.loads(bundle.books.read_checked(ROOT, bundle.S16_MANIFEST))
    roles = bundle.reservations.reserved_role_manifests(ROOT, admission)
    assert set(roles) == {"validation", "test"}
    assert [roles[role]["records"] for role in ("validation", "test")] == [10, 16]
    assert all("text" not in member for role in roles.values() for member in role["members"])


def test_rebuild_witness_reads_a_member_written_by_the_bundle(tmp_path):
    raw = b"verified shard bytes"
    bundle._publish(tmp_path, "shard.json", raw)
    assert witness.inventory(tmp_path) == ({"shard.json": witness.digest(raw)}, len(raw))
