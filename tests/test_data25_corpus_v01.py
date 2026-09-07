import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "twelve_six" / "data" / "corpus_v01.py"
spec = importlib.util.spec_from_file_location("corpus_v01", MODULE)
corpus_v01 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(corpus_v01)


def _small_config(tmp_path):
    config = json.loads((ROOT / "configs" / "data" / "corpus_v01.json").read_text(encoding="utf-8"))
    config["target_train_byte_tokens"] = {"uk": 12000, "en": 12000, "code": 12000}
    config["project_authored"]["max_candidates_per_stratum"] = 1000
    repo = tmp_path / "repo"
    config_path = repo / "configs" / "data" / "corpus.json"
    config_path.parent.mkdir(parents=True)
    external = repo / "data" / "external"
    external.mkdir(parents=True)
    (external / "external_sources.json").write_text((ROOT / "data" / "external" / "external_sources.json").read_text(encoding="utf-8"), encoding="utf-8")
    (external / "reserved_fingerprints.json").write_text((ROOT / "data" / "external" / "reserved_fingerprints.json").read_text(encoding="utf-8"), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
    return config_path


def test_full_rebuild_identity_and_shards_are_stable(tmp_path):
    manifest = corpus_v01.verify_rebuild(
        _small_config(tmp_path),
        tmp_path / "a",
        tmp_path / "b",
    )
    assert len(manifest["corpus_identity_sha256"]) == 64
    assert manifest["train_validation_content_overlap"] == 0
    assert manifest["external_training_eligible_sources"] == 0
    assert manifest["truth_boundary"]["contains_external_training_data"] is False
    assert manifest["truth_boundary"]["contains_project_authored_data"] is True
    assert sum(item["byte_tokens"] for item in manifest["shards"]) > 36_000
    assert manifest["by_modality"]["natural"]["documents"] > 0
    assert manifest["by_modality"]["code"]["documents"] > 0


def test_validation_never_enters_training(tmp_path):
    manifest = corpus_v01.build_corpus(
        _small_config(tmp_path),
        output_dir=tmp_path / "build",
    )
    train_hashes = set()
    validation_hashes = set()
    for shard in manifest["shards"]:
        for line in (tmp_path / "build" / shard["path"]).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            target = train_hashes if row["split"] == "train" else validation_hashes
            target.add(row["content_sha256"])
    assert train_hashes.isdisjoint(validation_hashes)


def test_registry_identity_is_fail_closed():
    registry = {"schema_version": "12-6.external-source-registry.v1", "sources": [], "registry_identity_sha256": "0" * 64}
    try:
        corpus_v01.eligible_external(registry)
    except corpus_v01.CorpusBuildError:
        pass
    else:
        raise AssertionError("tampered external registry must fail closed")


def test_reserved_eval_fingerprint_is_removed(tmp_path):
    config = json.loads((ROOT / "configs" / "data" / "corpus_v01.json").read_text(encoding="utf-8"))
    config["target_train_byte_tokens"] = {"uk": 4000, "en": 4000, "code": 4000}
    config["project_authored"]["max_candidates_per_stratum"] = 500
    config_path = tmp_path / "repo" / "configs" / "data" / "corpus.json"
    config_path.parent.mkdir(parents=True)
    data_dir = tmp_path / "repo" / "data" / "external"
    data_dir.mkdir(parents=True)
    config["external_registry"] = "data/external/external_sources.json"
    config["reserved_registry"] = "data/external/reserved_fingerprints.json"
    (data_dir / "external_sources.json").write_text((ROOT / "data" / "external" / "external_sources.json").read_text(encoding="utf-8"), encoding="utf-8")

    text = corpus_v01.norm(corpus_v01.authored_text("uk", 0), False)
    digest = corpus_v01.sha(text.encode())
    core = {
        "schema_version": "12-6.reserved-fingerprints.v1",
        "sets": [{
            "set_id": "test-reserved",
            "version": "1",
            "source_id": "test",
            "purpose": "evaluation",
            "normalized_sha256": [digest],
        }],
    }
    reserved = {**core, "registry_identity_sha256": corpus_v01.sha(corpus_v01.cjson(core))}
    (data_dir / "reserved_fingerprints.json").write_text(json.dumps(reserved, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    manifest = corpus_v01.build_corpus(config_path, output_dir=tmp_path / "out")
    assert manifest["counters"].get("reserved_eval_rejected_documents", 0) >= 1
