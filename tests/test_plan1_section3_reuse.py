import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.third_party_reuse import (
    prepare_trusted_base_checkpoint,
    require_reviewed_code,
    validate_base_lineage,
    validate_reuse_catalog,
    verify_reviewed_code_archive,
)

RAW = (Path(__file__).resolve().parents[1] /
       "configs/control/plan1_reuse_assets_v1.json").read_bytes()
GENESIS = {
    "genesis_id": "a" * 64,
    "model_spec_sha256": "b" * 64,
    "init_spec_sha256": "c" * 64,
    "origin": "local_random_init",
}


def wire(obj):
    return json.dumps(obj, sort_keys=True).encode()


def lineage():
    return {
        "schema_version": 1,
        "genesis_id": "a" * 64,
        "model_spec_sha256": "b" * 64,
        "init_spec_sha256": "c" * 64,
        "head_id": "d" * 64,
        "checkpoints": [
            {"id": "a" * 64, "parents": [], "origin": "canonical_base"},
            {"id": "d" * 64, "parents": ["a" * 64], "origin": "canonical_base"},
        ],
    }


def test_reuse_catalog_is_quarantined_and_restart_stable():
    assert len(validate_reuse_catalog(RAW)["assets"]) == 13
    assert validate_reuse_catalog(RAW) == validate_reuse_catalog(bytes(RAW))
    with pytest.raises(ValueError, match="unqualified"):
        require_reviewed_code(
            RAW, "pytorch", "e" * 64,
            independently_pinned_catalog_sha256=hashlib.sha256(RAW).hexdigest(),
        )
    with pytest.raises(ValueError, match="unknown external"):
        require_reviewed_code(
            RAW, "ghost", "e" * 64,
            independently_pinned_catalog_sha256=hashlib.sha256(RAW).hexdigest(),
        )


def test_no_second_authority_and_no_untrusted_catalog():
    obj = json.loads(RAW)
    obj["canonical_authorities"]["model"] = "vendor.model"
    with pytest.raises(ValueError, match="second canonical"):
        validate_reuse_catalog(wire(obj))
    obj = json.loads(RAW)
    obj["assets"].append(copy.deepcopy(obj["assets"][0]))
    with pytest.raises(ValueError, match="duplicate asset"):
        validate_reuse_catalog(wire(obj))
    with pytest.raises(ValueError, match="duplicate JSON"):
        validate_reuse_catalog(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="invalid strict"):
        validate_reuse_catalog(b'{"schema_version":NaN}')


def test_exact_code_license_security_rights_and_hash_check():
    obj = json.loads(RAW)
    item = obj["assets"][0]
    item.update(status="REVIEWED_CODE_ONLY", upstream_url="https://example.org",
                version="1.0", source_sha256="e"*64, license_spdx="MIT",
                license_evidence_sha256="f"*64, security_posture="REVIEWED",
                data_rights="NOT_APPLICABLE_CODE")
    pin = hashlib.sha256(wire(obj)).hexdigest()
    assert require_reviewed_code(
        wire(obj), "pytorch", "e" * 64, independently_pinned_catalog_sha256=pin
    )["version"] == "1.0"
    with pytest.raises(ValueError, match="approved catalog"):
        require_reviewed_code(
            wire(obj), "pytorch", "e" * 64,
            independently_pinned_catalog_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="source hash drift"):
        require_reviewed_code(
            wire(obj), "pytorch", "d" * 64, independently_pinned_catalog_sha256=pin
        )
    for key, bad in [("model_weights", "PRETRAINED"), ("license_spdx", "UNKNOWN"),
                     ("data_rights", "ALLOWED"), ("source_sha256", "z"*64),
                     ("security_posture", "UNKNOWN")]:
        other = copy.deepcopy(obj)
        other["assets"][0][key] = bad
        with pytest.raises(ValueError):
            validate_reuse_catalog(wire(other))


def test_closed_canonical_lineage_is_deterministic():
    assert validate_base_lineage(lineage(), GENESIS) == validate_base_lineage(
        copy.deepcopy(lineage()), copy.deepcopy(GENESIS)
    )


def test_foreign_root_unknown_parent_cycle_and_duplicate_are_rejected():
    cases = []
    m = lineage()
    m["checkpoints"][1]["origin"] = "pretrained"
    cases.append(m)
    m = lineage()
    m["checkpoints"][1]["parents"] = ["e"*64]
    cases.append(m)
    m = lineage()
    m["checkpoints"][0]["parents"] = ["d"*64]
    cases.append(m)
    m = lineage()
    m["checkpoints"][1]["parents"] = []
    cases.append(m)
    m = lineage()
    m["checkpoints"].append(copy.deepcopy(m["checkpoints"][0]))
    cases.append(m)
    m = lineage()
    m["head_id"] = "0"*64
    cases.append(m)
    m = lineage()
    m["schema_version"] = True
    cases.append(m)
    for bad in cases:
        with pytest.raises(ValueError):
            validate_base_lineage(bad, GENESIS)
    with pytest.raises(ValueError):
        validate_base_lineage(lineage(), dict(GENESIS, origin="foreign_pretrained"))


class TinyModel:
    def __init__(self, value=1.0):
        self.weights = np.array([value, value + 1], dtype=np.float64)
        self.mutations = 0

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        self.mutations += 1
        self.weights = state["weights"].copy()


def real_checkpoint_case(tmp_path):
    from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint

    init_spec_hash = "c" * 64
    identity = CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "tiny-reuse-contract", "width": 2},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"init_spec_sha256": init_spec_hash},
        seed=17,
        precision="float64-test",
        step=0,
        tokens_seen=0,
        optimizer={"name": "test-no-optimizer"},
        scheduler=None,
    )
    root = tmp_path / "scratch"
    manifest = save_checkpoint(root, model=TinyModel(), identity=identity)
    genesis = {
        "genesis_id": "a" * 64,
        "model_spec_sha256": manifest["identity"]["model_spec_hash"],
        "init_spec_sha256": init_spec_hash,
        "origin": "local_random_init",
    }
    chain = {
        "schema_version": 1,
        "genesis_id": genesis["genesis_id"],
        "model_spec_sha256": genesis["model_spec_sha256"],
        "init_spec_sha256": genesis["init_spec_sha256"],
        "head_id": manifest["checkpoint_id"],
        "checkpoints": [
            {"id": genesis["genesis_id"], "parents": [], "origin": "canonical_base"},
            {
                "id": manifest["checkpoint_id"],
                "parents": [genesis["genesis_id"]],
                "origin": "canonical_base",
            },
        ],
    }
    genesis_bytes = wire(genesis)
    return root, chain, genesis_bytes, hashlib.sha256(genesis_bytes).hexdigest()


def test_real_checkpoint_ingress_and_canonical_restore(tmp_path):
    from twelve_six.checkpoint import load_verified_checkpoint

    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    verified = prepare_trusted_base_checkpoint(
        root,
        lineage_bytes=wire(chain),
        trusted_genesis_bytes=genesis_bytes,
        expected_genesis_sha256=pin,
    )
    target = TinyModel(99.0)
    result = load_verified_checkpoint(verified, model=target, restore_rng=False)
    assert target.mutations == 1
    np.testing.assert_array_equal(target.weights, [1.0, 2.0])
    assert result.manifest["checkpoint_id"] == chain["head_id"]


def test_unpinned_foreign_or_drifted_base_is_rejected_pre_mutation(tmp_path):
    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    target = TinyModel(99.0)
    bad_cases = []
    forged = copy.deepcopy(chain)
    forged["head_id"] = "d" * 64
    forged["checkpoints"][1]["id"] = "d" * 64
    bad_cases.append((forged, genesis_bytes, pin))
    foreign = copy.deepcopy(chain)
    foreign["checkpoints"][1]["origin"] = "pretrained"
    bad_cases.append((foreign, genesis_bytes, pin))
    wrong_model = copy.deepcopy(chain)
    wrong_model["model_spec_sha256"] = "b" * 64
    bad_cases.append((wrong_model, genesis_bytes, pin))
    bad_cases.append((chain, genesis_bytes, "f" * 64))
    for lineage_value, trusted_bytes, expected in bad_cases:
        with pytest.raises(ValueError):
            prepare_trusted_base_checkpoint(
                root,
                lineage_bytes=wire(lineage_value),
                trusted_genesis_bytes=trusted_bytes,
                expected_genesis_sha256=expected,
            )
    assert target.mutations == 0


def test_actual_checkpoint_tamper_and_init_drift_fail_closed(tmp_path):
    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    init_drift = copy.deepcopy(chain)
    init_drift["init_spec_sha256"] = "0" * 64
    trusted_init_drift = json.loads(genesis_bytes)
    trusted_init_drift["init_spec_sha256"] = "0" * 64
    changed_genesis = wire(trusted_init_drift)
    with pytest.raises(ValueError, match="InitSpec"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(init_drift),
            trusted_genesis_bytes=changed_genesis,
            expected_genesis_sha256=hashlib.sha256(changed_genesis).hexdigest(),
        )
    with (root / "weights.safetensors").open("ab") as stream:
        stream.write(b"foreign checkpoint bytes")
    with pytest.raises(Exception, match="checksum|size mismatch"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(chain),
            trusted_genesis_bytes=genesis_bytes,
            expected_genesis_sha256=pin,
        )


def test_large_cyclic_lineage_fails_boundedly():
    m = lineage()
    m["checkpoints"] *= 2049
    with pytest.raises(ValueError, match="complete checkpoint graph"):
        validate_base_lineage(m, GENESIS)


def test_real_code_archive_license_and_review_pin_gate(tmp_path):
    source = tmp_path / "code.whl"
    notice = tmp_path / "LICENSE"
    source.write_bytes(b"audited adapter package bytes")
    notice.write_bytes(b"MIT license text fixture")
    catalog = json.loads(RAW)
    catalog["assets"][0].update(
        {
            "status": "REVIEWED_CODE_ONLY",
            "upstream_url": "https://example.org/audit-source",
            "version": "1.0.0+reviewed",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "license_spdx": "MIT",
            "license_evidence_sha256": hashlib.sha256(notice.read_bytes()).hexdigest(),
            "security_posture": "REVIEWED",
            "data_rights": "NOT_APPLICABLE_CODE",
        }
    )
    pinned = hashlib.sha256(wire(catalog)).hexdigest()
    assert verify_reviewed_code_archive(
        wire(catalog), "pytorch", source_archive=source, license_file=notice,
        independently_pinned_catalog_sha256=pinned,
    )["version"] == "1.0.0+reviewed"
    with pytest.raises(ValueError, match="approved catalog"):
        verify_reviewed_code_archive(
            wire(catalog), "pytorch", source_archive=source, license_file=notice,
            independently_pinned_catalog_sha256="0" * 64,
        )
    notice.write_bytes(b"modified license content")
    with pytest.raises(ValueError, match="license bytes drifted"):
        verify_reviewed_code_archive(
            wire(catalog), "pytorch", source_archive=source, license_file=notice,
            independently_pinned_catalog_sha256=pinned,
        )
    notice.write_bytes(b"MIT license text fixture")
    source.write_bytes(b"unapproved replacement")
    with pytest.raises(ValueError, match="source hash drift"):
        verify_reviewed_code_archive(
            wire(catalog), "pytorch", source_archive=source, license_file=notice,
            independently_pinned_catalog_sha256=pinned,
        )


def test_source_archive_symlink_is_not_admitted(tmp_path):
    source = tmp_path / "package.tar"
    source.write_bytes(b"package")
    link = tmp_path / "package.alias"
    link.symlink_to(source.name)
    from twelve_six.third_party_reuse import _hash_regular_archive

    with pytest.raises(ValueError, match="non-symlink"):
        _hash_regular_archive(link)


def test_installed_wheel_record_integrity_and_negative_recovery(monkeypatch, tmp_path):
    """Independent pin + real installed bytes, including tamper and restart cases."""
    import base64
    from importlib import metadata

    from twelve_six.third_party_reuse import verify_installed_wheel_record

    package_path = tmp_path / "example" / "__init__.py"
    package_path.parent.mkdir()
    package_path.write_bytes(b"abcd")
    digest = base64.urlsafe_b64encode(hashlib.sha256(b"abcd").digest()).rstrip(b"=").decode()
    good_record = (
        f"example/__init__.py,sha256={digest},4\n"
        "example-1.2.3.dist-info/RECORD,,\n"
    )

    class FakeDistribution:
        def __init__(self):
            self.metadata = {"Name": "example"}
            self.version = "1.2.3"
            self.record = good_record

        def read_text(self, filename):
            assert filename == "RECORD"
            return self.record

        def locate_file(self, name):
            return tmp_path / name

    installed = FakeDistribution()
    monkeypatch.setattr(metadata, "distribution", lambda name: installed)
    pin = hashlib.sha256(good_record.encode()).hexdigest()

    def check(*, record_pin=pin):
        return verify_installed_wheel_record(
            "example", expected_version="1.2.3",
            independently_pinned_record_sha256=record_pin,
        )

    result = check()
    assert result == {
        "distribution": "example",
        "version": "1.2.3",
        "record_sha256": pin,
        "verified_files": 1,
    }
    assert check() == result  # independently restart/recheck actual bytes

    with pytest.raises(ValueError, match="independent"):
        check(record_pin="")
    with pytest.raises(ValueError, match="RECORD pin drift"):
        check(record_pin="0" * 64)

    installed.version = "1.2.4"
    with pytest.raises(ValueError, match="version drift"):
        check()
    installed.version = "1.2.3"

    package_path.write_bytes(b"wxyz")  # same size; RECORD remains unchanged
    with pytest.raises(ValueError, match="contents drift"):
        check()
    package_path.write_bytes(b"abcd")
    assert check() == result  # repaired bytes independently requalify

    unsafe_record = "../escape,sha256=" + digest + ",4\n"
    unsafe_record += "example-1.2.3.dist-info/RECORD,,\n"
    installed.record = unsafe_record
    with pytest.raises(ValueError, match="unsafe"):
        check(record_pin=hashlib.sha256(unsafe_record.encode()).hexdigest())

    unpinned_record = "example/__init__.py,,\nexample-1.2.3.dist-info/RECORD,,\n"
    installed.record = unpinned_record
    with pytest.raises(ValueError, match="unhashed"):
        check(record_pin=hashlib.sha256(unpinned_record.encode()).hexdigest())
