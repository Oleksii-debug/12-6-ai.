import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from twelve_six.third_party_reuse import (
    _hash_regular_archive,
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



@pytest.mark.parametrize(
    ("field", "forged_value"),
    [
        ("data_rights", "ALLOWED"),
        ("model_weights", "PRETRAINED"),
        ("license_spdx", "MIT"),
        ("upstream_url", "https://example.org/forged"),
    ],
)
def test_quarantined_candidate_cannot_claim_rights_or_review(field, forged_value):
    """An unqualified catalog row may not disguise unreviewed rights as facts."""
    forged = json.loads(RAW)
    forged["assets"][0][field] = forged_value
    with pytest.raises(ValueError, match="candidate cannot claim"):
        validate_reuse_catalog(wire(forged))

@pytest.mark.parametrize(
    "untrusted_url",
    [
        "https://",
        "https:///missing-host",
        "https://127.0.0.1/source",
        "https://169.254.169.254/source",
        "https://192.168.1.1/source",
        "https://8.8.8.8/source",
        "https://127.0.1/source",
        "https://0.0.0.0/source",
        "https://user:secret@example.org/archive",
        "https://example.org/archive#unbound-fragment",
        " https://example.org/archive",
        "https://example.org/\\nambiguous",
        "https://example.org:invalid/archive",
        "https://example.org:0/archive",
        "https://example.org:/archive",
        "https://github.com\u200b.evil.example/archive",
        "https://github.com\u202eevil.example/archive",
        "https://github.com%2Eevil.example/archive",
        "https://./archive",
        "https://../archive",
        "https://example..org/archive",
        "https://-invalid.example/archive",
        "https://invalid-.example/archive",
        "https://example.org./archive",
        "https://" + "a" * 64 + ".example/archive",
    ],
)
def test_reviewed_source_requires_unambiguous_https_origin(untrusted_url):
    catalog = json.loads(RAW)
    catalog["assets"][0].update(
        status="REVIEWED_CODE_ONLY",
        upstream_url=untrusted_url,
        version="1.0.0",
        source_sha256="e" * 64,
        license_spdx="MIT",
        license_evidence_sha256="f" * 64,
        security_posture="REVIEWED",
        data_rights="NOT_APPLICABLE_CODE",
    )
    with pytest.raises(ValueError, match="missing HTTPS source"):
        validate_reuse_catalog(wire(catalog))


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



@pytest.mark.parametrize("boundary", [
    "adapter:",
    "adapter:pytorch.model_registry",
    "adapter:pytorch:checkpoint",
    "adapter:checkpoint",
])
def test_adapter_path_cannot_smuggle_duplicate_authority(boundary):
    candidate = json.loads(RAW)
    candidate["assets"][0]["replacement_boundary"] = boundary
    with pytest.raises(ValueError, match="adapter boundary"):
        validate_reuse_catalog(wire(candidate))


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
        expected_lineage_sha256=hashlib.sha256(wire(chain)).hexdigest(),
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
                expected_lineage_sha256=hashlib.sha256(wire(lineage_value)).hexdigest(),
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
            expected_lineage_sha256=hashlib.sha256(wire(init_drift)).hexdigest(),
        )
    with (root / "weights.safetensors").open("ab") as stream:
        stream.write(b"foreign checkpoint bytes")
    with pytest.raises(Exception, match="checksum|size mismatch"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(chain),
            trusted_genesis_bytes=genesis_bytes,
            expected_genesis_sha256=pin,
            expected_lineage_sha256=hashlib.sha256(wire(chain)).hexdigest(),
        )


def test_checkpoint_lineage_graph_graft_requires_separate_trust_pin(tmp_path):
    """A genuine genesis pin alone cannot authorize a modified ancestry graph."""
    root, chain, genesis_bytes, genesis_pin = real_checkpoint_case(tmp_path)
    approved_graph_pin = hashlib.sha256(wire(chain)).hexdigest()
    forged_graph = copy.deepcopy(chain)
    forged_graph["checkpoints"].append(
        {"id": "f" * 64, "parents": [chain["genesis_id"]], "origin": "canonical_base"}
    )
    # The structural validator alone allows a separately attached plausible child.
    assert validate_base_lineage(forged_graph, json.loads(genesis_bytes))
    with pytest.raises(ValueError, match="lineage digest mismatch"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(forged_graph),
            trusted_genesis_bytes=genesis_bytes,
            expected_genesis_sha256=genesis_pin,
            expected_lineage_sha256=approved_graph_pin,
        )
    with pytest.raises(ValueError, match="lineage SHA-256 required"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(chain),
            trusted_genesis_bytes=genesis_bytes,
            expected_genesis_sha256=genesis_pin,
            expected_lineage_sha256="",
        )


@pytest.mark.parametrize("bad_head", [None, [], {}, True, 1, "untrusted"])
def test_malformed_lineage_head_fails_closed_before_checkpoint_ingress(bad_head):
    """Forged JSON ancestry heads must fail as policy errors, not TypeError."""
    forged = lineage()
    forged["head_id"] = bad_head
    with pytest.raises(ValueError, match="invalid checkpoint head"):
        validate_base_lineage(forged, GENESIS)


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

    # A pinned RECORD for another distribution/version must not attest this one.
    for forged_self in (
        "foreign-1.2.3.dist-info/RECORD",
        "example-9.9.9.dist-info/RECORD",
    ):
        foreign_record = (
            f"example/__init__.py,sha256={digest},4\n"
            f"{forged_self},,\n"
        )
        installed.record = foreign_record
        foreign_pin = hashlib.sha256(foreign_record.encode()).hexdigest()
        with pytest.raises(ValueError, match="RECORD identity/version drift"):
            check(record_pin=foreign_pin)
    installed.record = good_record
    assert check() == result

    unsafe_record = "../escape,sha256=" + digest + ",4\n"
    unsafe_record += "example-1.2.3.dist-info/RECORD,,\n"
    installed.record = unsafe_record
    with pytest.raises(ValueError, match="unsafe"):
        check(record_pin=hashlib.sha256(unsafe_record.encode()).hexdigest())

    unpinned_record = "example/__init__.py,,\nexample-1.2.3.dist-info/RECORD,,\n"
    installed.record = unpinned_record
    with pytest.raises(ValueError, match="unhashed"):
        check(record_pin=hashlib.sha256(unpinned_record.encode()).hexdigest())

def test_same_size_in_place_mutation_during_archive_digest_is_rejected(monkeypatch, tmp_path):
    """A trusted digest must not attest to an already rewritten source archive."""
    import os

    from twelve_six.third_party_reuse import _hash_regular_archive

    archive = tmp_path / "reviewed.whl"
    archive.write_bytes(b"A" * 4096)
    original_read = os.read
    replaced = False

    def read_then_replace(fd, size):
        nonlocal replaced
        chunk = original_read(fd, size)
        if chunk and not replaced:
            archive.write_bytes(b"B" * 4096)
            replaced = True
        return chunk

    monkeypatch.setattr(os, "read", read_then_replace)
    with pytest.raises(ValueError, match="changed while reading"):
        _hash_regular_archive(archive)
    assert replaced


@pytest.mark.skipif(__import__("os").name == "nt", reason="POSIX FIFO race")
def test_fifo_swap_during_archive_open_rejected_without_block(monkeypatch, tmp_path):
    """Never block on a FIFO substituted after the regular-file lstat."""
    import os
    import stat

    archive = tmp_path / "source.whl"
    archive.write_bytes(b"approved source")
    original_open = os.open

    def substitute_fifo(path, flags, *args, **kwargs):
        assert flags & os.O_NONBLOCK, "opening substituted FIFO may hang"
        archive.unlink()
        os.mkfifo(archive)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", substitute_fifo)
    with pytest.raises(ValueError, match="changed while opening"):
        _hash_regular_archive(archive)
    assert stat.S_ISFIFO(archive.lstat().st_mode)


@pytest.mark.skipif(__import__("os").name == "nt", reason="open-file rename differs on Windows")
def test_pathname_swap_during_archive_digest_is_rejected(monkeypatch, tmp_path):
    """Descriptor stability must not bless hostile bytes swapped onto the pathname."""
    import os

    from twelve_six.third_party_reuse import _hash_regular_archive

    archive = tmp_path / "reviewed.whl"
    replacement = tmp_path / "replacement.whl"
    archive.write_bytes(b"trusted archive")
    replacement.write_bytes(b"untrusted replacement")
    original_read = os.read
    swapped = False

    def read_then_swap(fd, size):
        nonlocal swapped
        chunk = original_read(fd, size)
        if chunk and not swapped:
            os.replace(replacement, archive)
            swapped = True
        return chunk

    # Some filesystems update the old descriptor's ctime on rename, so the
    # existing fd check may catch the swap first. Force descriptor metadata
    # to remain stable to exercise the independent named-path invariant.
    original_fstat = os.fstat
    first_stat = None

    def stable_descriptor_stat(fd):
        nonlocal first_stat
        latest = original_fstat(fd)
        if first_stat is None:
            first_stat = latest
        return first_stat

    monkeypatch.setattr(os, "fstat", stable_descriptor_stat)
    monkeypatch.setattr(os, "read", read_then_swap)
    with pytest.raises(ValueError, match="path changed while reading"):
        _hash_regular_archive(archive)
    assert swapped
    assert archive.read_bytes() == b"untrusted replacement"


@pytest.mark.skipif(__import__("os").name == "nt", reason="directory swap differs on Windows")
def test_parent_directory_swap_during_archive_digest_is_rejected(monkeypatch, tmp_path):
    """Keep the same source inode but redirect its ancestor during a digest read."""
    import os

    parent = tmp_path / "review"
    parent.mkdir()
    archive = parent / "source.whl"
    archive.write_bytes(b"approved source")
    original_read = os.read
    swapped = False

    def swap_parent_after_read(fd, size):
        nonlocal swapped
        chunk = original_read(fd, size)
        if chunk and not swapped:
            moved = tmp_path / "original-review"
            parent.rename(moved)
            parent.symlink_to(moved, target_is_directory=True)
            swapped = True
        return chunk

    monkeypatch.setattr(os, "read", swap_parent_after_read)
    with pytest.raises(ValueError, match="parent directory"):
        _hash_regular_archive(archive)
    assert swapped and parent.is_symlink()


# These are upstream *expression-format* fixtures, NOT admitted dependencies.
# Even valid syntax requires independent source, binary, notices and security review.
@pytest.mark.parametrize(
    "expression",
    [
        "BSD-3-Clause AND 0BSD AND MIT AND CC0-1.0",
        (
            "Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause "
            "AND BSD-3-Clause AND BSL-1.0 AND ISC AND MIT AND Zlib"
        ),
    ],
)
def test_composite_permissive_spdx_requires_other_admission_evidence(expression):
    approved = json.loads(RAW)
    asset = approved["assets"][0]
    asset.update(
        status="REVIEWED_CODE_ONLY",
        upstream_url="https://example.org/independently-reviewed",
        version="1.0.0",
        source_sha256="e" * 64,
        license_spdx=expression,
        license_evidence_sha256="f" * 64,
        security_posture="REVIEWED",
        data_rights="NOT_APPLICABLE_CODE",
    )
    raw = wire(approved)
    assert validate_reuse_catalog(raw)["assets"][0]["license_spdx"] == expression
    with pytest.raises(ValueError, match="approved catalog"):
        require_reviewed_code(
            raw, "pytorch", "e" * 64,
            independently_pinned_catalog_sha256="0" * 64,
        )


@pytest.mark.parametrize(
    "forged",
    [
        "BSD-3-Clause OR GPL-3.0-only",
        "BSD-3-Clause AND GPL-3.0-only",
        "Apache-2.0 WITH Classpath-exception-2.0",
        "BSD-3-Clause AND ",
        " AND MIT",
        "MIT AND MIT",
        "MIT  AND BSD-3-Clause",
        "(BSD-3-Clause)",
        "MIT AND LicenseRef-Unreviewed",
    ],
)
def test_unknown_or_ambiguous_composite_license_fails_closed(forged):
    approved = json.loads(RAW)
    approved["assets"][0].update(
        status="REVIEWED_CODE_ONLY",
        upstream_url="https://example.org/independently-reviewed",
        version="1.0.0",
        source_sha256="e" * 64,
        license_spdx=forged,
        license_evidence_sha256="f" * 64,
        security_posture="REVIEWED",
        data_rights="NOT_APPLICABLE_CODE",
    )
    with pytest.raises(ValueError, match="unknown/incompatible license"):
        validate_reuse_catalog(wire(approved))


@pytest.mark.parametrize(
    "unresolved_version",
    [">=2.5", "2.5.*", "latest", "*", "~=2.5", "2.5; python_version > '3.11'",
     "https://example.org/package.whl", "2.5 || latest"],
)
def test_reviewed_code_requires_literal_exact_version(unresolved_version):
    """A review cannot pin a range, resolver instruction, or moving alias."""
    catalog = json.loads(RAW)
    item = catalog["assets"][0]
    item.update(
        status="REVIEWED_CODE_ONLY", upstream_url="https://example.org/reviewed",
        version="2.5.1+local", source_sha256="e" * 64,
        license_spdx="MIT", license_evidence_sha256="f" * 64,
        security_posture="REVIEWED", data_rights="NOT_APPLICABLE_CODE",
    )
    assert validate_reuse_catalog(wire(catalog))["assets"][0]["version"] == "2.5.1+local"
    item["version"] = unresolved_version
    with pytest.raises(ValueError, match="exact version required"):
        validate_reuse_catalog(wire(catalog))

def test_reviewed_archive_parent_symlink_is_rejected(tmp_path):
    """The leaf must not be certified through a redirected parent directory."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "source.tar").write_bytes(b"known source bytes")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(real, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks unavailable on this platform")

    with pytest.raises(ValueError, match="parent directory"):
        _hash_regular_archive(alias / "source.tar")



def test_wheel_record_current_environment_console_scripts(monkeypatch, tmp_path):
    """Accept only bound bin/Scripts wrappers; never admit arbitrary RECORD traversal."""
    import base64
    import sysconfig
    from importlib import metadata

    from twelve_six.third_party_reuse import verify_installed_wheel_record

    site = tmp_path / "env" / "lib" / "python3.11" / "site-packages"
    scripts = tmp_path / "env" / "bin"
    site.mkdir(parents=True)
    scripts.mkdir(parents=True)
    module = site / "example.py"
    wrapper = scripts / "wheel-tool"
    module.write_bytes(b"module")
    wrapper.write_bytes(b"script")

    def row(path, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        return f"{path},sha256={digest},{len(data)}\n"

    module_row = row("example.py", b"module")
    self_row = "example-1.2.3.dist-info/RECORD,,\n"

    class Installed:
        version = "1.2.3"

        def __init__(self):
            self.metadata = {"Name": "example"}
            self.record = module_row + row("../../../bin/wheel-tool", b"script") + self_row

        def read_text(self, name):
            assert name == "RECORD"
            return self.record

        def locate_file(self, name):
            return site / name

    installed = Installed()
    monkeypatch.setattr(metadata, "distribution", lambda name: installed)
    original_get_path = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig, "get_path",
        lambda key: str(scripts) if key == "scripts" else original_get_path(key),
    )

    def check():
        return verify_installed_wheel_record(
            "example", expected_version="1.2.3",
            independently_pinned_record_sha256=hashlib.sha256(
                installed.record.encode("utf-8")
            ).hexdigest(),
        )

    assert check()["verified_files"] == 2
    assert check()["verified_files"] == 2  # repeatable restoration/recheck

    wrapper.write_bytes(b"tamper")
    with pytest.raises(ValueError, match="contents drift"):
        check()
    wrapper.write_bytes(b"script")
    assert check()["verified_files"] == 2

    valid_record = installed.record
    for hostile in (
        "../../../../bin/wheel-tool",
        "../../../bin/nested/wheel-tool",
        "../../../bin/../bin/wheel-tool",
        "../../../bin/.",
        "../../../bin/..",
    ):
        installed.record = module_row + row(hostile, b"script") + self_row
        with pytest.raises(ValueError, match="unsafe"):
            check()
    installed.record = valid_record

    monkeypatch.setattr(
        sysconfig, "get_path",
        lambda key: str(tmp_path / "other" / "bin") if key == "scripts" else
        original_get_path(key),
    )
    with pytest.raises(ValueError, match="unsafe"):
        check()


def test_untrusted_review_and_ancestry_claims_are_bounded_before_digest(monkeypatch):
    """Oversized attacker input must not be SHA-256 processed before rejection."""
    from twelve_six.third_party_reuse import prepare_trusted_base_checkpoint

    oversized = b"{" + b"x" * 1048576
    genuine_hash = hashlib.sha256
    attempted = []

    def bounded_digest(value=b""):
        attempted.append(len(value))
        if len(value) > 1048576:
            raise AssertionError("unbounded attacker-controlled digest attempted")
        return genuine_hash(value)

    monkeypatch.setattr("twelve_six.third_party_reuse.hashlib.sha256", bounded_digest)
    with pytest.raises(ValueError, match="bounded reviewed catalog"):
        require_reviewed_code(
            oversized, "pytorch", "0" * 64,
            independently_pinned_catalog_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="bounded Base lineage"):
        prepare_trusted_base_checkpoint(
            "not-a-real-checkpoint", lineage_bytes=oversized,
            trusted_genesis_bytes=b"{}", expected_genesis_sha256="0" * 64,
            expected_lineage_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="bounded trusted genesis"):
        prepare_trusted_base_checkpoint(
            "not-a-real-checkpoint", lineage_bytes=b"{}",
            trusted_genesis_bytes=oversized, expected_genesis_sha256="0" * 64,
            expected_lineage_sha256="0" * 64,
        )
    assert attempted == []

def test_untrusted_archive_hashing_has_bounded_stat_and_stream(monkeypatch, tmp_path):
    """Oversized stat and a stream that grows beyond its claimed size both fail closed."""
    import os

    from twelve_six import third_party_reuse as reuse

    path = tmp_path / "untrusted.whl"
    with monkeypatch.context() as patch:
        patch.setattr(reuse, "MAX_REVIEW_ARTIFACT_BYTES", 8)
        path.write_bytes(b"ninebytes")
        with pytest.raises(ValueError, match="size limit"):
            reuse._hash_regular_archive(path)

        path.write_bytes(b"ok")
        original_read = os.read
        emitted = False

        def oversized_stream(fd, size):
            nonlocal emitted
            data = original_read(fd, size)
            if data and not emitted:
                emitted = True
                return b"x" * 9
            return data

        patch.setattr(os, "read", oversized_stream)
        with pytest.raises(ValueError, match="size limit"):
            reuse._hash_regular_archive(path)
        assert emitted



def test_review_archive_denials_do_not_read_untrusted_files(monkeypatch, tmp_path):
    """No large archive I/O is permitted until the review pin and status pass."""
    from twelve_six import third_party_reuse as reuse

    reads = []

    def forbidden_hash(path):
        reads.append(path)
        raise AssertionError("untrusted artifact was read before admission")

    monkeypatch.setattr(reuse, "_hash_regular_archive", forbidden_hash)
    kwargs = {
        "source_archive": tmp_path / "untrusted-source.tar",
        "license_file": tmp_path / "untrusted-license.txt",
    }
    with pytest.raises(ValueError, match="independently approved catalog"):
        reuse.verify_reviewed_code_archive(
            RAW, "pytorch", independently_pinned_catalog_sha256="0" * 64, **kwargs
        )
    with pytest.raises(ValueError, match="unqualified code asset"):
        reuse.verify_reviewed_code_archive(
            RAW, "pytorch",
            independently_pinned_catalog_sha256=hashlib.sha256(RAW).hexdigest(),
            **kwargs,
        )
    with pytest.raises(ValueError, match="unknown external asset"):
        reuse.verify_reviewed_code_archive(
            RAW, "missing-asset",
            independently_pinned_catalog_sha256=hashlib.sha256(RAW).hexdigest(),
            **kwargs,
        )
    assert reads == []
