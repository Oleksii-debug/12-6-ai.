import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from twelve_six.checkpoint import (
    CheckpointIdentity,
    CheckpointIntegrityError,
    export_hf_directory,
    hf_export,
    save_checkpoint,
    verify_checkpoint,
    verify_hf_directory,
)


class Model:
    def __init__(self, value: float):
        self.value = np.array([value], dtype=np.float64)

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state, strict=True):
        self.value = state["value"]


def identity(fill: str = "a") -> CheckpointIdentity:
    if len(fill) != 1:
        raise ValueError("identity fixture fill must be one character")
    git_fill = fill.lower()
    git_sha = (
        git_fill * 40
        if git_fill in "0123456789abcdef"
        else f"{ord(fill):040x}"
    )
    return CheckpointIdentity(
        git_sha=git_sha,
        model_spec={"model_type": "twelve_six_export_transactional"},
        parameter_count=1,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"steps": 1},
        seed=7,
        precision="float64",
        step=1,
        tokens_seen=1,
        optimizer={"name": "none"},
        scheduler=None,
    )


def test_identity_fixture_nonhex_label_has_canonical_git_sha():
    observed = identity("g").git_sha

    assert observed == f"{ord('g'):040x}"
    assert len(observed) == 40
    assert observed == observed.lower()
    assert all(char in "0123456789abcdef" for char in observed)


def snapshot_tree(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_existing_export_is_immutable_even_with_overwrite_true(tmp_path: Path):
    first_checkpoint = tmp_path / "checkpoint-a"
    second_checkpoint = tmp_path / "checkpoint-b"
    output = tmp_path / "hf"
    save_checkpoint(first_checkpoint, model=Model(1.0), identity=identity("a"))
    save_checkpoint(second_checkpoint, model=Model(2.0), identity=identity("b"))
    export_hf_directory(
        first_checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    before = snapshot_tree(output)

    with pytest.raises(FileExistsError, match="destructive replacement"):
        export_hf_directory(
            second_checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            overwrite=True,
        )

    assert snapshot_tree(output) == before
    verify_hf_directory(output)


def test_export_consumes_verified_checkpoint_snapshot_without_reopening_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(3.0), identity=identity("c"))
    original_weights = (checkpoint / "weights.safetensors").read_bytes()
    real_prepare = hf_export.prepare_checkpoint_load

    def prepare_then_tamper(path: Path):
        verified = real_prepare(path)
        (checkpoint / "weights.safetensors").write_bytes(b"tampered-after-snapshot")
        return verified

    monkeypatch.setattr(hf_export, "prepare_checkpoint_load", prepare_then_tamper)
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    assert (output / "model.safetensors").read_bytes() == original_weights
    verify_hf_directory(output)


def test_parity_hook_reads_verified_reference_after_source_path_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    manifest = save_checkpoint(checkpoint, model=Model(3.5), identity=identity("c"))
    original_weights = (checkpoint / "weights.safetensors").read_bytes()
    real_prepare = hf_export.prepare_checkpoint_load
    first_call = True

    def prepare_then_tamper(path: Path):
        nonlocal first_call
        verified = real_prepare(path)
        if first_call:
            first_call = False
            (checkpoint / "weights.safetensors").write_bytes(b"source-path-now-corrupt")
        return verified

    def parity_hook(reference: Path, candidate: Path):
        assert verify_checkpoint(reference)["checkpoint_id"] == manifest["checkpoint_id"]
        assert (reference / "weights.safetensors").read_bytes() == original_weights
        assert (candidate / "model.safetensors").read_bytes() == original_weights
        return {"status": "PASS", "evidence_ref": "verified-reference-test"}

    monkeypatch.setattr(hf_export, "prepare_checkpoint_load", prepare_then_tamper)
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
        parity_hook=parity_hook,
    )

    verify_hf_directory(output)
    assert not list(tmp_path.glob(".hf.reference-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))


def test_hook_failure_leaves_no_published_or_temporary_export(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.0), identity=identity("d"))

    def broken_hook(_reference: Path, _candidate: Path):
        raise RuntimeError("parity failed")

    with pytest.raises(RuntimeError, match="parity failed"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=broken_hook,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_retained_hook_candidate_cannot_mutate_final_publish_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.25), identity=identity("d"))
    expected_weights = (checkpoint / "weights.safetensors").read_bytes()
    retained: dict[str, Path] = {}

    def parity_hook(reference: Path, candidate: Path):
        retained["reference"] = reference
        retained["candidate"] = candidate
        return {"status": "PASS", "evidence_ref": "retained-path-regression"}

    real_publish = hf_export._publish_directory_noreplace

    def publish_after_retained_path_mutation(staging: Path, destination: Path):
        assert not os.path.lexists(retained["reference"])
        assert not os.path.lexists(retained["candidate"])
        assert staging != retained["candidate"]

        retained["candidate"].mkdir()
        (retained["candidate"] / "model.safetensors").write_bytes(b"late-hook-mutation")
        real_publish(staging, destination)

    monkeypatch.setattr(
        hf_export,
        "_publish_directory_noreplace",
        publish_after_retained_path_mutation,
    )
    try:
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=parity_hook,
        )
        assert (output / "model.safetensors").read_bytes() == expected_weights
        verify_hf_directory(output)
    finally:
        late_candidate = retained.get("candidate")
        if late_candidate is not None and late_candidate.exists():
            hf_export.shutil.rmtree(late_candidate)


def test_hook_visible_cleanup_failure_aborts_before_final_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.375), identity=identity("d"))
    real_rmtree = hf_export.shutil.rmtree

    def parity_hook(_reference: Path, _candidate: Path):
        return {"status": "PASS", "evidence_ref": "cleanup-failure-regression"}

    def fail_candidate_cleanup(path, *args, **kwargs):
        if ".hook-candidate-" in Path(path).name:
            raise OSError("simulated hook candidate cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hf_export.shutil, "rmtree", fail_candidate_cleanup)
    try:
        with pytest.raises(CheckpointIntegrityError, match="temporary cleanup failed"):
            export_hf_directory(
                checkpoint,
                output,
                hf_config={"model_type": "twelve_six_export_transactional"},
                parity_hook=parity_hook,
            )

        assert not output.exists()
        assert not list(tmp_path.glob(".hf.staging-*"))
        assert list(tmp_path.glob(".hf.hook-candidate-*"))
        assert not list(tmp_path.glob(".hf.reference-*"))
    finally:
        for path in tmp_path.glob(".hf.hook-candidate-*"):
            real_rmtree(path)


@pytest.mark.skipif(
    os.name != "nt" and not sys.platform.startswith("linux"),
    reason="atomic no-replace publish is only implemented on Windows/Linux",
)
def test_concurrent_destination_creation_is_not_replaced(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.5), identity=identity("d"))

    def racing_hook(_reference: Path, _candidate: Path):
        output.mkdir()
        (output / "owner-evidence.txt").write_text("preserve me", encoding="utf-8")
        return {"status": "PASS", "evidence_ref": "racing-destination-test"}

    with pytest.raises(FileExistsError, match="appeared during publish"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=racing_hook,
        )

    assert (output / "owner-evidence.txt").read_text(encoding="utf-8") == "preserve me"
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_export_verifier_rejects_payload_tamper(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(5.0), identity=identity("e"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    weights = output / "model.safetensors"
    weights.write_bytes(weights.read_bytes() + b"x")
    with pytest.raises(CheckpointIntegrityError, match="canonical"):
        verify_hf_directory(output)


def test_export_verifier_rejects_attestation_tamper(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(6.0), identity=identity("f"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    attestation_path = output / "12-6-export.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["compatibility"]["transformers_architecture"] = "CLAIMED"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    with pytest.raises(CheckpointIntegrityError, match="attestation checksum"):
        verify_hf_directory(output)


@pytest.mark.parametrize(
    "case",
    (
        "single_space",
        "tab_separator",
        "leading_space",
        "trailing_space",
        "extra_newline",
        "uppercase_digest",
        "nonhex_digest",
        "short_digest",
    ),
)
def test_export_verifier_rejects_noncanonical_checksum_bytes(
    tmp_path: Path,
    case: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(6.25), identity=identity("7"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    attestation = (output / hf_export.EXPORT_ATTESTATION_NAME).read_bytes()
    digest = hf_export.sha256_bytes(attestation)
    if case == "single_space":
        checksum = f"{digest} {hf_export.EXPORT_ATTESTATION_NAME}\n"
    elif case == "tab_separator":
        checksum = f"{digest}\t{hf_export.EXPORT_ATTESTATION_NAME}\n"
    elif case == "leading_space":
        checksum = f" {digest}  {hf_export.EXPORT_ATTESTATION_NAME}\n"
    elif case == "trailing_space":
        checksum = f"{digest}  {hf_export.EXPORT_ATTESTATION_NAME} \n"
    elif case == "extra_newline":
        checksum = f"{digest}  {hf_export.EXPORT_ATTESTATION_NAME}\n\n"
    elif case == "uppercase_digest":
        checksum = f"{'A' * 64}  {hf_export.EXPORT_ATTESTATION_NAME}\n"
    elif case == "nonhex_digest":
        checksum = f"{'g' * 64}  {hf_export.EXPORT_ATTESTATION_NAME}\n"
    elif case == "short_digest":
        checksum = f"{'0' * 63}  {hf_export.EXPORT_ATTESTATION_NAME}\n"
    else:
        raise AssertionError(f"unhandled checksum case: {case}")

    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        checksum,
        encoding="ascii",
    )
    with pytest.raises(
        CheckpointIntegrityError,
        match="invalid 12-6-export.sha256 format",
    ):
        verify_hf_directory(output)


def test_export_verifier_preserves_canonical_wrong_checksum_diagnostic(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(6.5), identity=identity("8"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        f"{'0' * 64}  {hf_export.EXPORT_ATTESTATION_NAME}\n",
        encoding="ascii",
    )
    with pytest.raises(
        CheckpointIntegrityError,
        match="attestation checksum mismatch",
    ):
        verify_hf_directory(output)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support unavailable")
def test_export_verifier_rejects_symlink_payload(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(7.0), identity=identity("0"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    weights = output / "model.safetensors"
    target = tmp_path / "weights-copy.safetensors"
    target.write_bytes(weights.read_bytes())
    weights.unlink()
    try:
        weights.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable in this environment")

    with pytest.raises(CheckpointIntegrityError, match="non-symlink"):
        verify_hf_directory(output)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_export_rejects_nonfinite_hf_config_without_publication(
    tmp_path: Path,
    value: float,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(8.0), identity=identity("1"))

    with pytest.raises(CheckpointIntegrityError, match="strict finite JSON"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={
                "model_type": "twelve_six_export_transactional",
                "unsafe": value,
            },
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_export_rejects_nonfinite_parity_hook_evidence_and_cleans_temp_paths(
    tmp_path: Path,
    value: float,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(8.5), identity=identity("2"))

    def parity_hook(_reference: Path, _candidate: Path):
        return {"status": "PASS", "metric": value}

    with pytest.raises(CheckpointIntegrityError, match="strict finite JSON"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=parity_hook,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_verifier_rejects_nonfinite_numeric_overflow_in_config(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(9.0), identity=identity("3"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    (output / hf_export.EXPORTED_CONFIG_NAME).write_text(
        '{"model_type":"twelve_six_export_transactional","unsafe":1e400}\n',
        encoding="utf-8",
    )
    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_hf_directory(output)


def test_verifier_rejects_duplicate_attestation_key_even_with_resealed_checksum(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(9.5), identity=identity("4"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    attestation_path = output / hf_export.EXPORT_ATTESTATION_NAME
    raw = attestation_path.read_text(encoding="utf-8").rstrip()
    assert raw.endswith("}")
    tampered = (
        raw[:-1]
        + ',"schema":"12-6.hf-style-export.v2"}\n'
    ).encode("utf-8")
    attestation_path.write_bytes(tampered)
    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        f"{hf_export.sha256_bytes(tampered)}  {hf_export.EXPORT_ATTESTATION_NAME}\n",
        encoding="ascii",
    )

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_hf_directory(output)


def test_verifier_rejects_unknown_attestation_claim_after_reseal(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(10.0), identity=identity("5"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    attestation_path = output / hf_export.EXPORT_ATTESTATION_NAME
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["training_authorized"] = True
    tampered = (
        json.dumps(
            attestation,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    attestation_path.write_bytes(tampered)
    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        f"{hf_export.sha256_bytes(tampered)}  {hf_export.EXPORT_ATTESTATION_NAME}\n",
        encoding="ascii",
    )

    with pytest.raises(CheckpointIntegrityError, match="fields mismatch"):
        verify_hf_directory(output)


def test_verifier_rejects_unknown_parity_request_claim(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(10.5), identity=identity("6"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    parity_path = output / hf_export.PARITY_REQUEST_NAME
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["training_authorized"] = True
    parity_path.write_text(
        json.dumps(
            parity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CheckpointIntegrityError, match="fields mismatch"):
        verify_hf_directory(output)


def test_verifier_rejects_duplicate_config_key_before_hash_checks(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(11.0), identity=identity("7"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    (output / hf_export.EXPORTED_CONFIG_NAME).write_text(
        '{"model_type":"twelve_six_export_transactional","model_type":"shadow"}\n',
        encoding="utf-8",
    )

    with pytest.raises(CheckpointIntegrityError, match="strict UTF-8 JSON"):
        verify_hf_directory(output)


@pytest.mark.parametrize("root_name", ["reference", "candidate"])
@pytest.mark.parametrize("replace_root", [False, True])
def test_hook_cannot_make_exporter_delete_substituted_private_root(
    tmp_path: Path,
    root_name: str,
    replace_root: bool,
):
    """A hook-controlled pathname is not proof of private-directory ownership."""

    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(12.0), identity=identity("8"))
    retained: dict[str, Path] = {}

    def swapping_hook(reference: Path, candidate: Path):
        selected = {"reference": reference, "candidate": candidate}[root_name]
        original = tmp_path / f"retained-original-{root_name}"
        selected.rename(original)
        retained["original"] = original
        if replace_root:
            selected.mkdir()
            (selected / "unrelated-evidence.txt").write_text(
                "preserve unrelated user evidence", encoding="utf-8"
            )
            retained["substituted"] = selected
        return {"status": "PASS", "evidence_ref": "root-identity-regression"}

    with pytest.raises(CheckpointIntegrityError, match="temporary cleanup failed"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=swapping_hook,
        )

    assert retained["original"].is_dir()
    if replace_root:
        assert (
            retained["substituted"] / "unrelated-evidence.txt"
        ).read_text(encoding="utf-8") == "preserve unrelated user evidence"
    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))



def test_private_root_identity_requires_an_available_inode():
    class UnknownInodeRoot:
        def lstat(self):
            return SimpleNamespace(
                st_mode=stat.S_IFDIR | 0o700, st_dev=1, st_ino=0
            )

        def __str__(self):
            return "unknown-inode-test-root"

    with pytest.raises(CheckpointIntegrityError, match="inode identity unavailable"):
        hf_export._temporary_directory_identity(UnknownInodeRoot())


def test_missing_private_root_identity_preserves_unknown_root_and_cleans_known_root(
    tmp_path: Path,
):
    unknown = tmp_path / "unknown"
    known = tmp_path / "known"
    unknown.mkdir()
    known.mkdir()
    (unknown / "user-evidence.txt").write_text("preserve", encoding="utf-8")
    (known / "private-artifact.txt").write_text("cleanup", encoding="utf-8")
    known_identity = hf_export._temporary_directory_identity(known)

    with pytest.raises(CheckpointIntegrityError, match="temporary cleanup failed"):
        hf_export._cleanup_temp_paths_strict(
            (
                (unknown, "unowned root", None),
                (known, "owned root", known_identity),
            )
        )

    assert (unknown / "user-evidence.txt").read_text(encoding="utf-8") == "preserve"
    assert not known.exists()


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_hf_parser_rejects_overdeep_json_as_integrity_error(nesting: str):
    if nesting == "arrays":
        raw = '{"payload":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        raw = '{"payload":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    with pytest.raises(CheckpointIntegrityError, match="not valid strict UTF-8 JSON"):
        hf_export._json_object(raw.encode("utf-8"), artifact="overdeep-input")


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
@pytest.mark.parametrize(
    "artifact",
    (
        hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
        hf_export.EXPORTED_CONFIG_NAME,
        hf_export.PARITY_REQUEST_NAME,
        hf_export.EXPORT_ATTESTATION_NAME,
    ),
)
def test_hf_verifier_rejects_overdeep_artifact_without_trusting_hash(
    tmp_path: Path,
    nesting: str,
    artifact: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(13.0), identity=identity("9"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    if nesting == "arrays":
        raw = '{"payload":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    else:
        raw = '{"payload":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"
    target = output / artifact
    target.write_text(raw, encoding="utf-8")
    if artifact == hf_export.EXPORT_ATTESTATION_NAME:
        (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
            f"{hf_export.sha256_bytes(target.read_bytes())}  {artifact}\n",
            encoding="ascii",
        )

    original = target.read_bytes()
    with pytest.raises(CheckpointIntegrityError, match="not valid strict UTF-8 JSON"):
        verify_hf_directory(output)
    assert target.read_bytes() == original


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_hf_encoder_rejects_overdeep_json_as_integrity_error(nesting: str):
    payload: object = 0
    for _ in range(10000):
        payload = [payload] if nesting == "arrays" else {"k": payload}
    with pytest.raises(CheckpointIntegrityError, match="not strict finite JSON"):
        hf_export._strict_json_bytes({"payload": payload}, artifact="overdeep-output")


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_hf_export_rejects_overdeep_config_without_publication(
    tmp_path: Path, nesting: str
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(14.0), identity=identity("a"))
    payload: object = 0
    for _ in range(10000):
        payload = [payload] if nesting == "arrays" else {"k": payload}
    config = {"model_type": "twelve_six_export_transactional", "payload": payload}

    with pytest.raises(CheckpointIntegrityError, match="not strict finite JSON"):
        export_hf_directory(checkpoint, output, hf_config=config)
    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))


def test_hf_unexpected_product_recursion_is_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(15.0), identity=identity("b"))

    def unexpected(_path: Path):
        raise RecursionError("unexpected checkpoint Product recursion")

    monkeypatch.setattr(hf_export, "prepare_checkpoint_load", unexpected)
    with pytest.raises(RecursionError, match="unexpected checkpoint Product recursion"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
        )
    assert not output.exists()


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_hf_hook_overdeep_result_cleans_private_roots(
    tmp_path: Path, nesting: str
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(16.0), identity=identity("c"))
    payload: object = 0
    for _ in range(10000):
        payload = [payload] if nesting == "arrays" else {"k": payload}

    def overdeep_hook(_reference: Path, _candidate: Path):
        return {"nested": payload}

    with pytest.raises(CheckpointIntegrityError, match="not strict finite JSON"):
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
            parity_hook=overdeep_hook,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".hf.reference-*"))
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))
    assert not list(tmp_path.glob(".hf.staging-*"))

@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_hook_candidate_materialization_interrupt_cleans_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    primary = interrupt_type("candidate materialization interrupted")
    real_write_bytes = Path.write_bytes

    def interrupt_candidate_write(path: Path, data: bytes):
        if ".hook-candidate-" in path.parent.name:
            raise primary
        return real_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", interrupt_candidate_write)

    with pytest.raises(interrupt_type) as caught:
        hf_export._materialize_hook_candidate(
            parent=tmp_path,
            name="hf",
            weights=b"weights",
            config=b"{}",
            source_manifest=b"{}",
        )

    assert caught.value is primary
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))


def test_parity_failure_remains_primary_when_candidate_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.4375), identity=identity("d"))
    primary = RuntimeError("parity primary failure")
    real_rmtree = hf_export.shutil.rmtree

    def broken_hook(_reference: Path, _candidate: Path):
        raise primary

    def fail_candidate_cleanup(path, *args, **kwargs):
        if ".hook-candidate-" in Path(path).name:
            raise OSError("simulated candidate cleanup double-fault")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hf_export.shutil, "rmtree", fail_candidate_cleanup)
    try:
        with pytest.raises(RuntimeError) as caught:
            export_hf_directory(
                checkpoint,
                output,
                hf_config={"model_type": "twelve_six_export_transactional"},
                parity_hook=broken_hook,
            )

        assert caught.value is primary
        assert any(
            "HF parity hook candidate cleanup also failed" in note
            for note in getattr(primary, "__notes__", ())
        )
        assert not output.exists()
        assert list(tmp_path.glob(".hf.hook-candidate-*"))
        assert not list(tmp_path.glob(".hf.reference-*"))
    finally:
        for path in tmp_path.glob(".hf.hook-candidate-*"):
            real_rmtree(path)


def test_publish_failure_remains_primary_when_staging_cleanup_also_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(4.46875), identity=identity("d"))
    primary = RuntimeError("publish primary failure")
    real_rmtree = hf_export.shutil.rmtree

    def fail_publish(_staging: Path, _destination: Path):
        raise primary

    def fail_staging_cleanup(path, *args, **kwargs):
        if ".staging-" in Path(path).name:
            raise OSError("simulated staging cleanup double-fault")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(hf_export, "_publish_directory_noreplace", fail_publish)
    monkeypatch.setattr(hf_export.shutil, "rmtree", fail_staging_cleanup)
    try:
        with pytest.raises(RuntimeError) as caught:
            export_hf_directory(
                checkpoint,
                output,
                hf_config={"model_type": "twelve_six_export_transactional"},
            )

        assert caught.value is primary
        assert any(
            "HF export staging cleanup also failed" in note
            for note in getattr(primary, "__notes__", ())
        )
        assert not output.exists()
        assert list(tmp_path.glob(".hf.staging-*"))
    finally:
        for path in tmp_path.glob(".hf.staging-*"):
            real_rmtree(path)


@pytest.mark.parametrize("interrupt_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
def test_cleanup_only_interrupt_retains_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    root = tmp_path / ".hf.hook-candidate-interrupt"
    root.mkdir()
    identity_value = hf_export._temporary_directory_identity(root)
    primary = interrupt_type("cleanup interrupted")

    def interrupt_cleanup(*_args, **_kwargs):
        raise primary

    monkeypatch.setattr(hf_export, "_remove_temp_path_strict", interrupt_cleanup)

    with pytest.raises(interrupt_type) as caught:
        hf_export._cleanup_temp_paths_strict(
            ((root, "HF parity hook candidate", identity_value),)
        )

    assert caught.value is primary


class _HostileNotePrimary(RuntimeError):
    def add_note(self, _note: str) -> None:
        raise AssertionError("virtual add_note dispatch is forbidden")


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_verified_reference_materialization_interrupt_cleans_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    primary = interrupt_type("verified reference materialization interrupted")
    real_write_bytes = Path.write_bytes

    def interrupt_reference_write(path: Path, data: bytes):
        if ".reference-" in path.parent.name:
            raise primary
        return real_write_bytes(path, data)

    monkeypatch.setattr(Path, "write_bytes", interrupt_reference_write)

    with pytest.raises(interrupt_type) as caught:
        hf_export._materialize_verified_reference(
            SimpleNamespace(_manifest_bytes=b"{}", _artifacts={}),
            tmp_path,
            "hf",
        )

    assert caught.value is primary
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_verified_reference_cleanup_double_fault_preserves_hostile_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    primary = _HostileNotePrimary("verified reference primary")
    real_write_bytes = Path.write_bytes
    real_rmtree = hf_export.shutil.rmtree

    def fail_reference_write(path: Path, data: bytes):
        if ".reference-" in path.parent.name:
            raise primary
        return real_write_bytes(path, data)

    def fail_reference_cleanup(path, *args, **kwargs):
        if ".reference-" in Path(path).name:
            raise OSError("verified reference cleanup double-fault")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_bytes", fail_reference_write)
    monkeypatch.setattr(hf_export.shutil, "rmtree", fail_reference_cleanup)
    try:
        with pytest.raises(_HostileNotePrimary) as caught:
            hf_export._materialize_verified_reference(
                SimpleNamespace(_manifest_bytes=b"{}", _artifacts={}),
                tmp_path,
                "hf",
            )

        assert caught.value is primary
        assert any(
            "verified checkpoint reference cleanup also failed" in note
            for note in getattr(primary, "__notes__", ())
        )
        assert list(tmp_path.glob(".hf.reference-*"))
    finally:
        monkeypatch.setattr(hf_export.shutil, "rmtree", real_rmtree)
        for path_item in tmp_path.glob(".hf.reference-*"):
            real_rmtree(path_item)


def test_malformed_primary_notes_do_not_mask_cleanup_double_fault(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / ".hf.staging-malformed-notes"
    root.mkdir()
    identity_value = hf_export._temporary_directory_identity(root)
    primary = RuntimeError("publish primary with malformed notes")
    primary.__notes__ = "malformed-note-storage"

    def fail_cleanup(*_args, **_kwargs):
        raise OSError("staging cleanup double-fault")

    monkeypatch.setattr(hf_export, "_remove_temp_path_strict", fail_cleanup)

    hf_export._cleanup_temp_paths_strict(
        ((root, "HF export staging", identity_value),),
        primary_exc=primary,
    )

    assert primary.__notes__ == "malformed-note-storage"


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_cleanup_later_interrupt_outranks_earlier_ordinary_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    first = tmp_path / ".hf.cleanup-first"
    second = tmp_path / ".hf.cleanup-second"
    first.mkdir()
    second.mkdir()
    identities = {
        first: hf_export._temporary_directory_identity(first),
        second: hf_export._temporary_directory_identity(second),
    }
    primary = interrupt_type("second cleanup interrupted")
    seen: list[Path] = []

    def fail_cleanup(
        path: Path,
        *,
        label: str,
        expected_identity: tuple[int, int],
    ):
        assert expected_identity == identities[path]
        seen.append(path)
        if path == first:
            raise OSError(f"{label} ordinary failure")
        if path == second:
            raise primary
        raise AssertionError(f"unexpected cleanup path: {path}")

    monkeypatch.setattr(hf_export, "_remove_temp_path_strict", fail_cleanup)

    with pytest.raises(interrupt_type) as caught:
        hf_export._cleanup_temp_paths_strict(
            (
                (first, "first root", identities[first]),
                (second, "second root", identities[second]),
            )
        )

    assert caught.value is primary
    assert seen == [first, second]
    assert any(
        "first root cleanup also failed while second root raised" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_cleanup_multiple_ordinary_failures_remain_integrity_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    first = tmp_path / ".hf.cleanup-first-ordinary"
    second = tmp_path / ".hf.cleanup-second-ordinary"
    first.mkdir()
    second.mkdir()
    identities = {
        first: hf_export._temporary_directory_identity(first),
        second: hf_export._temporary_directory_identity(second),
    }
    seen: list[Path] = []

    def fail_cleanup(
        path: Path,
        *,
        label: str,
        expected_identity: tuple[int, int],
    ):
        assert expected_identity == identities[path]
        seen.append(path)
        raise OSError(f"{label} ordinary failure")

    monkeypatch.setattr(hf_export, "_remove_temp_path_strict", fail_cleanup)

    with pytest.raises(
        CheckpointIntegrityError,
        match="temporary cleanup failed for: first root, second root",
    ) as caught:
        hf_export._cleanup_temp_paths_strict(
            (
                (first, "first root", identities[first]),
                (second, "second root", identities[second]),
            )
        )

    assert seen == [first, second]
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "first root ordinary failure"


def _close_then_raise_oserror(
    real_close,
    real_fstat,
    closed: list[int],
):
    def injected_close(fd: int):
        real_close(fd)
        try:
            real_fstat(fd)
        except OSError:
            closed.append(fd)
        else:
            raise AssertionError("descriptor remained open after real close")
        raise OSError("simulated HF artifact descriptor close failure")

    return injected_close


def test_hf_reader_close_failure_after_success_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot close HF-style export artifact",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert closed
    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == "simulated HF artifact descriptor close failure"


def test_hf_reader_primary_integrity_failure_survives_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []

    def report_nonregular(fd: int):
        opened = real_fstat(fd)
        return SimpleNamespace(
            st_mode=stat.S_IFDIR | 0o700,
            st_dev=opened.st_dev,
            st_ino=opened.st_ino,
        )

    monkeypatch.setattr(hf_export.os, "fstat", report_nonregular)
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export artifact changed type",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert closed
    assert any(
        "HF-style export artifact close also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_hf_reader_interrupt_survives_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    primary = interrupt_type("HF artifact read interrupted")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []

    def interrupt_fstat(_fd: int):
        raise primary

    monkeypatch.setattr(hf_export.os, "fstat", interrupt_fstat)
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    with pytest.raises(interrupt_type) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value is primary
    assert closed
    assert any(
        "HF-style export artifact close also failed" in note
        for note in getattr(primary, "__notes__", ())
    )


def test_hf_reader_ambient_exception_does_not_hide_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    try:
        raise RuntimeError("unrelated ambient caller failure")
    except RuntimeError:
        with pytest.raises(
            CheckpointIntegrityError,
            match="cannot close HF-style export artifact",
        ) as caught:
            hf_export._read_regular_bytes(root, name)

    assert closed
    assert isinstance(caught.value.__cause__, OSError)

class _FailingReadHandle:
    def __init__(self, failure: BaseException):
        self._failure = failure

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _traceback):
        return False

    def read(self, *_args):
        raise self._failure


def test_hf_reader_fstat_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    failure = OSError("simulated HF artifact fstat failure")

    def fail_fstat(_fd: int):
        raise failure

    monkeypatch.setattr(hf_export.os, "fstat", fail_fstat)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect HF-style export artifact",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value.__cause__ is failure


def test_hf_reader_read_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    failure = OSError("simulated HF artifact read failure")

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(failure),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot read HF-style export artifact",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value.__cause__ is failure


def test_hf_reader_read_oserror_survives_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    failure = OSError("simulated HF artifact read failure")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(failure),
    )
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot read HF-style export artifact",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value.__cause__ is failure
    assert closed
    assert any(
        "HF-style export artifact close also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_hf_reader_read_interrupt_retains_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    primary = interrupt_type("simulated HF artifact read interrupt")

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(primary),
    )

    with pytest.raises(interrupt_type) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value is primary

def test_hf_reader_artifact_lstat_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    name = "artifact.bin"
    (root / name).write_bytes(b"exact artifact bytes")
    failure = OSError("simulated HF artifact lstat failure")
    real_lstat = Path.lstat

    def fail_artifact_lstat(path: Path):
        if path == root / name:
            raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_artifact_lstat)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect HF-style export artifact",
    ) as caught:
        hf_export._read_regular_bytes(root, name)

    assert caught.value.__cause__ is failure


def test_hf_snapshot_root_lstat_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    failure = OSError("simulated HF root lstat failure")
    real_lstat = Path.lstat

    def fail_root_lstat(path: Path):
        if path == root:
            raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_root_lstat)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect HF-style export directory",
    ) as caught:
        hf_export._read_export_snapshot(root)

    assert caught.value.__cause__ is failure


def test_hf_snapshot_inventory_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    failure = OSError("simulated HF root inventory failure")
    real_iterdir = Path.iterdir

    def fail_root_iterdir(path: Path):
        if path == root:
            raise failure
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_root_iterdir)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot enumerate HF-style export directory",
    ) as caught:
        hf_export._read_export_snapshot(root)

    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_hf_snapshot_inventory_interrupt_retains_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    root = tmp_path / "hf"
    root.mkdir()
    primary = interrupt_type("simulated HF inventory interruption")
    real_iterdir = Path.iterdir

    def interrupt_root_iterdir(path: Path):
        if path == root:
            raise primary
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", interrupt_root_iterdir)

    with pytest.raises(interrupt_type) as caught:
        hf_export._read_export_snapshot(root)

    assert caught.value is primary

def test_verified_reference_identity_failure_cleans_empty_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure = OSError("simulated verified-reference identity failure")
    real_lstat = Path.lstat

    def fail_reference_identity(path: Path):
        if ".reference-" in path.name:
            raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_reference_identity)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect private temporary root",
    ) as caught:
        hf_export._materialize_verified_reference(
            SimpleNamespace(_manifest_bytes=b"{}", _artifacts={}),
            tmp_path,
            "hf",
        )

    assert caught.value.__cause__ is failure
    assert not list(tmp_path.glob(".hf.reference-*"))


def test_hook_candidate_identity_failure_cleans_empty_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    failure = OSError("simulated hook-candidate identity failure")
    real_lstat = Path.lstat

    def fail_candidate_identity(path: Path):
        if ".hook-candidate-" in path.name:
            raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_candidate_identity)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect private temporary root",
    ) as caught:
        hf_export._materialize_hook_candidate(
            parent=tmp_path,
            name="hf",
            weights=b"weights",
            config=b"{}",
            source_manifest=b"{}",
        )

    assert caught.value.__cause__ is failure
    assert not list(tmp_path.glob(".hf.hook-candidate-*"))


def test_staging_identity_failure_cleans_empty_private_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(17.0), identity=identity("e"))
    failure = OSError("simulated staging identity failure")
    real_lstat = Path.lstat

    def fail_staging_identity(path: Path):
        if ".staging-" in path.name:
            raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_staging_identity)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect private temporary root",
    ) as caught:
        export_hf_directory(
            checkpoint,
            output,
            hf_config={"model_type": "twelve_six_export_transactional"},
        )

    assert caught.value.__cause__ is failure
    assert not output.exists()
    assert not list(tmp_path.glob(".hf.staging-*"))


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_private_root_identity_interrupt_cleans_and_retains_exact_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    primary = interrupt_type("private-root identity interrupted")
    real_lstat = Path.lstat

    def interrupt_identity(path: Path):
        if ".hf.identity-interrupt-" in path.name:
            raise primary
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", interrupt_identity)

    with pytest.raises(interrupt_type) as caught:
        hf_export._create_private_temp_directory(
            prefix=".hf.identity-interrupt-",
            parent=tmp_path,
            label="identity interrupt root",
        )

    assert caught.value is primary
    assert not list(tmp_path.glob(".hf.identity-interrupt-*"))


def test_private_root_identity_cleanup_double_fault_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    primary = _HostileNotePrimary("private-root identity primary")
    real_identity = hf_export._temporary_directory_identity
    real_rmdir = hf_export.os.rmdir
    created: list[Path] = []

    def fail_identity(path: Path):
        created.append(path)
        raise primary

    def fail_empty_root_cleanup(path):
        raise OSError(f"simulated empty-root rmdir failure: {path}")

    def forbid_recursive_cleanup(*_args, **_kwargs):
        raise AssertionError("recursive cleanup before trusted identity is forbidden")

    monkeypatch.setattr(hf_export, "_temporary_directory_identity", fail_identity)
    monkeypatch.setattr(hf_export.os, "rmdir", fail_empty_root_cleanup)
    monkeypatch.setattr(hf_export.shutil, "rmtree", forbid_recursive_cleanup)
    try:
        with pytest.raises(_HostileNotePrimary) as caught:
            hf_export._create_private_temp_directory(
                prefix=".hf.identity-double-fault-",
                parent=tmp_path,
                label="identity double-fault root",
            )

        assert caught.value is primary
        assert len(created) == 1
        assert created[0].is_dir()
        assert any(
            "identity double-fault root empty-root cleanup also failed" in note
            for note in getattr(primary, "__notes__", ())
        )
    finally:
        monkeypatch.setattr(hf_export, "_temporary_directory_identity", real_identity)
        monkeypatch.setattr(hf_export.os, "rmdir", real_rmdir)
        for path_item in created:
            if path_item.exists():
                real_rmdir(path_item)


def test_direct_materializer_helpers_preserve_path_return_contract(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model=Model(18.0), identity=identity("f"))
    verified = hf_export.prepare_checkpoint_load(checkpoint)

    reference = hf_export._materialize_verified_reference(
        verified,
        tmp_path,
        "hf",
    )
    candidate = hf_export._materialize_hook_candidate(
        parent=tmp_path,
        name="hf",
        weights=b"weights",
        config=b"{}",
        source_manifest=b"{}",
    )

    assert isinstance(reference, Path)
    assert isinstance(candidate, Path)

    reference_identity = hf_export._temporary_directory_identity(reference)
    candidate_identity = hf_export._temporary_directory_identity(candidate)
    hf_export._cleanup_temp_paths_strict(
        (
            (candidate, "HF parity hook candidate", candidate_identity),
            (reference, "verified checkpoint reference", reference_identity),
        )
    )


def test_export_pins_each_private_root_identity_exactly_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(19.0), identity=identity("1"))
    real_identity = hf_export._temporary_directory_identity
    seen: list[str] = []

    def observe_identity(path: Path):
        if ".reference-" in path.name:
            seen.append("reference")
        elif ".hook-candidate-" in path.name:
            seen.append("candidate")
        elif ".staging-" in path.name:
            seen.append("staging")
        return real_identity(path)

    def parity_hook(_reference: Path, _candidate: Path):
        return {"status": "PASS", "evidence_ref": "identity-count-regression"}

    monkeypatch.setattr(
        hf_export,
        "_temporary_directory_identity",
        observe_identity,
    )

    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
        parity_hook=parity_hook,
    )

    assert seen == ["reference", "candidate", "staging"]
    verify_hf_directory(output)




def test_hf_snapshot_rejects_root_replacement_after_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    root = tmp_path / "hf"
    moved = tmp_path / "hf-original"
    save_checkpoint(checkpoint, model=Model(21.0), identity=identity("g"))
    export_hf_directory(
        checkpoint,
        root,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    real_iterdir = Path.iterdir

    def replace_after_inventory(path: Path):
        entries = list(real_iterdir(path))
        if path == root:
            path.rename(moved)
            path.mkdir()
        return iter(entries)

    monkeypatch.setattr(Path, "iterdir", replace_after_inventory)

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export directory identity changed while reading",
    ):
        hf_export._read_export_snapshot(root)


def test_hf_snapshot_rejects_root_replacement_during_artifact_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    root = tmp_path / "hf"
    moved = tmp_path / "hf-original"
    save_checkpoint(checkpoint, model=Model(22.0), identity=identity("h"))
    export_hf_directory(
        checkpoint,
        root,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    real_read = hf_export._read_regular_bytes
    replaced = False

    def replace_after_first_read(
        path: Path,
        name: str,
        *,
        max_bytes: int | None = None,
    ):
        nonlocal replaced
        data = real_read(path, name, max_bytes=max_bytes)
        if path == root and not replaced:
            path.rename(moved)
            path.mkdir()
            replaced = True
        return data

    monkeypatch.setattr(hf_export, "_read_regular_bytes", replace_after_first_read)

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export directory identity changed while reading",
    ):
        hf_export._read_export_snapshot(root)

    assert replaced


def test_hf_snapshot_recheck_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    root = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(23.0), identity=identity("i"))
    export_hf_directory(
        checkpoint,
        root,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    failure = OSError("simulated HF root identity recheck failure")
    real_lstat = Path.lstat
    root_checks = 0

    def fail_root_recheck(path: Path):
        nonlocal root_checks
        if path == root:
            root_checks += 1
            if root_checks == 2:
                raise failure
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", fail_root_recheck)

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot inspect HF-style export directory while reading",
    ) as caught:
        hf_export._read_export_snapshot(root)

    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_hf_snapshot_recheck_interrupt_retains_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    checkpoint = tmp_path / "checkpoint"
    root = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(24.0), identity=identity("j"))
    export_hf_directory(
        checkpoint,
        root,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    primary = interrupt_type("simulated HF root identity recheck interrupt")
    real_lstat = Path.lstat
    root_checks = 0

    def interrupt_root_recheck(path: Path):
        nonlocal root_checks
        if path == root:
            root_checks += 1
            if root_checks == 2:
                raise primary
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", interrupt_root_recheck)

    with pytest.raises(interrupt_type) as caught:
        hf_export._read_export_snapshot(root)

    assert caught.value is primary


def _reseal_hf_after_source_manifest_mutation(
    output: Path,
    source_manifest: dict,
) -> None:
    source_manifest["checkpoint_id"] = hf_export.hash_json(
        {
            "identity": source_manifest["identity"],
            "files": source_manifest["files"],
        }
    )
    source_bytes = hf_export._strict_json_bytes(
        source_manifest,
        artifact=hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
    ) + b"\n"
    (output / hf_export.EXPORTED_SOURCE_MANIFEST_NAME).write_bytes(source_bytes)

    parity_path = output / hf_export.PARITY_REQUEST_NAME
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["checkpoint_id"] = source_manifest["checkpoint_id"]
    parity_bytes = hf_export._strict_json_bytes(
        parity,
        artifact=hf_export.PARITY_REQUEST_NAME,
    ) + b"\n"
    parity_path.write_bytes(parity_bytes)

    attestation_path = output / hf_export.EXPORT_ATTESTATION_NAME
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["checkpoint_id"] = source_manifest["checkpoint_id"]
    attestation["source_manifest_sha256"] = hf_export.sha256_bytes(source_bytes)
    attestation["parity_request_sha256"] = hf_export.sha256_bytes(parity_bytes)
    attestation_bytes = hf_export._strict_json_bytes(
        attestation,
        artifact=hf_export.EXPORT_ATTESTATION_NAME,
    ) + b"\n"
    attestation_path.write_bytes(attestation_bytes)
    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        (
            f"{hf_export.sha256_bytes(attestation_bytes)}  "
            f"{hf_export.EXPORT_ATTESTATION_NAME}\n"
        ),
        encoding="ascii",
    )


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("unknown_top_level", "fields mismatch"),
        ("missing_file", "file inventory mismatch"),
        ("extra_file", "file inventory mismatch"),
        ("record_extra_field", "fields mismatch"),
        ("serialization_drift", "serialization declaration is noncanonical"),
        ("bool_format_version", "format_version must be an integer"),
        ("float_format_version", "format_version must be an integer"),
        ("invalid_created_at", "created_at_utc is not canonical UTC"),
        ("offset_created_at", "created_at_utc is not canonical UTC"),
        ("nonstring_created_at", "created_at_utc is not canonical UTC"),
    ),
)
def test_hf_verifier_rejects_resealed_noncanonical_source_manifest(
    tmp_path: Path,
    case: str,
    expected: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(25.0), identity=identity("k"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    source_path = output / hf_export.EXPORTED_SOURCE_MANIFEST_NAME
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))

    if case == "unknown_top_level":
        source_manifest["unexpected_authority"] = True
    elif case == "missing_file":
        source_manifest["files"].pop(hf_export.STATE_TREE_NAME)
    elif case == "extra_file":
        source_manifest["files"]["unexpected.bin"] = {
            "sha256": "0" * 64,
            "bytes": 0,
        }
    elif case == "record_extra_field":
        source_manifest["files"][hf_export.WEIGHTS_NAME]["unexpected"] = False
    elif case == "serialization_drift":
        source_manifest["serialization"]["pickle"] = True
    elif case == "bool_format_version":
        source_manifest["format_version"] = True
    elif case == "float_format_version":
        source_manifest["format_version"] = 1.0
    elif case == "invalid_created_at":
        source_manifest["created_at_utc"] = "not-a-timestamp"
    elif case == "offset_created_at":
        source_manifest["created_at_utc"] = "2026-10-06T17:00:00+00:00"
    elif case == "nonstring_created_at":
        source_manifest["created_at_utc"] = 0
    else:
        raise AssertionError(f"unhandled mutation case: {case}")

    _reseal_hf_after_source_manifest_mutation(output, source_manifest)

    with pytest.raises(CheckpointIntegrityError, match=expected):
        verify_hf_directory(output)


def test_hf_verifier_preserves_unsupported_integer_source_version_semantics(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(25.5), identity=identity("n"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    source_path = output / hf_export.EXPORTED_SOURCE_MANIFEST_NAME
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest["format_version"] = 2
    _reseal_hf_after_source_manifest_mutation(output, source_manifest)

    with pytest.raises(
        hf_export.CheckpointCompatibilityError,
        match="unsupported checkpoint format",
    ):
        verify_hf_directory(output)


def test_hf_verifier_rejects_nonmapping_source_files_as_typed_integrity(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(26.0), identity=identity("l"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    source_path = output / hf_export.EXPORTED_SOURCE_MANIFEST_NAME
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_manifest["files"] = ["not", "a", "mapping"]

    source_bytes = hf_export._strict_json_bytes(
        source_manifest,
        artifact=hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
    ) + b"\n"
    source_path.write_bytes(source_bytes)

    with pytest.raises(
        CheckpointIntegrityError,
        match="exported source manifest files must be a mapping",
    ):
        verify_hf_directory(output)


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("unknown_field", "identity fields mismatch"),
        ("bad_git_sha", "identity.git_sha"),
        ("bad_tokenizer_hash", "identity.tokenizer_hash"),
        ("bad_dataset_hash", "identity.dataset_manifest_hash"),
        ("bad_run_hash", "identity.run_manifest_hash"),
        ("bad_environment_lock_hash", "identity.environment_lock_hash"),
        ("bool_parameter_count", "identity.parameter_count"),
        ("negative_step", "identity.step and identity.tokens_seen"),
        (
            "bound_run_manifest_mismatch",
            "identity.training_config run manifest hash disagrees with identity",
        ),
    ),
)
def test_hf_verifier_rejects_resealed_noncanonical_source_identity(
    tmp_path: Path,
    case: str,
    expected: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(27.0), identity=identity("m"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    source_path = output / hf_export.EXPORTED_SOURCE_MANIFEST_NAME
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source_identity = source_manifest["identity"]

    if case == "unknown_field":
        source_identity["training_authorized"] = True
    elif case == "bad_git_sha":
        source_identity["git_sha"] = "not-a-git-sha"
    elif case == "bad_tokenizer_hash":
        source_identity["tokenizer_hash"] = "g" * 64
    elif case == "bad_dataset_hash":
        source_identity["dataset_manifest_hash"] = "g" * 64
    elif case == "bad_run_hash":
        source_identity["run_manifest_hash"] = "g" * 64
    elif case == "bad_environment_lock_hash":
        source_identity["environment_lock_hash"] = "g" * 64
    elif case == "bool_parameter_count":
        source_identity["parameter_count"] = True
    elif case == "negative_step":
        source_identity["step"] = -1
    elif case == "bound_run_manifest_mismatch":
        source_identity["training_config"]["run_manifest_sha256"] = "5" * 64
        source_identity["training_config_hash"] = hf_export.hash_json(
            source_identity["training_config"]
        )
    else:
        raise AssertionError(f"unhandled identity mutation case: {case}")

    _reseal_hf_after_source_manifest_mutation(output, source_manifest)

    with pytest.raises(CheckpointIntegrityError, match=expected):
        verify_hf_directory(output)

@pytest.mark.parametrize(
    "artifact_name",
    (
        hf_export.EXPORTED_CONFIG_NAME,
        hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
        hf_export.PARITY_REQUEST_NAME,
        hf_export.EXPORT_ATTESTATION_NAME,
        hf_export.EXPORT_CHECKSUM_NAME,
    ),
)
def test_hf_snapshot_rejects_oversized_metadata_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(28.0), identity=identity("o"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )

    if artifact_name == hf_export.EXPORT_CHECKSUM_NAME:
        limit = (output / artifact_name).stat().st_size
        monkeypatch.setattr(hf_export, "_MAX_EXPORT_CHECKSUM_BYTES", limit)
    else:
        metadata_names = (
            hf_export.EXPORTED_CONFIG_NAME,
            hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
            hf_export.PARITY_REQUEST_NAME,
            hf_export.EXPORT_ATTESTATION_NAME,
        )
        limit = max((output / name).stat().st_size for name in metadata_names)
        monkeypatch.setattr(hf_export, "_MAX_EXPORT_METADATA_BYTES", limit)

    (output / artifact_name).write_bytes(b"x" * (limit + 1))

    with pytest.raises(
        CheckpointIntegrityError,
        match=f"HF-style export artifact exceeds read limit: {artifact_name}",
    ):
        hf_export._read_export_snapshot(output)


def test_hf_bounded_read_rejects_growth_after_stale_fstat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    root.mkdir()
    artifact = root / "metadata.json"
    artifact.write_bytes(b"x" * 33)
    real_fstat = hf_export.os.fstat

    def stale_small_fstat(fd: int):
        observed = real_fstat(fd)
        return SimpleNamespace(
            st_mode=observed.st_mode,
            st_dev=observed.st_dev,
            st_ino=observed.st_ino,
            st_size=0,
        )

    monkeypatch.setattr(hf_export.os, "fstat", stale_small_fstat)

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export artifact exceeds read limit: metadata.json",
    ):
        hf_export._read_regular_bytes(
            root,
            "metadata.json",
            max_bytes=32,
        )


def test_hf_snapshot_reads_only_bounded_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(29.0), identity=identity("p"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    real_read = hf_export._read_regular_bytes
    observed_limits: dict[str, int | None] = {}

    def observe_limit(
        root: Path,
        name: str,
        *,
        max_bytes: int | None = None,
    ):
        assert name != hf_export.EXPORTED_WEIGHTS_NAME
        observed_limits[name] = max_bytes
        return real_read(root, name, max_bytes=max_bytes)

    def forbid_weight_stream(*_args, **_kwargs):
        raise AssertionError("metadata snapshot must not read model weights")

    monkeypatch.setattr(hf_export, "_read_regular_bytes", observe_limit)
    monkeypatch.setattr(hf_export, "_stream_regular_sha256", forbid_weight_stream)

    hf_export._read_export_snapshot(output)

    assert hf_export.EXPORTED_WEIGHTS_NAME not in observed_limits
    assert observed_limits[hf_export.EXPORT_CHECKSUM_NAME] == 256
    for name in (
        hf_export.EXPORTED_CONFIG_NAME,
        hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
        hf_export.PARITY_REQUEST_NAME,
        hf_export.EXPORT_ATTESTATION_NAME,
    ):
        assert observed_limits[name] == 8 * 1024 * 1024


def test_hf_streamed_weight_digest_is_exact_across_multiple_chunks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    root.mkdir()
    name = hf_export.EXPORTED_WEIGHTS_NAME
    payload = b"streamed-weights-" * 17
    (root / name).write_bytes(payload)
    monkeypatch.setattr(hf_export, "_STREAM_HASH_CHUNK_BYTES", 13)

    digest, byte_count = hf_export._stream_regular_sha256(root, name)

    assert digest == hf_export.sha256_bytes(payload)
    assert byte_count == len(payload)


def test_hf_streamed_weight_read_oserror_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    root.mkdir()
    name = hf_export.EXPORTED_WEIGHTS_NAME
    (root / name).write_bytes(b"streamed weights")
    failure = OSError("simulated streamed weight read failure")

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(failure),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot read HF-style export artifact",
    ) as caught:
        hf_export._stream_regular_sha256(root, name)

    assert caught.value.__cause__ is failure


def test_hf_streamed_weight_read_oserror_survives_close_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "root"
    root.mkdir()
    name = hf_export.EXPORTED_WEIGHTS_NAME
    (root / name).write_bytes(b"streamed weights")
    failure = OSError("simulated streamed weight read failure")
    real_close = hf_export.os.close
    real_fstat = hf_export.os.fstat
    closed: list[int] = []

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(failure),
    )
    monkeypatch.setattr(
        hf_export.os,
        "close",
        _close_then_raise_oserror(real_close, real_fstat, closed),
    )

    with pytest.raises(
        CheckpointIntegrityError,
        match="cannot read HF-style export artifact",
    ) as caught:
        hf_export._stream_regular_sha256(root, name)

    assert caught.value.__cause__ is failure
    assert closed
    assert any(
        "HF-style export artifact close also failed" in note
        for note in getattr(caught.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    "interrupt_type",
    [KeyboardInterrupt, SystemExit, GeneratorExit],
)
def test_hf_streamed_weight_interrupt_retains_exact_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_type: type[BaseException],
):
    root = tmp_path / "root"
    root.mkdir()
    name = hf_export.EXPORTED_WEIGHTS_NAME
    (root / name).write_bytes(b"streamed weights")
    primary = interrupt_type("simulated streamed weight interruption")

    monkeypatch.setattr(
        hf_export.os,
        "fdopen",
        lambda *_args, **_kwargs: _FailingReadHandle(primary),
    )

    with pytest.raises(interrupt_type) as caught:
        hf_export._stream_regular_sha256(root, name)

    assert caught.value is primary

@pytest.mark.parametrize(
    ("artifact_name", "replacement", "expected"),
    (
        (
            hf_export.EXPORT_CHECKSUM_NAME,
            b"invalid checksum\n",
            "invalid 12-6-export.sha256 format",
        ),
        (
            hf_export.EXPORTED_SOURCE_MANIFEST_NAME,
            b"{invalid-json\n",
            "12-6-checkpoint-manifest.json is not valid strict UTF-8 JSON",
        ),
        (
            hf_export.EXPORTED_CONFIG_NAME,
            b"{invalid-json\n",
            "config.json is not valid strict UTF-8 JSON",
        ),
        (
            hf_export.PARITY_REQUEST_NAME,
            b"{invalid-json\n",
            "12-6-parity-request.json is not valid strict UTF-8 JSON",
        ),
    ),
)
def test_hf_verifier_rejects_bad_metadata_before_weight_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_name: str,
    replacement: bytes,
    expected: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(29.5), identity=identity("r"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    (output / artifact_name).write_bytes(replacement)

    def forbid_weight_stream(*_args, **_kwargs):
        raise AssertionError("invalid bounded metadata must fail before weight scan")

    monkeypatch.setattr(hf_export, "_stream_regular_sha256", forbid_weight_stream)

    with pytest.raises(CheckpointIntegrityError, match=expected):
        verify_hf_directory(output)


def test_hf_verifier_rejects_parity_weight_binding_before_weight_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(29.625), identity=identity("u"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    parity_path = output / hf_export.PARITY_REQUEST_NAME
    parity = json.loads(parity_path.read_text(encoding="utf-8"))
    parity["candidate_weights_sha256"] = "0" * 64
    parity_path.write_bytes(
        hf_export._strict_json_bytes(
            parity,
            artifact=hf_export.PARITY_REQUEST_NAME,
        )
        + b"\n"
    )

    def forbid_weight_stream(*_args, **_kwargs):
        raise AssertionError("bad parity binding must fail before weight scan")

    monkeypatch.setattr(hf_export, "_stream_regular_sha256", forbid_weight_stream)

    with pytest.raises(
        CheckpointIntegrityError,
        match="export parity request candidate_weights_sha256 mismatch",
    ):
        verify_hf_directory(output)


@pytest.mark.parametrize("case", ("invalid_json", "weight_binding"))
def test_hf_verifier_rejects_resealed_bad_attestation_before_weight_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(29.6875), identity=identity("v"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    attestation_path = output / hf_export.EXPORT_ATTESTATION_NAME
    if case == "invalid_json":
        attestation_bytes = b"{invalid-json\n"
        expected = "12-6-export.json is not valid strict UTF-8 JSON"
    else:
        attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
        attestation["model_safetensors_sha256"] = "0" * 64
        attestation_bytes = hf_export._strict_json_bytes(
            attestation,
            artifact=hf_export.EXPORT_ATTESTATION_NAME,
        ) + b"\n"
        expected = "HF-style export attestation model_safetensors_sha256 mismatch"
    attestation_path.write_bytes(attestation_bytes)
    (output / hf_export.EXPORT_CHECKSUM_NAME).write_text(
        (
            f"{hf_export.sha256_bytes(attestation_bytes)}  "
            f"{hf_export.EXPORT_ATTESTATION_NAME}\n"
        ),
        encoding="ascii",
    )

    def forbid_weight_stream(*_args, **_kwargs):
        raise AssertionError("bad attestation must fail before weight scan")

    monkeypatch.setattr(hf_export, "_stream_regular_sha256", forbid_weight_stream)

    with pytest.raises(CheckpointIntegrityError, match=expected):
        verify_hf_directory(output)


def test_hf_verifier_streams_weights_once_after_metadata_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(29.75), identity=identity("s"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    real_stream = hf_export._stream_regular_sha256
    streamed: list[str] = []

    def observe_stream(root: Path, name: str):
        streamed.append(name)
        return real_stream(root, name)

    monkeypatch.setattr(hf_export, "_stream_regular_sha256", observe_stream)

    verify_hf_directory(output)

    assert streamed == [hf_export.EXPORTED_WEIGHTS_NAME]


def test_hf_verifier_rejects_root_replacement_before_weight_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    moved = tmp_path / "hf-original"
    save_checkpoint(checkpoint, model=Model(29.875), identity=identity("t"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    real_snapshot = hf_export._read_export_snapshot
    streamed = False

    def replace_after_metadata(path: Path):
        payloads, root_identity = real_snapshot(path)
        path.rename(moved)
        path.mkdir()
        return payloads, root_identity

    def observe_stream(*_args, **_kwargs):
        nonlocal streamed
        streamed = True
        raise AssertionError("replaced root must fail before weight stream")

    monkeypatch.setattr(hf_export, "_read_export_snapshot", replace_after_metadata)
    monkeypatch.setattr(hf_export, "_stream_regular_sha256", observe_stream)

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export directory identity changed while reading",
    ):
        verify_hf_directory(output)

    assert not streamed


def test_hf_snapshot_rejects_seventh_inventory_entry(
    tmp_path: Path,
):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    save_checkpoint(checkpoint, model=Model(30.0), identity=identity("q"))
    export_hf_directory(
        checkpoint,
        output,
        hf_config={"model_type": "twelve_six_export_transactional"},
    )
    (output / "unexpected-seventh.txt").write_text("x", encoding="utf-8")

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export inventory exceeds expected size",
    ):
        hf_export._read_export_snapshot(output)


def test_hf_snapshot_stops_inventory_after_seventh_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "hf"
    root.mkdir()
    real_iterdir = Path.iterdir
    consumed: list[str] = []

    def oversized_entries():
        for name in sorted(hf_export._EXPORT_FILES):
            consumed.append(name)
            yield SimpleNamespace(name=name)
        consumed.append("unexpected-seventh.txt")
        yield SimpleNamespace(name="unexpected-seventh.txt")
        raise AssertionError("eighth inventory entry must not be requested")

    def bounded_iterdir(path: Path):
        if path == root:
            return oversized_entries()
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", bounded_iterdir)

    with pytest.raises(
        CheckpointIntegrityError,
        match="HF-style export inventory exceeds expected size",
    ):
        hf_export._read_export_snapshot(root)

    assert len(consumed) == len(hf_export._EXPORT_FILES) + 1

