import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "validate_eval303_selection_validation_composite.py"
MANIFEST = ROOT / "configs/evaluation/eval303_selection_validation_composite_v1.json"
PROOF = ROOT / "evidence/eval303/data300-exact-exclusion-proof-v1.json"
EXPECTED_SELECTION_IDENTITY = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"


def _load_validator():
    spec = importlib.util.spec_from_file_location("_eval303_validator_test", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify() -> dict:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--repo-root", str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_eval303_composite_verifies():
    result = _verify()
    assert result["status"] == "PASS"
    assert result["selection_identity_sha256"] == EXPECTED_SELECTION_IDENTITY
    assert result["documents"] == 10


def test_selection_only_and_code_fail_closed():
    manifest = _load_json(MANIFEST)
    usage = manifest["usage_contract"]
    assert usage["selection_only"] is True
    assert usage["may_train"] is False
    assert usage["may_fit_tokenizer"] is False
    assert usage["may_update_model"] is False
    assert usage["may_report_final_test"] is False
    assert manifest["strata"]["code"]["documents"] == 0
    assert manifest["strata"]["code"]["selection_eligible"] is False


def test_data300_exact_exclusion_is_hash_bound_not_family_washed():
    proof = _load_json(PROOF)
    comparisons = proof["comparisons"]
    assert comparisons["selected_content_vs_training_raw_or_normalized_sha256_overlap"] == []
    assert comparisons["selected_git_blob_vs_training_git_blob_overlap"] == []
    assert comparisons["selected_source_family_vs_training_source_family_overlap"] == [
        "github:encode/httpx",
        "github:psf/requests",
    ]
    assert len(comparisons["same_family_distinct_object_evidence"]) == 2
    assert proof["verdict"]["near_copy_or_dedup_cluster_scan_claimed"] is False


def test_final_test_outcomes_and_payload_not_consumed_by_eval303():
    firewall = _load_json(MANIFEST)["final_test_firewall"]
    assert firewall["outcomes_read_by_eval303"] is False
    assert firewall["final_test_payload_read_by_eval303"] is False
    assert firewall["final_test_bytes_copied_into_composite"] is False


@pytest.mark.parametrize(
    "raw",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":-1e400}',
    ],
)
def test_strict_decoder_rejects_non_finite_numbers(raw: str) -> None:
    with pytest.raises(VALIDATOR.Eval303ValidationError):
        VALIDATOR._decode_json_object(raw, label="fixture")


def test_strict_decoder_rejects_nested_duplicate_members() -> None:
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="duplicate JSON key: value",
    ):
        VALIDATOR._decode_json_object(
            '{"outer":{"value":1,"value":2}}',
            label="fixture",
        )


def test_strict_decoder_preserves_valid_finite_json() -> None:
    assert VALIDATOR._decode_json_object(
        '{"value":1e20,"nested":{"ok":true}}',
        label="fixture",
    ) == {"value": 1e20, "nested": {"ok": True}}


def test_membership_jsonl_rejects_non_finite_constant(tmp_path: Path) -> None:
    membership = tmp_path / "membership.jsonl"
    membership.write_text('{"record_id":"x","score":NaN}\n', encoding="utf-8")
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="non-finite JSON constant",
    ):
        VALIDATOR.load_records(membership)


def test_materialize_fresh_directory_reverifies_exact_files(tmp_path: Path) -> None:
    output = tmp_path / "new output"
    VALIDATOR.materialize(ROOT, output)

    for relative in (VALIDATOR.MANIFEST, VALIDATOR.MEMBERSHIP, VALIDATOR.PROOF):
        assert (output / relative).read_bytes() == (ROOT / relative).read_bytes()
    assert VALIDATOR.verify(output) == VALIDATOR.verify(ROOT)
    assert not list(tmp_path.glob(".new output.staging-*"))


def test_materialize_existing_user_directory_is_never_deleted(tmp_path: Path) -> None:
    output = tmp_path / "existing output"
    output.mkdir()
    marker = output / "unrelated user data.txt"
    marker.write_text("preserve me", encoding="utf-8")

    with pytest.raises(VALIDATOR.Eval303ValidationError, match="already exists"):
        VALIDATOR.materialize(ROOT, output)

    assert marker.read_text(encoding="utf-8") == "preserve me"
    assert not list(tmp_path.glob(".existing output.staging-*"))


def test_materialize_cannot_delete_its_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep this adversarial source in tmp_path, never risk the actual checkout.
    root = tmp_path / "source"
    root.mkdir()
    marker = root / "do not delete.txt"
    marker.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr(VALIDATOR, "verify", lambda _root: {"status": "PASS"})

    with pytest.raises(VALIDATOR.Eval303ValidationError, match="already exists"):
        VALIDATOR.materialize(root, root)

    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_materialize_rejects_final_symlink_without_touching_target(
    tmp_path: Path,
) -> None:
    protected = tmp_path / "protected"
    protected.mkdir()
    marker = protected / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")
    alias = tmp_path / "symlink output"
    try:
        alias.symlink_to(protected, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(VALIDATOR.Eval303ValidationError, match="already exists"):
        VALIDATOR.materialize(ROOT, alias)

    assert alias.is_symlink()
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_materialize_destination_race_fails_without_clobber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "racing output"
    publish = VALIDATOR._publish_directory_noreplace

    def create_destination_first(staging: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "other-owner.txt").write_text("preserve me", encoding="utf-8")
        publish(staging, destination)

    monkeypatch.setattr(
        VALIDATOR, "_publish_directory_noreplace", create_destination_first
    )
    with pytest.raises(FileExistsError, match="appeared during publish"):
        VALIDATOR.materialize(ROOT, output)

    assert (output / "other-owner.txt").read_text(encoding="utf-8") == "preserve me"
    retained = list(tmp_path.glob(".racing output.staging-*"))
    assert len(retained) == 1
    assert (retained[0] / VALIDATOR.MANIFEST).is_file()


def test_materialize_failed_stage_verification_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "unpublished"
    original_verify = VALIDATOR.verify

    def fail_second_verification(root: Path) -> dict:
        if root != ROOT:
            raise VALIDATOR.Eval303ValidationError("candidate did not reverify")
        return original_verify(root)

    monkeypatch.setattr(VALIDATOR, "verify", fail_second_verification)
    with pytest.raises(VALIDATOR.Eval303ValidationError, match="did not reverify"):
        VALIDATOR.materialize(ROOT, output)

    assert not output.exists()
    retained = list(tmp_path.glob(".unpublished.staging-*"))
    assert len(retained) == 1
    assert (retained[0] / VALIDATOR.MANIFEST).is_file()


def test_materialize_existing_target_cli_failure_is_one_json_line(
    tmp_path: Path,
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "materialize",
            "--repo-root", str(ROOT), "--output-dir", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "FAIL"
    assert marker.read_text(encoding="utf-8") == "preserve me"


def test_materialize_success_cli_reports_published_directory(tmp_path: Path) -> None:
    output = tmp_path / "fresh"
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "materialize",
            "--repo-root", str(ROOT), "--output-dir", str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stderr == ""
    assert json.loads(result.stdout)["status"] == "PASS"
    assert VALIDATOR.verify(output)["status"] == "PASS"


def test_verify_cli_semantic_error_is_machine_readable(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "verify", "--repo-root", str(empty),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "FAIL"


def test_verify_cli_unexpected_error_is_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_root: Path) -> dict:
        raise RuntimeError("unexpected programming error")

    monkeypatch.setattr(VALIDATOR, "verify", unexpected)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "verify", "--repo-root", str(ROOT)]
    )
    with pytest.raises(RuntimeError, match="unexpected programming error"):
        VALIDATOR.main()


def test_strict_decoder_rejects_integer_exceeding_interpreter_limit() -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer digit limit is disabled")
    raw = '{"value":' + '9' * (limit + 1) + '}'
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="JSON integer exceeds interpreter limit",
    ):
        VALIDATOR._decode_json_object(raw, label="fixture")


@pytest.mark.parametrize(
    "relative",
    [VALIDATOR.MANIFEST, VALIDATOR.PROOF, VALIDATOR.MEMBERSHIP],
)
def test_verify_cli_rejects_oversized_json_integer_as_one_error(
    tmp_path: Path, relative: Path
) -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer digit limit is disabled")
    for original in (VALIDATOR.MANIFEST, VALIDATOR.PROOF, VALIDATOR.MEMBERSHIP):
        target = tmp_path / original
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / original).read_bytes())
    (tmp_path / relative).write_text(
        '{"count":' + '9' * (limit + 1) + '}\n', encoding="utf-8"
    )
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "verify", "--repo-root", str(tmp_path)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert result.stderr == ""
    assert len(result.stdout.splitlines()) == 1
    failure = json.loads(result.stdout)
    assert failure["status"] == "FAIL"
    assert "JSON integer exceeds interpreter limit" in failure["error"]


def test_failed_materialize_never_deletes_swapped_stage_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unpublished"
    unrelated = tmp_path / "user data"
    unrelated.mkdir()
    marker = unrelated / "keep.txt"
    marker.write_text("preserve me", encoding="utf-8")
    original_verify = VALIDATOR.verify

    def swap_and_fail(root: Path) -> dict:
        if root != ROOT:
            root.rename(tmp_path / "original stage retained")
            unrelated.rename(root)
            raise VALIDATOR.Eval303ValidationError("stage verification failed")
        return original_verify(root)

    monkeypatch.setattr(VALIDATOR, "verify", swap_and_fail)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError, match="stage verification failed"
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    swapped = list(tmp_path.glob(".unpublished.staging-*"))
    assert len(swapped) == 1
    assert (swapped[0] / "keep.txt").read_text(encoding="utf-8") == "preserve me"
    assert (tmp_path / "original stage retained" / VALIDATOR.MANIFEST).is_file()


def test_failed_materialize_never_unlinks_substituted_user_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unpublished"
    user_file = tmp_path / "outside marker.txt"
    user_file.write_text("preserve me", encoding="utf-8")
    original_verify = VALIDATOR.verify

    def replace_file_and_fail(root: Path) -> dict:
        if root != ROOT:
            private = root / VALIDATOR.MANIFEST
            private.rename(root / "displaced private manifest")
            user_file.rename(private)
            raise VALIDATOR.Eval303ValidationError("stage verification failed")
        return original_verify(root)

    monkeypatch.setattr(VALIDATOR, "verify", replace_file_and_fail)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError, match="stage verification failed"
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    retained = list(tmp_path.glob(".unpublished.staging-*"))
    assert len(retained) == 1
    assert (retained[0] / VALIDATOR.MANIFEST).read_text(
        encoding="utf-8"
    ) == "preserve me"
    assert (retained[0] / "displaced private manifest").is_file()
    with pytest.raises(
        VALIDATOR.Eval303ValidationError, match="previous private stage exists"
    ):
        VALIDATOR.materialize(ROOT, output)
    assert len(list(tmp_path.glob(".unpublished.staging-*"))) == 1


def test_failed_materialize_preserves_unexpected_stage_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unexpected"
    original_verify = VALIDATOR.verify
    injected = []

    def inject_unexpected_entry(root: Path) -> dict:
        if root != ROOT:
            extra = root / "unrelated extra.txt"
            extra.write_text("preserve me", encoding="utf-8")
            injected.append(extra)
            raise VALIDATOR.Eval303ValidationError("staging verification failed")
        return original_verify(root)

    monkeypatch.setattr(VALIDATOR, "verify", inject_unexpected_entry)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError, match="staging verification failed"
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    assert len(injected) == 1
    assert injected[0].read_text(encoding="utf-8") == "preserve me"


def test_cli_materialize_failure_reports_retained_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "failed cli"

    def fail_publication(_staging: Path, _destination: Path) -> None:
        raise OSError("injected publication failure")

    monkeypatch.setattr(VALIDATOR, "_publish_directory_noreplace", fail_publication)
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SCRIPT), "materialize", "--repo-root", str(ROOT), "--output-dir", str(output)],
    )
    assert VALIDATOR.main() == 2
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.splitlines()) == 1
    failure = json.loads(captured.out)
    assert failure["status"] == "FAIL"
    assert "injected publication failure" in failure["error"]
    assert "private stage retained for manual inspection" in failure["error"]
    assert not output.exists()
    retained = list(tmp_path.glob(".failed cli.staging-*"))
    assert len(retained) == 1
    assert str(retained[0]) in failure["error"]


def _deep_authority_json(nesting: str) -> str:
    if nesting == "arrays":
        return '{"root":' + "[" * 10000 + "0" + "]" * 10000 + "}"
    return '{"root":' + '{"k":' * 10000 + "0" + "}" * 10000 + "}"


@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_strict_decoder_rejects_excessive_json_nesting(nesting: str) -> None:
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="JSON nesting limit exceeded",
    ):
        VALIDATOR._decode_json_object(
            _deep_authority_json(nesting),
            label="fixture",
        )


@pytest.mark.parametrize("action", ("verify", "materialize"))
@pytest.mark.parametrize(
    "relative",
    (VALIDATOR.MANIFEST, VALIDATOR.PROOF, VALIDATOR.MEMBERSHIP),
)
@pytest.mark.parametrize("nesting", ("arrays", "objects"))
def test_cli_rejects_deep_authority_json_without_publication(
    tmp_path: Path,
    action: str,
    relative: Path,
    nesting: str,
) -> None:
    paths = []
    for source in (VALIDATOR.MANIFEST, VALIDATOR.PROOF, VALIDATOR.MEMBERSHIP):
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / source).read_bytes())
        paths.append(target)
    target = tmp_path / relative
    raw = _deep_authority_json(nesting)
    if relative == VALIDATOR.MEMBERSHIP:
        raw += "\n"
    target.write_text(raw, encoding="utf-8")
    originals = {path: path.read_bytes() for path in paths}
    output = tmp_path / "must-not-publish"

    args = [sys.executable, str(SCRIPT), action, "--repo-root", str(tmp_path)]
    if action == "materialize":
        args.extend(("--output-dir", str(output)))
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == 1
    report = json.loads(lines[0])
    assert report["status"] == "FAIL"
    assert "JSON nesting limit exceeded" in report["error"]
    assert {path: path.read_bytes() for path in paths} == originals
    assert not output.exists()


def test_cli_does_not_mask_unexpected_product_recursion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(_root: Path) -> dict:
        raise RecursionError("unexpected Product recursion")

    monkeypatch.setattr(VALIDATOR, "verify", unexpected)
    monkeypatch.setattr(
        sys, "argv", [str(SCRIPT), "verify", "--repo-root", str(ROOT)]
    )
    with pytest.raises(RecursionError, match="unexpected Product recursion"):
        VALIDATOR.main()


def test_successful_materialize_publishes_only_authenticated_tree(
    tmp_path: Path,
) -> None:
    output = tmp_path / "exact authenticated output"
    VALIDATOR.materialize(ROOT, output)
    expected_files = {
        VALIDATOR.MANIFEST, VALIDATOR.MEMBERSHIP, VALIDATOR.PROOF
    }
    expected_dirs = {
        parent
        for relative in expected_files
        for parent in relative.parents
        if parent != Path(".")
    }
    actual = {entry.relative_to(output) for entry in output.rglob("*")}
    assert actual == expected_files | expected_dirs
    for relative in expected_files:
        assert (output / relative).read_bytes() == (ROOT / relative).read_bytes()


@pytest.mark.parametrize("entry_kind", ("file", "directory"))
def test_verified_stage_does_not_publish_unexpected_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry_kind: str
) -> None:
    output = tmp_path / "unpublished extra"
    original_verify = VALIDATOR.verify

    def add_extra_after_verified_files(root: Path) -> dict:
        report = original_verify(root)
        if root != ROOT:
            extra = root / "unverified extra"
            if entry_kind == "file":
                extra.write_text("outside verified evidence", encoding="utf-8")
            else:
                extra.mkdir()
        return report

    monkeypatch.setattr(VALIDATOR, "verify", add_extra_after_verified_files)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="private staging tree has unexpected or missing entries",
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    candidates = list(tmp_path.glob(".unpublished extra.staging-*"))
    assert len(candidates) == 1
    assert (candidates[0] / "unverified extra").exists()
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="previous private stage exists",
    ):
        VALIDATOR.materialize(ROOT, output)
    assert len(list(tmp_path.glob(".unpublished extra.staging-*"))) == 1


def test_verified_stage_rejects_symlinked_expected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe = tmp_path / "symlink probe"
    try:
        probe.symlink_to(ROOT / VALIDATOR.MANIFEST)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable on this host")
    probe.unlink()
    output = tmp_path / "unpublished symlink"
    original_verify = VALIDATOR.verify

    def swap_expected_file_after_verify(root: Path) -> dict:
        report = original_verify(root)
        if root != ROOT:
            candidate = root / VALIDATOR.MANIFEST
            candidate.unlink()
            candidate.symlink_to(ROOT / VALIDATOR.MANIFEST)
        return report

    monkeypatch.setattr(VALIDATOR, "verify", swap_expected_file_after_verify)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="private materialization staging path changed",
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    assert (ROOT / VALIDATOR.MANIFEST).is_file()
    candidates = list(tmp_path.glob(".unpublished symlink.staging-*"))
    assert len(candidates) == 1
    assert (candidates[0] / VALIDATOR.MANIFEST).is_symlink()


def test_verified_stage_rejects_root_swap_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unpublished swap"
    original_verify = VALIDATOR.verify
    original_stage = tmp_path / "original stage retained"

    def replace_stage_after_verify(root: Path) -> dict:
        report = original_verify(root)
        if root != ROOT:
            impostor = tmp_path / "impostor"
            for relative in (
                VALIDATOR.MANIFEST, VALIDATOR.MEMBERSHIP, VALIDATOR.PROOF
            ):
                copied = impostor / relative
                copied.parent.mkdir(parents=True, exist_ok=True)
                copied.write_bytes((ROOT / relative).read_bytes())
            root.rename(original_stage)
            impostor.rename(root)
        return report

    monkeypatch.setattr(VALIDATOR, "verify", replace_stage_after_verify)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="private materialization staging path changed",
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    assert (original_stage / VALIDATOR.MANIFEST).is_file()
    assert len(list(tmp_path.glob(".unpublished swap.staging-*"))) == 1


def test_verified_stage_rejects_same_bytes_replaced_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "unpublished replaced file"
    original_verify = VALIDATOR.verify
    displaced = tmp_path / "displaced genuine file"

    def swap_file_after_verify(root: Path) -> dict:
        report = original_verify(root)
        if root != ROOT:
            candidate = root / VALIDATOR.MANIFEST
            candidate.rename(displaced)
            candidate.write_bytes((ROOT / VALIDATOR.MANIFEST).read_bytes())
        return report

    monkeypatch.setattr(VALIDATOR, "verify", swap_file_after_verify)
    with pytest.raises(
        VALIDATOR.Eval303ValidationError,
        match="private materialization staging path changed",
    ):
        VALIDATOR.materialize(ROOT, output)
    assert not output.exists()
    assert displaced.read_bytes() == (ROOT / VALIDATOR.MANIFEST).read_bytes()
    candidates = list(tmp_path.glob(".unpublished replaced file.staging-*"))
    assert len(candidates) == 1
    assert (candidates[0] / VALIDATOR.MANIFEST).read_bytes() == displaced.read_bytes()
