import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from twelve_six.integration_qualification import (
    freeze_candidate,
    load_candidate_freeze,
    load_integration_preview,
    prepare_integration,
    verify_accepted_integration,
    verify_frozen_checkout,
)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", "-c", "user.name=Integration fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "core.autocrlf=false", *arguments),
        cwd=root, stderr=subprocess.PIPE, text=True,
    ).strip()


def _commit(root: Path, message: str) -> str:
    _git(root, "add", "--all")
    _git(root, "commit", "-qm", message)
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def repository(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "owned").mkdir()
    (tmp_path / "owned/component.py").write_bytes(b"version = 1\n")
    (tmp_path / "dependency.txt").write_bytes(b"contract v1\n")
    base = _commit(tmp_path, "base")
    _git(tmp_path, "switch", "-qc", "candidate")
    (tmp_path / "owned/component.py").write_bytes(b"version = 2\n")
    head = _commit(tmp_path, "candidate")
    return tmp_path, base, head


def _freeze(repository, **overrides):
    root, base, head = repository
    options = {
        "base_sha": base,
        "expected_head_sha": head,
        "owned_paths": ("owned",),
        "dependency_paths": ("dependency.txt",),
    }
    return freeze_candidate(root, **(options | overrides))


def _preview(root, freeze, digest, live, **overrides):
    options = {
        "expected_freeze_sha256": digest,
        "qualified_head_sha": freeze.head_sha,
        "qualified_tree_sha": freeze.tree_sha,
        "qualification_sha256": "a" * 64,
        "observed_pr_head_sha": freeze.head_sha,
        "live_base_sha": live,
        "mergeable": True,
        "main_ref": "refs/heads/main",
    }
    return prepare_integration(root, freeze, **(options | overrides))


def test_freeze_rejects_dirty_checkout_head_drift_and_scope_escape(repository):
    root, base, head = repository
    frozen, digest = _freeze(repository)
    assert frozen.changed_paths == ("owned/component.py",)
    verify_frozen_checkout(root, frozen, expected_sha256=digest)
    (root / "unrelated.txt").write_bytes(b"unexpected\n")
    with pytest.raises(ValueError, match="dirty|clean"):
        verify_frozen_checkout(root, frozen, expected_sha256=digest)
    new_head = _commit(root, "unrelated mutation")
    with pytest.raises(ValueError, match="exact-head mismatch"):
        verify_frozen_checkout(root, frozen, expected_sha256=digest)
    with pytest.raises(ValueError, match="outside its declared ownership"):
        _freeze((root, base, new_head))
    with pytest.raises(ValueError, match="canonical repository-relative"):
        _freeze((root, base, new_head), owned_paths=("../outside",))
    assert head != new_head


def test_freeze_rejects_gitlink_hidden_by_config_and_masked_tracked_paths(repository):
    root, base, head = repository
    _git(root, "update-index", "--assume-unchanged", "owned/component.py")
    (root / "owned/component.py").write_bytes(b"unobserved change\n")
    with pytest.raises(ValueError, match="masks tracked paths"):
        _freeze(repository)
    _git(root, "update-index", "--no-assume-unchanged", "owned/component.py")
    _git(root, "restore", "owned/component.py")
    _git(root, "config", "diff.ignoreSubmodules", "all")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{head},foreign")
    _git(root, "commit", "-qm", "unowned gitlink")
    (root / "foreign").mkdir()
    with pytest.raises(ValueError, match="outside its declared ownership"):
        _freeze((root, base, _git(root, "rev-parse", "HEAD")))


def test_stale_qualification_changed_pr_head_and_forged_freeze_are_rejected(repository):
    root, base, head = repository
    frozen, digest = _freeze(repository)
    with pytest.raises(ValueError, match="qualification does not match"):
        _preview(root, frozen, digest, base, qualified_tree_sha="a" * 40)
    with pytest.raises(ValueError, match="observed PR head"):
        _preview(root, frozen, digest, base, observed_pr_head_sha=base)
    with pytest.raises(ValueError, match="mergeability"):
        _preview(root, frozen, digest, base, mergeable=False)
    forged = replace(frozen, dependency_paths=())
    with pytest.raises(ValueError, match="trusted record"):
        _preview(root, forged, digest, base)
    document = json.loads(json.dumps(frozen.to_dict()))
    assert load_candidate_freeze(document, expected_sha256=digest).head_sha == head
    document["dependency_paths"] = []
    with pytest.raises(ValueError, match="trusted record"):
        load_candidate_freeze(document, expected_sha256=digest)


def test_disjoint_main_advance_preserves_actual_squash_tree(repository):
    root, base, head = repository
    frozen, digest = _freeze(repository)
    _git(root, "switch", "-q", "main")
    (root / "tracker.md").write_bytes(b"status updated\n")
    live = _commit(root, "tracker advance")
    before = _git(root, "status", "--porcelain=v1")
    preview, identity = _preview(root, frozen, digest, live)
    assert _git(root, "status", "--porcelain=v1") == before == ""
    assert _git(root, "rev-parse", "HEAD") == live
    assert preview.expected_tree_sha != frozen.tree_sha
    _git(root, "merge", "--squash", head)
    accepted = _commit(root, "accepted")
    document = json.loads(json.dumps(preview.to_dict()))
    restored = load_integration_preview(document, expected_sha256=identity)
    receipt = verify_accepted_integration(
        root, restored, expected_preview_sha256=identity, accepted_sha=accepted,
        observed_main_sha=accepted,
    )
    assert receipt["accepted_tree_sha"] == _git(root, "rev-parse", f"{accepted}^{{tree}}")
    assert receipt["expected_pr_head_sha"] == head and len(receipt["identity_sha256"]) == 64
    with pytest.raises(ValueError, match="canonical main ref"):
        verify_accepted_integration(
            root, restored, expected_preview_sha256=identity, accepted_sha=accepted,
            observed_main_sha=head,
        )
    (root / "dependency.txt").write_bytes(b"later incompatible contract\n")
    drift = _commit(root, "later dependency drift")
    with pytest.raises(ValueError, match="overlap"):
        verify_accepted_integration(
            root, restored, expected_preview_sha256=identity, accepted_sha=accepted,
            observed_main_sha=drift,
        )
    document["qualification_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="trusted record"):
        load_integration_preview(document, expected_sha256=identity)
    assert base != live


def test_current_main_ref_cannot_be_replaced_with_a_valid_unpublished_commit(repository):
    root, base, _ = repository
    frozen, digest = _freeze(repository)
    preview, identity = _preview(root, frozen, digest, base)
    unpublished = _git(root, "commit-tree", frozen.tree_sha, "-p", base, "-m", "unpublished")
    with pytest.raises(ValueError, match="canonical main ref"):
        _preview(root, frozen, digest, unpublished)
    with pytest.raises(ValueError, match="canonical main ref"):
        verify_accepted_integration(
            root, preview, expected_preview_sha256=identity, accepted_sha=unpublished,
            observed_main_sha=unpublished,
        )
    with pytest.raises(ValueError, match="required ancestor"):
        verify_accepted_integration(
            root, preview, expected_preview_sha256=identity, accepted_sha=unpublished,
            observed_main_sha=base,
        )


def test_main_dependency_overlap_and_exact_base_requirement_block(repository):
    root, base, _ = repository
    frozen, digest = _freeze(repository)
    exact, exact_digest = _freeze(repository, exact_base_required=True)
    _git(root, "switch", "-q", "main")
    (root / "dependency.txt").write_bytes(b"contract v2\n")
    live = _commit(root, "dependency advance")
    with pytest.raises(ValueError, match="overlap.*dependency.txt"):
        _preview(root, frozen, digest, live)
    with pytest.raises(ValueError, match="original exact base"):
        _preview(root, exact, exact_digest, live)
    assert base != live


def test_readback_rejects_wrong_tree_and_parent_and_accepts_true_merge(repository):
    root, base, head = repository
    frozen, digest = _freeze(repository)
    preview, identity = _preview(root, frozen, digest, base, strategy="merge")
    _git(root, "switch", "-q", "main")
    _git(root, "merge", "--no-ff", "-m", "accepted merge", head)
    accepted = _git(root, "rev-parse", "HEAD")
    assert verify_accepted_integration(
        root, preview, expected_preview_sha256=identity, accepted_sha=accepted,
        observed_main_sha=accepted,
    )["accepted_sha"] == accepted

    with pytest.raises(ValueError, match="parents differ"):
        verify_accepted_integration(
            root, preview, expected_preview_sha256=identity, accepted_sha=head,
            observed_main_sha=accepted,
        )
    wrong_tree = _git(root, "rev-parse", f"{base}^{{tree}}")
    forged = _git(root, "commit-tree", wrong_tree, "-p", base, "-p", head, "-m", "lost candidate files")
    with pytest.raises(ValueError, match="tree differs"):
        verify_accepted_integration(
            root, preview, expected_preview_sha256=identity, accepted_sha=forged,
            observed_main_sha=accepted,
        )
