"""Exact Git candidate freeze, integration preview and accepted-tree readback."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from twelve_six.ai_qa_control import _canonical_json_bytes, probe_candidate_parents
from twelve_six.physical_qualification import _bounded_git_env
from twelve_six.sil_qualification import require_exact_clean_git_state

_GIT_SHA = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_FREEZE_SCHEMA = "12-6.plan8-candidate-freeze.v1"
_PREVIEW_SCHEMA = "12-6.plan8-integration-preview.v1"
_MAIN_REFS = {"refs/heads/main", "refs/remotes/origin/main"}


def _require(value: object, pattern: re.Pattern[str], label: str) -> None:
    if type(value) is not str or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _identity(payload: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _paths(paths: tuple[str, ...]) -> None:
    if type(paths) is not tuple or any(type(path) is not str for path in paths):
        raise ValueError("integration paths must be a tuple of strings")
    if tuple(sorted(set(paths))) != paths:
        raise ValueError("integration paths must be unique and sorted")
    for path in paths:
        parsed = PurePosixPath(path)
        if (
            not parsed.parts
            or parsed.is_absolute()
            or parsed.as_posix() != path
            or ".." in parsed.parts
            or "\\" in path
            or any(ord(character) < 32 for character in path)
        ):
            raise ValueError("integration paths must be canonical repository-relative paths")


def _under(path: str, scope: str) -> bool:
    return path == scope or path.startswith(scope + "/")


def _git(
    root: Path, *arguments: str, index: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = _bounded_git_env()
    if index is not None:
        env["GIT_INDEX_FILE"] = str(index)
    return subprocess.run(
        ("git", *arguments), cwd=root, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=60, check=False,
    )


def _checked_git(root: Path, *arguments: str, index: Path | None = None) -> bytes:
    result = _git(root, *arguments, index=index)
    if result.returncode:
        raise ValueError(f"Git integration check failed: {arguments[0]}")
    return result.stdout


def _tree(root: Path, commit: str) -> str:
    _require(commit, _GIT_SHA, "Git commit SHA")
    if _checked_git(root, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise ValueError("integration identity does not name an exact commit")
    tree = _checked_git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip()
    _require(tree, _GIT_SHA, "Git tree SHA")
    return tree


def _require_ancestor(root: Path, ancestor: str, descendant: str) -> None:
    if _git(root, "merge-base", "--is-ancestor", ancestor, descendant).returncode:
        raise ValueError("Git topology does not contain the required ancestor")


def _changed(root: Path, before: str, after: str) -> tuple[str, ...]:
    raw = _checked_git(
        root, "diff", "--no-renames", "--no-ext-diff", "--no-textconv",
        "--ignore-submodules=none", "--name-only", "-z", before, after, "--",
    )
    if not raw:
        return ()
    if not raw.endswith(b"\0"):
        raise ValueError("Git path listing has no final delimiter")
    try:
        paths = tuple(sorted(part.decode("utf-8") for part in raw[:-1].split(b"\0")))
    except UnicodeError as exc:
        raise ValueError("Git path listing is not UTF-8") from exc
    _paths(paths)
    return paths


def _require_disjoint(changed: tuple[str, ...], scopes: tuple[str, ...]) -> None:
    for path in changed:
        if any(_under(path, scope) or _under(scope, path) for scope in scopes):
            raise ValueError(f"main changes overlap a declared integration path: {path}")


def _verify_checkout(root: Path, head: str) -> None:
    require_exact_clean_git_state(root, head)
    status = _checked_git(
        root, "-c", "core.fsmonitor=false", "status", "--porcelain=v1",
        "--untracked-files=all", "--ignore-submodules=none", "-z",
    )
    entries = _checked_git(root, "ls-files", "-v", "-z").split(b"\0")
    if status or any(entry[:1] == b"S" or entry[:1].islower() for entry in entries):
        raise ValueError("integration checkout is dirty or masks tracked paths")


def _require_main_tip(root: Path, main_ref: str, observed_sha: str) -> None:
    if type(main_ref) is not str or main_ref not in _MAIN_REFS:
        raise ValueError("integration requires a canonical main ref")
    _require(observed_sha, _GIT_SHA, "observed main SHA")
    actual = _checked_git(root, "show-ref", "--verify", "--hash", main_ref).decode().strip()
    if actual != observed_sha:
        raise ValueError("observed main SHA differs from the canonical main ref")


@dataclass(frozen=True, slots=True)
class CandidateFreeze:
    base_sha: str
    head_sha: str
    tree_sha: str
    owned_paths: tuple[str, ...]
    dependency_paths: tuple[str, ...]
    changed_paths: tuple[str, ...]
    exact_base_required: bool

    def __post_init__(self) -> None:
        for value in (self.base_sha, self.head_sha, self.tree_sha):
            _require(value, _GIT_SHA, "Git SHA")
        for paths in (self.owned_paths, self.dependency_paths, self.changed_paths):
            _paths(paths)
        if self.base_sha == self.head_sha or not self.changed_paths or type(self.exact_base_required) is not bool:
            raise ValueError("candidate must freeze an exact, changed Git tree and base scope")
        if any(not any(_under(path, scope) for scope in self.owned_paths) for path in self.changed_paths):
            raise ValueError("candidate changes a path outside its declared ownership")

    def to_dict(self) -> dict[str, object]:
        CandidateFreeze.__post_init__(self)
        return {
            "schema_version": _FREEZE_SCHEMA,
            "base_sha": self.base_sha, "head_sha": self.head_sha, "tree_sha": self.tree_sha,
            "owned_paths": list(self.owned_paths), "dependency_paths": list(self.dependency_paths),
            "changed_paths": list(self.changed_paths), "exact_base_required": self.exact_base_required,
        }


def load_candidate_freeze(document: dict[str, object], *, expected_sha256: str) -> CandidateFreeze:
    _require(expected_sha256, _DIGEST, "trusted freeze SHA-256")
    if type(document) is not dict or set(document) != {
        "schema_version", "base_sha", "head_sha", "tree_sha", "owned_paths",
        "dependency_paths", "changed_paths", "exact_base_required",
    } or document["schema_version"] != _FREEZE_SCHEMA:
        raise ValueError("invalid candidate freeze fields")
    if _identity(document) != expected_sha256:
        raise ValueError("candidate freeze differs from the trusted record")
    lists = ("owned_paths", "dependency_paths", "changed_paths")
    if any(type(document[name]) is not list for name in lists):
        raise ValueError("invalid candidate freeze path arrays")
    return CandidateFreeze(
        document["base_sha"], document["head_sha"], document["tree_sha"],
        *(tuple(document[name]) for name in lists), document["exact_base_required"],
    )


def freeze_candidate(
    repo_root: str | Path,
    *,
    base_sha: str,
    expected_head_sha: str,
    owned_paths: tuple[str, ...],
    dependency_paths: tuple[str, ...] = (),
    exact_base_required: bool = False,
) -> tuple[CandidateFreeze, str]:
    root = Path(repo_root)
    _verify_checkout(root, expected_head_sha)
    _tree(root, base_sha)
    tree = _tree(root, expected_head_sha)
    _require_ancestor(root, base_sha, expected_head_sha)
    freeze = CandidateFreeze(
        base_sha, expected_head_sha, tree, owned_paths, dependency_paths,
        _changed(root, base_sha, expected_head_sha), exact_base_required,
    )
    _verify_checkout(root, expected_head_sha)
    return freeze, _identity(freeze.to_dict())


def _verify_freeze(root: Path, freeze: CandidateFreeze, expected_sha256: str) -> None:
    if type(freeze) is not CandidateFreeze:
        raise ValueError("exact candidate freeze required")
    _require(expected_sha256, _DIGEST, "trusted freeze SHA-256")
    if _identity(freeze.to_dict()) != expected_sha256:
        raise ValueError("candidate freeze differs from the trusted record")
    _tree(root, freeze.base_sha)
    _require_ancestor(root, freeze.base_sha, freeze.head_sha)
    if (
        _tree(root, freeze.head_sha) != freeze.tree_sha
        or _changed(root, freeze.base_sha, freeze.head_sha) != freeze.changed_paths
    ):
        raise ValueError("Git objects differ from the frozen candidate")


def verify_frozen_checkout(
    repo_root: str | Path, freeze: CandidateFreeze, *, expected_sha256: str,
) -> None:
    root = Path(repo_root)
    _verify_freeze(root, freeze, expected_sha256)
    _verify_checkout(root, freeze.head_sha)


@dataclass(frozen=True, slots=True)
class IntegrationPreview:
    freeze: CandidateFreeze
    freeze_sha256: str
    qualification_sha256: str
    main_ref: str
    live_base_sha: str
    expected_tree_sha: str
    strategy: str

    def __post_init__(self) -> None:
        if type(self.freeze) is not CandidateFreeze:
            raise ValueError("integration preview needs an exact freeze")
        _require(self.freeze_sha256, _DIGEST, "trusted freeze SHA-256")
        if _identity(self.freeze.to_dict()) != self.freeze_sha256:
            raise ValueError("integration preview freeze identity mismatch")
        _require(self.qualification_sha256, _DIGEST, "qualification SHA-256")
        if type(self.main_ref) is not str or self.main_ref not in _MAIN_REFS:
            raise ValueError("integration requires a canonical main ref")
        _require(self.live_base_sha, _GIT_SHA, "live base SHA")
        _require(self.expected_tree_sha, _GIT_SHA, "predicted tree SHA")
        if type(self.strategy) is not str or self.strategy not in {"squash", "merge"}:
            raise ValueError("unsupported integration strategy")

    def to_dict(self) -> dict[str, object]:
        IntegrationPreview.__post_init__(self)
        return {
            "schema_version": _PREVIEW_SCHEMA, "freeze": self.freeze.to_dict(),
            "freeze_sha256": self.freeze_sha256, "qualification_sha256": self.qualification_sha256,
            "main_ref": self.main_ref, "live_base_sha": self.live_base_sha,
            "expected_tree_sha": self.expected_tree_sha,
            "strategy": self.strategy,
        }


def load_integration_preview(document: dict[str, object], *, expected_sha256: str) -> IntegrationPreview:
    _require(expected_sha256, _DIGEST, "trusted preview SHA-256")
    if type(document) is not dict or set(document) != {
        "schema_version", "freeze", "freeze_sha256", "qualification_sha256",
        "main_ref", "live_base_sha", "expected_tree_sha", "strategy",
    } or document["schema_version"] != _PREVIEW_SCHEMA:
        raise ValueError("invalid integration preview fields")
    if _identity(document) != expected_sha256:
        raise ValueError("integration preview differs from the trusted record")
    freeze = load_candidate_freeze(document["freeze"], expected_sha256=document["freeze_sha256"])
    return IntegrationPreview(
        freeze, document["freeze_sha256"], document["qualification_sha256"],
        document["main_ref"], document["live_base_sha"], document["expected_tree_sha"], document["strategy"],
    )


def _predict_tree(root: Path, freeze: CandidateFreeze, live_base_sha: str) -> str:
    _tree(root, live_base_sha)
    _require_ancestor(root, freeze.base_sha, live_base_sha)
    if live_base_sha == freeze.base_sha:
        return freeze.tree_sha
    if freeze.exact_base_required:
        raise ValueError("qualification requires the original exact base")
    _require_disjoint(
        _changed(root, freeze.base_sha, live_base_sha),
        (*freeze.owned_paths, *freeze.dependency_paths),
    )
    with TemporaryDirectory(prefix="integration-index-") as directory:
        index = Path(directory) / "index"
        _checked_git(root, "read-tree", "-i", "-m", freeze.base_sha, live_base_sha, freeze.head_sha, index=index)
        tree = _checked_git(root, "write-tree", index=index).decode().strip()
    _require(tree, _GIT_SHA, "predicted tree SHA")
    return tree


def prepare_integration(
    repo_root: str | Path,
    freeze: CandidateFreeze,
    *,
    expected_freeze_sha256: str,
    qualified_head_sha: str,
    qualified_tree_sha: str,
    qualification_sha256: str,
    observed_pr_head_sha: str,
    live_base_sha: str,
    mergeable: bool,
    main_ref: str = "refs/remotes/origin/main",
    strategy: str = "squash",
) -> tuple[IntegrationPreview, str]:
    """Prepare a guarded merge from independently qualified and observed facts."""
    root = Path(repo_root)
    _verify_freeze(root, freeze, expected_freeze_sha256)
    _require(qualified_head_sha, _GIT_SHA, "qualified head SHA")
    _require(qualified_tree_sha, _GIT_SHA, "qualified tree SHA")
    _require(observed_pr_head_sha, _GIT_SHA, "observed PR head SHA")
    if (qualified_head_sha, qualified_tree_sha) != (freeze.head_sha, freeze.tree_sha):
        raise ValueError("qualification does not match the frozen head and tree")
    if observed_pr_head_sha != freeze.head_sha or mergeable is not True:
        raise ValueError("observed PR head or mergeability does not match the freeze")
    _require_main_tip(root, main_ref, live_base_sha)
    preview = IntegrationPreview(
        freeze, expected_freeze_sha256, qualification_sha256, main_ref,
        live_base_sha, _predict_tree(root, freeze, live_base_sha), strategy,
    )
    _require_main_tip(root, main_ref, live_base_sha)
    return preview, _identity(preview.to_dict())


def verify_accepted_integration(
    repo_root: str | Path,
    preview: IntegrationPreview,
    *,
    expected_preview_sha256: str,
    accepted_sha: str,
    observed_main_sha: str,
) -> dict[str, object]:
    """Read back the accepted commit, tree and current main lineage."""
    if type(preview) is not IntegrationPreview:
        raise ValueError("exact integration preview required")
    _require(expected_preview_sha256, _DIGEST, "trusted preview SHA-256")
    if _identity(preview.to_dict()) != expected_preview_sha256:
        raise ValueError("integration preview differs from the trusted record")
    root = Path(repo_root)
    _require_main_tip(root, preview.main_ref, observed_main_sha)
    freeze = preview.freeze
    _verify_freeze(root, freeze, preview.freeze_sha256)
    if _predict_tree(root, freeze, preview.live_base_sha) != preview.expected_tree_sha:
        raise ValueError("integration preview tree does not match current Git objects")
    accepted_tree = _tree(root, accepted_sha)
    _tree(root, observed_main_sha)
    parents = (preview.live_base_sha,)
    if preview.strategy == "merge":
        parents += (freeze.head_sha,)
    if probe_candidate_parents(root, accepted_sha) != parents:
        raise ValueError("accepted commit parents differ from the integration preview")
    if accepted_tree != preview.expected_tree_sha:
        raise ValueError("accepted tree differs from the integration preview")
    _require_ancestor(root, accepted_sha, observed_main_sha)
    _require_disjoint(
        _changed(root, accepted_sha, observed_main_sha),
        (*freeze.owned_paths, *freeze.dependency_paths),
    )
    _require_main_tip(root, preview.main_ref, observed_main_sha)
    receipt = {
        "schema_version": "12-6.plan8-accepted-integration.v1",
        "preview_sha256": expected_preview_sha256, "accepted_sha": accepted_sha,
        "accepted_tree_sha": accepted_tree, "observed_main_sha": observed_main_sha,
        "main_ref": preview.main_ref,
        "expected_pr_head_sha": freeze.head_sha,
    }
    return {**receipt, "identity_sha256": _identity(receipt)}
