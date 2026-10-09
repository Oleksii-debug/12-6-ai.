"""Plan-2 S15: independently re-execute the existing audit from exact Git archives.

This checks source-tree reproducibility, NOT production corpus admissibility.
Only Git-tracked bytes enter the two isolated source trees. Neither result can
turn candidate-only rights, synthetic tokenizers, or fixtures into a release.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "12-6.plan2-s15-git-clean-checkout.v1"
OUTPUT = "plan2-s15-git-checkout-evidence.json"


class GitCheckoutDenied(ValueError):
    """A non-reproducible, unsafe, or unqualified source checkout."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise GitCheckoutDenied(reason)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=60,
    )
    need(proc.returncode == 0, "Git source checkout or inventory unavailable")
    return proc.stdout


def _source(root: Path) -> tuple[str, str, bytes]:
    need(root.is_dir() and not root.is_symlink(), "invalid checkout root")
    top = Path(_git(root, "rev-parse", "--show-toplevel").decode().strip())
    need(top.resolve() == root.resolve(), "source root is not Git checkout root")
    need(not _git(root, "status", "--porcelain", "--untracked-files=no"),
         "tracked working tree is dirty; HEAD would not match source")
    head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
    tree = _git(root, "rev-parse", "HEAD^{tree}").decode("ascii").strip()
    need(len(head) == 40 and len(tree) == 40, "invalid Git source identity")
    archive = _git(root, "archive", "--format=tar", "HEAD")
    need(bool(archive), "empty Git archive")
    return head, tree, archive


def _extract(archive: bytes, directory: Path) -> int:
    """Fail closed on archive traversal, links, duplicate paths and special files."""
    seen: set[str] = set()
    file_count = 0
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tar:
        for member in tar:
            raw_name = member.name.rstrip("/") if member.isdir() else member.name
            parts = raw_name.split("/")
            need(
                bool(raw_name) and not raw_name.startswith("/")
                and "\\" not in raw_name
                and all(part not in ("", ".", "..") for part in parts),
                "unsafe Git archive member path",
            )
            need(raw_name not in seen, "duplicate Git archive member")
            seen.add(raw_name)
            target = directory.joinpath(*parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                need(not target.exists() and not target.is_symlink(),
                     "archive member would overwrite source")
                stream = tar.extractfile(member)
                need(stream is not None, "missing archive file stream")
                with stream, target.open("xb") as output:
                    shutil.copyfileobj(stream, output)
                file_count += 1
            else:
                raise GitCheckoutDenied("Git archive contains link or special member")
    need(file_count > 0, "no tracked checkout files")
    return file_count


def verify(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    head, tree, archive = _source(root)
    reports: list[bytes] = []
    with tempfile.TemporaryDirectory(prefix="plan2-s15-git-archive-") as workspace:
        for i in (1, 2):
            checkout = Path(workspace) / f"source-{i}"
            checkout.mkdir()
            members = _extract(archive, checkout)
            need(members > 0, "empty clean checkout")
            evidence_dir = Path(workspace) / f"evidence-{i}"
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join((str(checkout / "src"), str(checkout)))
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            proc = subprocess.run(
                [sys.executable, "-m", "tools.plan2_terminal_qualification_v1",
                 "--root", str(checkout), "--out-dir", str(evidence_dir)],
                cwd=checkout, env=env, check=False,
                capture_output=True, timeout=600,
            )
            need(proc.returncode == 0, "existing S15 audit failed in clean Git checkout")
            report_path = evidence_dir / "plan2-section15-audit.json"
            need(report_path.is_file() and not report_path.is_symlink(),
                 "missing clean-checkout audit receipt")
            raw = report_path.read_bytes()
            try:
                proof = json.loads(raw.decode("utf-8"))
                logged = json.loads(proc.stdout.decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                raise GitCheckoutDenied("invalid clean-checkout receipt") from exc
            need(raw == canonical(proof), "noncanonical checkout evidence")
            unsigned = {k: v for k, v in proof.items() if k != "audit_sha256"}
            need(proof.get("audit_sha256") == sha(canonical(unsigned))
                 and proof.get("terminal_done") is False
                 and proof.get("production_release_authorized") is False
                 and logged.get("audit_sha256") == proof["audit_sha256"]
                 and logged.get("terminal_done") is False,
                 "unexpected release or broken audit evidence")
            reports.append(raw)
    need(reports[0] == reports[1], "independent Git checkout rebuilds diverged")
    identity = {
        "schema_version": SCHEMA,
        "decision": "CLEAN_GIT_CHECKOUT_REPRODUCED_NOT_PRODUCTION_RELEASE",
        "git_head": head,
        "git_tree": tree,
        "git_archive_sha256": sha(archive),
        "independent_source_checkouts": 2,
        "audit_sha256": sha(reports[0]),
        "terminal_done": False,
        "production_release_authorized": False,
    }
    return {**identity, "evidence_sha256": sha(canonical(identity))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(path.is_symlink() for path in (destination, *destination.parents)),
         "symlink publication path")
    evidence = verify(root)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / OUTPUT
    raw = canonical(evidence)
    if path.exists() or path.is_symlink():
        need(path.is_file() and not path.is_symlink() and path.read_bytes() == raw,
             "immutable Git-checkout evidence changed")
    else:
        with path.open("xb") as stream:
            stream.write(raw)
    need(path.read_bytes() == raw, "Git checkout evidence readback mismatch")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan2 S15 reproducibility from Git HEAD")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({"decision": report["decision"],
                      "evidence_sha256": report["evidence_sha256"],
                      "terminal_done": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
