"""Fail-closed live code-evaluation authority refresh for NEXT100-054."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

AUTHORITY_BRANCH = "next100-057/code-selection-validation-v2-20260826"
AUTHORITY_HEAD = "6713fe972b875b8a516122bda347264fb4099b2b"
AUTHORITY_EVIDENCE_PATH = "evidence/next100-057/code-selection-validation-v2.json"
AUTHORITY_EVIDENCE_BLOB_SHA1 = "95fb3ac2c7505d1451575d3d7a599a9f3a65067c"
AUTHORITY_IDENTITY_SHA256 = "08a5876d24d054e94171eeaebb3610e3992b39bed5b038550148348e621ac41c"
AUTHORITY_CONTRACT_BLOB_SHA1 = "622e193a35ba485ca162d9db736823f97303b7ff"
REPORT_NAME = "next100-054-live-code-eval-authority.json"
MANIFEST_PATH = Path("configs/data/next100_054_urllib3_code_rights_v1.json")


class LiveAuthorityError(RuntimeError):
    pass


def _cjson(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _download(url: str, *, max_bytes: int = 50_000) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "12-6-NEXT100-054-live-authority/1"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise LiveAuthorityError(f"bounded download exceeded {max_bytes} bytes: {url}")
    return data


def _download_json(url: str, *, max_bytes: int = 50_000) -> dict[str, Any]:
    return json.loads(_download(url, max_bytes=max_bytes))


def refresh(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest = json.loads((repo_root / MANIFEST_PATH).read_text(encoding="utf-8"))
    selected = manifest["selected_files"]
    selected_blobs = sorted(item["blob_sha1"] for item in selected)
    if len(selected_blobs) != 8 or len(set(selected_blobs)) != 8:
        raise LiveAuthorityError("urllib3 selected blob inventory is not exactly eight unique blobs")

    encoded_branch = urllib.parse.quote(AUTHORITY_BRANCH, safe="")
    commits = _download_json(
        "https://api.github.com/repos/Oleksii-debug/12-6-ai./commits"
        f"?sha={encoded_branch}&per_page=1"
    )
    if not isinstance(commits, list) or not commits:
        raise LiveAuthorityError("could not resolve live code-eval authority branch")
    live_head = commits[0].get("sha")
    if live_head != AUTHORITY_HEAD:
        raise LiveAuthorityError(
            f"live code-eval authority moved: {live_head} != {AUTHORITY_HEAD}; refresh required"
        )

    raw_url = (
        "https://raw.githubusercontent.com/Oleksii-debug/12-6-ai./"
        f"{AUTHORITY_HEAD}/{AUTHORITY_EVIDENCE_PATH}"
    )
    raw = _download(raw_url, max_bytes=20_000)
    if _git_blob_sha1(raw) != AUTHORITY_EVIDENCE_BLOB_SHA1:
        raise LiveAuthorityError("live code-eval evidence Git blob identity drift")
    evidence = json.loads(raw)

    if evidence.get("authority_identity_sha256") != AUTHORITY_IDENTITY_SHA256:
        raise LiveAuthorityError("live code-eval authority identity drift")
    if evidence.get("contract", {}).get("git_blob_sha1") != AUTHORITY_CONTRACT_BLOB_SHA1:
        raise LiveAuthorityError("live code-eval contract identity drift")
    if evidence.get("decision") != "BLOCKED":
        raise LiveAuthorityError("code selection authority is no longer blocked")
    if evidence.get("eligibility", {}).get("eligible_object_count") != 0:
        raise LiveAuthorityError("eligible evaluation code objects now exist")
    selection = evidence.get("selection", {})
    if selection.get("record_count") != 0 or selection.get("records") != []:
        raise LiveAuthorityError("code selection reservation is no longer empty")
    if selection.get("jsonl", {}).get("published") is not False:
        raise LiveAuthorityError("code evaluation selection JSONL is now published")
    if evidence.get("terminal_for_observed_authority_vector") is not True:
        raise LiveAuthorityError("latest code selection authority is not terminal for its vector")
    if evidence.get("status") != "BLOCKED_NO_PRISTINE_CODE_OBJECTS_WITH_EXPLICIT_EVALUATION_RESERVATION":
        raise LiveAuthorityError("unexpected latest code selection status")

    reservations = evidence.get("live_authority_findings", {}).get(
        "purpose_specific_code_evaluation_reservation_branches"
    )
    if reservations != ["eval289/code-evaluation-rights-reservation-20260826"]:
        raise LiveAuthorityError("purpose-specific code evaluation reservation branch set drift")
    eval289 = evidence.get("live_authority_findings", {}).get("eval289", {})
    if eval289.get("reservation_active") is not False or eval289.get("eligible_object_count") != 0:
        raise LiveAuthorityError("EVAL-289 is no longer an inactive empty reservation")

    report: dict[str, Any] = {
        "schema_version": "12-6.next100-054-live-code-eval-refresh.v1",
        "worker_id": "NEXT100-054-CODE-URLLIB3",
        "authority": "NEXT100-057-CODE-EVAL-SET-V2",
        "authority_branch": AUTHORITY_BRANCH,
        "authority_head": live_head,
        "authority_identity_sha256": AUTHORITY_IDENTITY_SHA256,
        "authority_evidence_path": AUTHORITY_EVIDENCE_PATH,
        "authority_evidence_blob_sha1": AUTHORITY_EVIDENCE_BLOB_SHA1,
        "authority_evidence_raw_sha256": _sha256(raw),
        "decision": "PASS_EMPTY_CODE_EVALUATION_SELECTION",
        "eligible_evaluation_object_count": 0,
        "selected_evaluation_record_count": 0,
        "selected_urllib3_blob_sha1": selected_blobs,
        "selected_urllib3_evaluation_collision_count": 0,
        "candidate_role": "TRAINING_ONLY",
        "evaluation_use_authorized": False,
        "execution_class": "LOCAL_FREE",
        "training_performed": False,
        "paid_compute_used": False,
    }
    report["report_identity_sha256"] = _sha256(_cjson(report))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_NAME).write_bytes(_cjson(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = refresh(args.repo_root.resolve(), args.output_dir)
    print(json.dumps({
        "authority_head": report["authority_head"],
        "collision_count": report["selected_urllib3_evaluation_collision_count"],
        "decision": report["decision"],
        "report_identity_sha256": report["report_identity_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
