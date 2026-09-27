#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${GITHUB_TOKEN:?GITHUB_TOKEN is required for immutable Actions artifact retrieval}"

EXPECTED_MATERIALIZER_BLOB="830087f91d1fa24385c5cbc8d2f687e4cc46b419"
EXPECTED_MATERIALIZER_V2_BLOB="a4de33f1e225ab8598197e4c1cba28480180d575"
EXPECTED_PRIVACY_BLOB="bcc5938395724f6728ab212f98b39f2334b0f37d"
PHYSICAL_RUN_ID="36026689718"
PHYSICAL_ARTIFACT_ID="10820342689"
PHYSICAL_CARRIER_HEAD="6af889c7c3d15d38f463791c9e2ba56c8e936e16"
PHYSICAL_ARTIFACT_NAME="clean-post-g05-g06-${PHYSICAL_CARRIER_HEAD}"
EXPECTED_ARTIFACT_ZIP_SHA="99069ce2183abbbc374749cca5c538efa259df0c64658a9b88cc96b25c0fbba0"
EXPECTED_CURRENT_SHA="bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8"
EXPECTED_INVENTORY_FILE_SHA="3804a43eba5e0bfa6ce2568782cb03bf681e53a68808742874cf89139488c69e"
EXPECTED_EVIDENCE_FILE_SHA="877b6233738e0ca6593acbbf9922f5f178ec8456bf101b95f2f4a300805509e4"
EXPECTED_PROOF_FILE_SHA="16260584c5c2bb7d8eab7a16e1b5260300e9cd2b67e9110282287e619013078f"
EXPECTED_RECORD_INVENTORY_SHA="dbdf741884ec1f147827647908b17a846584b145454e3f82fdb63120422c6059"
EXPECTED_PAYLOAD_INVENTORY_SHA="2384480c89c19b14d188aa57130ee2967463512bb54241f90b6f17a525653a1e"
EXPECTED_MATERIALIZATION_ID="7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
EXPECTED_MATERIALIZATION_SCHEMA="12-6.d03-post-g05-g06-materialization.v2"
EXPECTED_RECORD_COUNT="257"
EXPECTED_PAYLOAD_BYTES="5601716"
EXPECTED_DISTINCT_SOURCE_IDS="244"
export EXPECTED_MATERIALIZER_BLOB EXPECTED_MATERIALIZER_V2_BLOB EXPECTED_PRIVACY_BLOB
export PHYSICAL_RUN_ID PHYSICAL_ARTIFACT_ID PHYSICAL_CARRIER_HEAD PHYSICAL_ARTIFACT_NAME
export EXPECTED_ARTIFACT_ZIP_SHA EXPECTED_CURRENT_SHA EXPECTED_RECORD_INVENTORY_SHA
export EXPECTED_PAYLOAD_INVENTORY_SHA EXPECTED_MATERIALIZATION_ID EXPECTED_MATERIALIZATION_SCHEMA
export EXPECTED_RECORD_COUNT EXPECTED_PAYLOAD_BYTES EXPECTED_DISTINCT_SOURCE_IDS

# Preserve the implementation binding established by #2060. This carrier now consumes the
# independently qualified physical v2 output rather than re-running historical dirty inputs.
test "$(git hash-object src/twelve_six/data/post_g05_g06_materialization_v1.py)" = "$EXPECTED_MATERIALIZER_BLOB"
test "$(git hash-object src/twelve_six/data/post_g05_g06_materialization_v2.py)" = "$EXPECTED_MATERIALIZER_V2_BLOB"
test "$(git hash-object src/twelve_six/data/privacy_filter_v3.py)" = "$EXPECTED_PRIVACY_BLOB"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/physical" artifacts/eval647

api_get() {
  curl --fail --silent --show-error --location --retry 5 --retry-all-errors \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    -H "X-GitHub-Api-Version: 2022-11-28" "$1" -o "$2"
}

api_get "https://api.github.com/repos/Oleksii-debug/12-6-ai./actions/runs/${PHYSICAL_RUN_ID}" "$WORK/run.json"
api_get "https://api.github.com/repos/Oleksii-debug/12-6-ai./actions/artifacts/${PHYSICAL_ARTIFACT_ID}" "$WORK/artifact.json"
python - "$WORK/run.json" "$WORK/artifact.json" <<'PY'
import json
import os
import sys
from pathlib import Path

run = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
artifact = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
run_id = int(os.environ["PHYSICAL_RUN_ID"])
artifact_id = int(os.environ["PHYSICAL_ARTIFACT_ID"])
head = os.environ["PHYSICAL_CARRIER_HEAD"]
name = os.environ["PHYSICAL_ARTIFACT_NAME"]
zip_sha = os.environ["EXPECTED_ARTIFACT_ZIP_SHA"]
assert run["id"] == run_id
assert run["status"] == "completed" and run["conclusion"] == "success"
assert run["head_sha"] == head
assert run["head_branch"] == "exec/2049-clean-post-g05-g06-physical-v1"
assert artifact["id"] == artifact_id and artifact["name"] == name
assert artifact["expired"] is False
assert artifact["digest"] == f"sha256:{zip_sha}"
assert artifact["size_in_bytes"] == 1644141
workflow_run = artifact["workflow_run"]
assert workflow_run["id"] == run_id
assert workflow_run["head_sha"] == head
assert workflow_run["head_branch"] == "exec/2049-clean-post-g05-g06-physical-v1"
PY

api_get "https://api.github.com/repos/Oleksii-debug/12-6-ai./actions/artifacts/${PHYSICAL_ARTIFACT_ID}/zip" "$WORK/physical.zip"
test "$(sha256sum "$WORK/physical.zip" | cut -d' ' -f1)" = "$EXPECTED_ARTIFACT_ZIP_SHA"
unzip -q "$WORK/physical.zip" -d "$WORK/physical"
test "$(sha256sum "$WORK/physical/records.jsonl" | cut -d' ' -f1)" = "$EXPECTED_CURRENT_SHA"
test "$(sha256sum "$WORK/physical/inventory.json" | cut -d' ' -f1)" = "$EXPECTED_INVENTORY_FILE_SHA"
test "$(sha256sum "$WORK/physical/evidence.json" | cut -d' ' -f1)" = "$EXPECTED_EVIDENCE_FILE_SHA"
test "$(sha256sum "$WORK/physical/proof.json" | cut -d' ' -f1)" = "$EXPECTED_PROOF_FILE_SHA"
test "$(wc -l < "$WORK/physical/records.jsonl" | tr -d ' ')" = "$EXPECTED_RECORD_COUNT"

python - "$WORK/physical" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

root = Path(sys.argv[1])
expected_jsonl = os.environ["EXPECTED_CURRENT_SHA"]
expected_record_inventory = os.environ["EXPECTED_RECORD_INVENTORY_SHA"]
expected_payload_inventory = os.environ["EXPECTED_PAYLOAD_INVENTORY_SHA"]
expected_materialization = os.environ["EXPECTED_MATERIALIZATION_ID"]
expected_schema = os.environ["EXPECTED_MATERIALIZATION_SCHEMA"]
expected_v1_blob = os.environ["EXPECTED_MATERIALIZER_BLOB"]
expected_v2_blob = os.environ["EXPECTED_MATERIALIZER_V2_BLOB"]
expected_privacy_blob = os.environ["EXPECTED_PRIVACY_BLOB"]
record_count = int(os.environ["EXPECTED_RECORD_COUNT"])
payload_bytes = int(os.environ["EXPECTED_PAYLOAD_BYTES"])
distinct_sources = int(os.environ["EXPECTED_DISTINCT_SOURCE_IDS"])

assert {p.name for p in root.iterdir()} == {"evidence.json", "inventory.json", "proof.json", "records.jsonl"}
assert all(p.is_file() for p in root.iterdir())


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        assert key not in value, f"duplicate JSON key: {key}"
        value[key] = item
    return value


def load(path: str) -> dict[str, Any]:
    value = json.loads((root / path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    assert isinstance(value, dict)
    return value


def keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    assert set(value) == expected, f"{label} schema drift"


def exact_int(value: Any, expected: int, label: str) -> None:
    assert type(value) is int and value == expected, f"{label} drift"


raw = (root / "records.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == expected_jsonl
rows: list[dict[str, Any]] = []
for number, line in enumerate(raw.decode("utf-8", errors="strict").splitlines(), start=1):
    assert line.strip(), f"blank row {number}"
    row = json.loads(line, object_pairs_hook=no_duplicates)
    assert isinstance(row, dict)
    keys(row, {"record_id", "source_id", "family", "modality", "normalized_payload"}, f"row {number}")
    assert all(isinstance(row[k], str) and row[k] for k in row)
    rows.append(row)
assert len(rows) == record_count
assert len({row["record_id"] for row in rows}) == record_count
assert len({row["source_id"] for row in rows}) == distinct_sources
assert sum(len(row["normalized_payload"].encode("utf-8")) for row in rows) == payload_bytes

inventory = load("inventory.json")
keys(inventory, {"schema_version", "record_count", "total_payload_bytes", "record_inventory_digest_sha256", "payload_inventory_digest_sha256", "records"}, "inventory")
assert inventory["schema_version"] == "12-6.data526-record-inventory.v1"
exact_int(inventory["record_count"], record_count, "inventory.record_count")
exact_int(inventory["total_payload_bytes"], payload_bytes, "inventory.total_payload_bytes")
assert inventory["record_inventory_digest_sha256"] == expected_record_inventory
assert inventory["payload_inventory_digest_sha256"] == expected_payload_inventory
assert isinstance(inventory["records"], list) and len(inventory["records"]) == record_count

evidence = load("evidence.json")
keys(evidence, {"schema_version", "status", "execution_profile", "execution_head_sha", "input", "base_v1_materialization_identity_sha256", "materializer_implementation_git_blob_sha1", "materializer_v2_implementation_git_blob_sha1", "provenance_guard", "consumed_blockers", "remaining_materialization_blockers", "repeat_materialization_byte_identical", "transform", "result", "truth_boundary", "materialization_identity_sha256"}, "evidence")
assert evidence["schema_version"] == expected_schema
assert evidence["status"] == "MATERIALIZED_ZERO_CREDIT"
assert evidence["execution_profile"] == "LOCAL_FREE"
assert evidence["execution_head_sha"] == "4588b660fd7650d6ddb072e9a9b666f5cc97238c"
assert evidence["materializer_implementation_git_blob_sha1"] == expected_v1_blob
assert evidence["materializer_v2_implementation_git_blob_sha1"] == expected_v2_blob
assert evidence["materialization_identity_sha256"] == expected_materialization
assert evidence["repeat_materialization_byte_identical"] is True
result = evidence["result"]
keys(result, {"record_count", "source_object_count", "total_payload_bytes", "record_payload_jsonl_sha256", "record_inventory_digest_sha256", "payload_inventory_digest_sha256"}, "evidence.result")
exact_int(result["record_count"], record_count, "result.record_count")
exact_int(result["source_object_count"], distinct_sources, "result.source_object_count")
exact_int(result["total_payload_bytes"], payload_bytes, "result.total_payload_bytes")
assert result["record_payload_jsonl_sha256"] == expected_jsonl
assert result["record_inventory_digest_sha256"] == expected_record_inventory
assert result["payload_inventory_digest_sha256"] == expected_payload_inventory
source = evidence["input"]
exact_int(source["record_count"], 274, "input.record_count")
exact_int(source["source_object_count"], 261, "input.source_object_count")
exact_int(source["total_payload_bytes"], 6093662, "input.total_payload_bytes")
assert source["record_payload_jsonl_sha256"] == "dc22d829921890ea8c5b51cbedaae099688c37c3cb624a597973305c7fa5b2c3"
assert source["record_inventory_digest_sha256"] == "7d6782e91243505c01b0f2f6d6f85b5bbe3a6abf628d73721b1e0c1c77e4f352"
assert source["payload_inventory_digest_sha256"] == "59f9c5a7b5db9e45fcde2e3bab10a4adc97cb6832f906fc241edc3e35248b576"
assert source["composition_preflight_identity_sha256"] == "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95"
assert source["privacy_implementation_git_blob_sha1"] == expected_privacy_blob
truth = evidence["truth_boundary"]
for key in ("current_corpus_eligible", "tokenizer_fit_authorized", "training_executed", "learned_weights_created", "paid_compute_used", "foreign_pretrained_weights", "final_test_outcomes_read"):
    assert truth[key] is False
for key in ("authorized_optimized_target_exposure", "authorized_unique_loss_positions", "optimizer_updates_executed_on_real_targets", "training_authorized_bytes"):
    exact_int(truth[key], 0, f"truth.{key}")
assert evidence["remaining_materialization_blockers"] == ["SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"]
assert evidence["provenance_guard"]["known_external_llm_contamination_absent"] is True
assert evidence["provenance_guard"]["whole_corpus_external_llm_cleanliness_claimed"] is False

proof = load("proof.json")
assert proof["schema"] == "12-6.d03-clean-post-g05-g06-two-run-physical.v1"
assert proof["execution_profile"] == "LOCAL_FREE"
assert proof["execution_carrier_head_sha"] == os.environ["PHYSICAL_CARRIER_HEAD"]
assert proof["materializer_v1_git_blob_sha1"] == expected_v1_blob
assert proof["materializer_v2_git_blob_sha1"] == expected_v2_blob
assert proof["privacy_implementation_git_blob_sha1"] == expected_privacy_blob
assert proof["composition_preflight_identity_sha256"] == "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95"
exact_int(proof["output_record_count"], record_count, "proof.output_record_count")
exact_int(proof["output_payload_bytes"], payload_bytes, "proof.output_payload_bytes")
assert proof["output_jsonl_sha256"] == expected_jsonl
assert proof["output_record_inventory_digest_sha256"] == expected_record_inventory
assert proof["output_payload_inventory_digest_sha256"] == expected_payload_inventory
assert proof["materialization_identity_sha256"] == expected_materialization
assert proof["two_fresh_cli_processes"] is True
assert proof["payload_byte_identical"] is True
assert proof["inventory_byte_identical"] is True
assert proof["evidence_byte_identical"] is True
assert proof["known_nomis_pr462_payload_absent"] is True
assert proof["privacy_redactions_rescanned_allow"] is True
assert proof["truth_boundary"]["training_executed"] is False
assert proof["truth_boundary"]["final_test_outcomes_read"] is False
PY

# Retain the incumbent retry/fallback behavior for the two pinned EVAL-647 source fetches.
mkdir -p "$WORK/transport"
cat > "$WORK/transport/sitecustomize.py" <<'PY'
import io
import ssl
import subprocess
import time
import urllib.error
import urllib.request

_original_urlopen = urllib.request.urlopen


class _CurlResponse:
    def __init__(self, payload: bytes) -> None:
        self._stream = io.BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stream.close()


def _hardened_urlopen(request, *args, **kwargs):
    last_error = None
    for attempt in range(4):
        try:
            return _original_urlopen(request, *args, **kwargs)
        except (urllib.error.URLError, ssl.SSLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1 + attempt)
    if not hasattr(request, "full_url"):
        raise last_error
    command = ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "120"]
    for key, value in request.header_items():
        command.extend(["-H", f"{key}: {value}"])
    command.append(request.full_url)
    completed = subprocess.run(command, check=True, capture_output=True)
    return _CurlResponse(completed.stdout)


urllib.request.urlopen = _hardened_urlopen
PY
export PYTHONPATH="$WORK/transport${PYTHONPATH:+:$PYTHONPATH}"

python tools/execute_eval647_current_corpus_overlap_v1.py \
  --training-records-jsonl "$WORK/physical/records.jsonl" \
  --materialization-evidence-json "$WORK/physical/evidence.json" \
  --manifest configs/evaluation/eval_code_reserve_v1.json \
  --output artifacts/eval647/current_corpus_overlap_v1.json \
  --expected-materialization-identity-sha256 "$EXPECTED_MATERIALIZATION_ID" \
  --expected-training-jsonl-sha256 "$EXPECTED_CURRENT_SHA" \
  --timeout 60

python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("artifacts/eval647/current_corpus_overlap_v1.json").read_text(encoding="utf-8"))
assert report["status"] == "PASS_CURRENT_CORPUS_OVERLAP_ONLY"
assert report["current_corpus_overlap_zero"] is True
assert report["match_evidence_count"] == 0
assert report["selection_validation_records_authorized"] == 0
assert report["current_corpus"]["record_count"] == 257
assert report["current_corpus"]["record_payload_jsonl_sha256"] == "bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8"
assert report["current_corpus"]["materialization_identity_sha256"] == "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
print(json.dumps(report, sort_keys=True))
PY
