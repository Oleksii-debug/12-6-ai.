#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${GITHUB_TOKEN:?GITHUB_TOKEN is required for immutable Actions artifact retrieval}"

AUTHORITATIVE_MAIN="2da11c62f066671a1122888ec116242502f4fb9c"
EXPECTED_MATERIALIZER_BLOB="830087f91d1fa24385c5cbc8d2f687e4cc46b419"
EXPECTED_MATERIALIZER_V2_BLOB="a4de33f1e225ab8598197e4c1cba28480180d575"
EXPECTED_PRIVACY_BLOB="bcc5938395724f6728ab212f98b39f2334b0f37d"
EXPECTED_INPUT_SHA="3f60cfe55435daf53908c492be358f36d7ebbc2ebee532c921a69ba92b2f6b25"
EXPECTED_CURRENT_SHA="b1ec0433fbd9675645b7e29c1e32b406e638fa081868cad7a56afec8f9a601cc"
EXPECTED_MATERIALIZATION_ID="27fd5c5dac5739ea3a983359308222870619bb807fc1a51b681b9143087cebff"
EXPECTED_PREFLIGHT_ID="703cba4fab46c6dad12cc2c3f916a9c47aa6ed59ff0a5d7acff92a43eafd64e2"
REMOTE="https://github.com/Oleksii-debug/12-6-ai..git"

# Fail closed if the execution-critical Product implementation on this branch is not the
# exact implementation that produced the already-terminal 255-row retained corpus.
test "$(git hash-object src/twelve_six/data/post_g05_g06_materialization_v1.py)" = "$EXPECTED_MATERIALIZER_BLOB"
test "$(git hash-object src/twelve_six/data/post_g05_g06_materialization_v2.py)" = "$EXPECTED_MATERIALIZER_V2_BLOB"
test "$(git hash-object src/twelve_six/data/privacy_filter_v3.py)" = "$EXPECTED_PRIVACY_BLOB"

WORK="$(mktemp -d)"
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

mkdir -p \
  "$WORK/historical-a" "$WORK/historical-b" \
  "$WORK/bulk-a" "$WORK/bulk-b" \
  "$WORK/current-a" "$WORK/current-b" \
  "$WORK/post1247" "$WORK/post-g05-g06" \
  "$WORK/g05" "$WORK/g06" "$WORK/v8-a" "$WORK/v8-b" \
  artifacts/eval647

# Reconstruct from the same immutable historical source commits used by the terminal
# post-G05/G06 execution. These repositories are source history, not pretrained weights.
git clone --quiet --no-checkout --filter=blob:none "$REMOTE" "$WORK/survivor-product"
git -C "$WORK/survivor-product" fetch --quiet --depth=1 origin e3f0ead971091bd4a42316dfcc97aa1c3b116f66
git -C "$WORK/survivor-product" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$WORK/survivor-product" rev-parse HEAD)" = "e3f0ead971091bd4a42316dfcc97aa1c3b116f66"

git clone --quiet --no-checkout --filter=blob:none "$REMOTE" "$WORK/survivor-product/_data526_hist"
git -C "$WORK/survivor-product/_data526_hist" fetch --quiet --depth=1 origin 70d6ccc87396d129d00771bbf0b6b29bc673bfc4
git -C "$WORK/survivor-product/_data526_hist" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$WORK/survivor-product/_data526_hist" rev-parse HEAD)" = "70d6ccc87396d129d00771bbf0b6b29bc673bfc4"

git clone --quiet --no-checkout --filter=blob:none "$REMOTE" "$WORK/survivor-product/_data526_v7"
git -C "$WORK/survivor-product/_data526_v7" fetch --quiet --depth=1 origin d3333ec1b4a508df232a5aefccd6686adda745fb
git -C "$WORK/survivor-product/_data526_v7" checkout --quiet --detach FETCH_HEAD
test "$(git -C "$WORK/survivor-product/_data526_v7" rev-parse HEAD)" = "d3333ec1b4a508df232a5aefccd6686adda745fb"

python -m pip install --require-hashes --no-deps \
  -r "$WORK/survivor-product/_data526_hist/requirements/locks/linux-x86_64/dev.lock.txt"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends poppler-utils >/dev/null

fetch_artifact() {
  local artifact_id="$1"
  local destination="$2"
  curl --fail --silent --show-error --location --retry 5 --retry-all-errors \
    -H "Authorization: Bearer $GITHUB_TOKEN" \
    -H "Accept: application/vnd.github+json" \
    "https://api.github.com/repos/Oleksii-debug/12-6-ai./actions/artifacts/${artifact_id}/zip" \
    -o "$destination"
}

for build in a b; do
  fetch_artifact 9600107886 "$WORK/data213-${build}.zip"
  fetch_artifact 10032510626 "$WORK/v8-${build}.zip"
done
fetch_artifact 10292063266 "$WORK/g05.zip"
fetch_artifact 10297428350 "$WORK/g06.zip"

test "$(sha256sum "$WORK/data213-a.zip" | cut -d' ' -f1)" = "927e10ccb83f919d57a8b78e3ea9cae72f9d4e0232401043234a35eec150074d"
test "$(sha256sum "$WORK/data213-b.zip" | cut -d' ' -f1)" = "927e10ccb83f919d57a8b78e3ea9cae72f9d4e0232401043234a35eec150074d"
test "$(sha256sum "$WORK/v8-a.zip" | cut -d' ' -f1)" = "960c678fdbe6245658fa69c39edf355eb7b38ef39b7f8f833155c894dbd90385"
test "$(sha256sum "$WORK/v8-b.zip" | cut -d' ' -f1)" = "960c678fdbe6245658fa69c39edf355eb7b38ef39b7f8f833155c894dbd90385"
test "$(sha256sum "$WORK/g06.zip" | cut -d' ' -f1)" = "d9b8237a5f70bffe447185a299e9f7bbb9659149e6cae1d18bb157a235ac9993"
cmp "$WORK/data213-a.zip" "$WORK/data213-b.zip"
cmp "$WORK/v8-a.zip" "$WORK/v8-b.zip"

unzip -q "$WORK/v8-a.zip" -d "$WORK/v8-a"
unzip -q "$WORK/v8-b.zip" -d "$WORK/v8-b"
unzip -q "$WORK/g05.zip" -d "$WORK/g05"
unzip -q "$WORK/g06.zip" -d "$WORK/g06"
test "$(sha256sum "$WORK/g05/g05-a/authority.json" | cut -d' ' -f1)" = "e796e5150bfcc68ea47023a03636fffff3003970602cd2fd310181a14db71fda"
test "$(sha256sum "$WORK/g05/g05-b/authority.json" | cut -d' ' -f1)" = "e796e5150bfcc68ea47023a03636fffff3003970602cd2fd310181a14db71fda"
cmp "$WORK/g05/g05-a/authority.json" "$WORK/g05/g05-b/authority.json"
test "$(sha256sum "$WORK/g06/execution-a.json" | cut -d' ' -f1)" = "225d6cb0b8a02b06dd989fd7a8eb6d8b7e3f17f1302bab13ea2428f56df4cb40"

# Historical materializers fetch pinned public source payloads. Keep the previous fail-closed
# retry behavior: retry urllib, then curl the exact same URL; never substitute another source.
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
    command = [
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "5", "--retry-all-errors", "--connect-timeout", "30", "--max-time", "120",
    ]
    for key, value in request.header_items():
        command.extend(["-H", f"{key}: {value}"])
    command.append(request.full_url)
    completed = subprocess.run(command, check=True, capture_output=True)
    return _CurlResponse(completed.stdout)

urllib.request.urlopen = _hardened_urlopen
PY

export PYTHONPATH="$WORK/transport${PYTHONPATH:+:$PYTHONPATH}"

(
  cd "$WORK/survivor-product/_data526_hist"
  python tools/run_data526_materializer_data_only.py \
    --v7-root ../_data526_v7 \
    --data213-zip "$WORK/data213-a.zip" \
    --records-jsonl "$WORK/historical-a/records.jsonl" \
    --inventory-json "$WORK/historical-a/inventory.json" \
    --evidence-json "$WORK/historical-a/evidence.json"
  python tools/run_data526_materializer_data_only.py \
    --v7-root ../_data526_v7 \
    --data213-zip "$WORK/data213-b.zip" \
    --records-jsonl "$WORK/historical-b/records.jsonl" \
    --inventory-json "$WORK/historical-b/inventory.json" \
    --evidence-json "$WORK/historical-b/evidence.json"
)
cmp "$WORK/historical-a/records.jsonl" "$WORK/historical-b/records.jsonl"
test "$(sha256sum "$WORK/historical-a/records.jsonl" | cut -d' ' -f1)" = "2f21f7655c9287bbc7424b410788f431f20f0936f445fefd26fcf6114b772dbd"

(
  cd "$WORK/survivor-product"
  python tools/materialize_data_bulk_code1_permissive_python_bundle.py \
    --workspace "$WORK/bulk-a" --output "$WORK/bulk-a-report.json"
  python tools/materialize_data_bulk_code1_permissive_python_bundle.py \
    --workspace "$WORK/bulk-b" --output "$WORK/bulk-b-report.json"
)
cmp "$WORK/bulk-a-report.json" "$WORK/bulk-b-report.json"

(
  cd "$WORK/survivor-product"
  python tools/compose_data526_records_from_v8.py \
    --historical-records-jsonl "$WORK/historical-a/records.jsonl" \
    --bulk-report "$WORK/bulk-a-report.json" \
    --bulk-workspace "$WORK/bulk-a" \
    --v8-report "$WORK/v8-a/v8-a.json" \
    --v8-survivors "$WORK/v8-a/v8-survivors-a.json" \
    --records-jsonl "$WORK/current-a/records.jsonl" \
    --inventory-json "$WORK/current-a/inventory.json" \
    --evidence-json "$WORK/current-a/evidence.json" \
    --execution-head-sha b5bf78ce8e74e74876838b80294a0a7e1ee1281b
  python tools/compose_data526_records_from_v8.py \
    --historical-records-jsonl "$WORK/historical-b/records.jsonl" \
    --bulk-report "$WORK/bulk-b-report.json" \
    --bulk-workspace "$WORK/bulk-b" \
    --v8-report "$WORK/v8-b/v8-a.json" \
    --v8-survivors "$WORK/v8-b/v8-survivors-a.json" \
    --records-jsonl "$WORK/current-b/records.jsonl" \
    --inventory-json "$WORK/current-b/inventory.json" \
    --evidence-json "$WORK/current-b/evidence.json" \
    --execution-head-sha b5bf78ce8e74e74876838b80294a0a7e1ee1281b
)
cmp "$WORK/current-a/records.jsonl" "$WORK/current-b/records.jsonl"
test "$(sha256sum "$WORK/current-a/records.jsonl" | cut -d' ' -f1)" = "18a933b2e637b4cff497f4acfbff31041d4c0ba21fba397af53d213edfd7a1b7"

(
  cd "$WORK/survivor-product"
  python tools/materialize_post1247_data526_survivors_v1.py \
    --records-jsonl-a "$WORK/current-a/records.jsonl" \
    --records-jsonl-b "$WORK/current-b/records.jsonl" \
    --pre-materialization-evidence evidence/data526/v8/materialization_evidence.json \
    --decontamination-evidence evidence/data232/current_reserved_decontamination_v1_terminal.json \
    --survivor-records-jsonl "$WORK/post1247/survivors.jsonl" \
    --inventory-json "$WORK/post1247/record_inventory.json" \
    --evidence-json "$WORK/post1247/materialization_evidence.json" \
    --execution-head-sha b5bf78ce8e74e74876838b80294a0a7e1ee1281b
)
test "$(sha256sum "$WORK/post1247/survivors.jsonl" | cut -d' ' -f1)" = "$EXPECTED_INPUT_SHA"
test "$(wc -l < "$WORK/post1247/survivors.jsonl" | tr -d ' ')" = "272"

preflight_identity="$(
  python tools/materialize_d03_final_g05_g06_coverage_v1.py \
    --native-current-execution-preflight \
    --g05-authority "$WORK/g05/g05-a/authority.json" \
    --expected-g05-execution-identity 415e8b5aec57150facd5b8fc081d00a856033597ecf7607e94f1c10224999bfa \
    --g06-execution-envelope "$WORK/g06/execution-a.json" \
    --expected-g06-envelope-identity 696a8d264f1449b944fe8261a2c57ad856f6c44bbe14e29e2e2085f0e96a3451 \
    --expected-g06-execution-identity f0633cb40ae933774e161d001c097878fd1bd4399e6d7da9f44099610e5ae0cd \
    --g06-terminal-qualification evidence/data526/post1247/g06_terminal_qualification_receipt_v1.json \
    --expected-g06-terminal-qualification-identity c5f99dd4385a8400402a1b46abeea4bede58202a68521dbdce19cf702e6c4464 \
    --expected-input-rows-sha256 7abf25e62075b6e01f7bafa0892fd57d8bae393576782ac023d3a894f829f62c \
    --expected-survivor-evidence-identity 284d122a9afd4e5d4676d78a202d3001e5cf87438776022d3004d61fdf10e579 \
    --expected-survivor-jsonl-sha256 "$EXPECTED_INPUT_SHA" \
    --expected-survivor-record-inventory-sha256 90e306ce74a82016c835a5e106088f7bdb586e13010301b5002ce5d55e11e27c \
    --expected-survivor-payload-inventory-sha256 36e30427a8f6bc089c911690af82b1c59bfec1c883f3107597ca7fb07838b8df \
    --expected-survivor-record-count 272 \
    --expected-survivor-payload-bytes 5921963 \
    --expected-survivor-source-object-count 259 \
    --expected-privacy-policy-sha256 66f94332ecc6fd59727b8a840e458255b65566b4cc6ca3f149586ac4232eaf90 \
    --expected-privacy-implementation-git-blob-sha1 "$EXPECTED_PRIVACY_BLOB" \
    --output "$WORK/post1247/current-preflight.json"
)"
test "$preflight_identity" = "$EXPECTED_PREFLIGHT_ID"

python tools/materialize_post_g05_g06_payload_v1.py \
  --input-jsonl "$WORK/post1247/survivors.jsonl" \
  --composition-preflight "$WORK/post1247/current-preflight.json" \
  --expected-composition-preflight-identity "$EXPECTED_PREFLIGHT_ID" \
  --g05-authority "$WORK/g05/g05-a/authority.json" \
  --expected-g05-execution-identity 415e8b5aec57150facd5b8fc081d00a856033597ecf7607e94f1c10224999bfa \
  --g06-execution-envelope "$WORK/g06/execution-a.json" \
  --expected-g06-envelope-identity 696a8d264f1449b944fe8261a2c57ad856f6c44bbe14e29e2e2085f0e96a3451 \
  --expected-g06-execution-identity f0633cb40ae933774e161d001c097878fd1bd4399e6d7da9f44099610e5ae0cd \
  --g06-terminal-qualification evidence/data526/post1247/g06_terminal_qualification_receipt_v1.json \
  --expected-g06-terminal-qualification-identity c5f99dd4385a8400402a1b46abeea4bede58202a68521dbdce19cf702e6c4464 \
  --privacy-source src/twelve_six/data/privacy_filter_v3.py \
  --execution-head-sha "$AUTHORITATIVE_MAIN" \
  --expected-materializer-implementation-git-blob-sha1 "$EXPECTED_MATERIALIZER_BLOB" \
  --expected-materializer-v2-implementation-git-blob-sha1 "$EXPECTED_MATERIALIZER_V2_BLOB" \
  --output-jsonl "$WORK/post-g05-g06/records.jsonl" \
  --output-inventory "$WORK/post-g05-g06/inventory.json" \
  --output-evidence "$WORK/post-g05-g06/evidence.json"

test "$(sha256sum "$WORK/post-g05-g06/records.jsonl" | cut -d' ' -f1)" = "$EXPECTED_CURRENT_SHA"
test "$(wc -l < "$WORK/post-g05-g06/records.jsonl" | tr -d ' ')" = "255"
python - "$WORK/post-g05-g06/evidence.json" "$EXPECTED_MATERIALIZATION_ID" <<'PY'
import json
import sys
from pathlib import Path
p = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert p["materialization_identity_sha256"] == sys.argv[2]
assert p["status"] == "MATERIALIZED_ZERO_CREDIT"
assert p["result"]["record_count"] == 255
assert p["result"]["record_payload_jsonl_sha256"] == "b1ec0433fbd9675645b7e29c1e32b406e638fa081868cad7a56afec8f9a601cc"
assert p["truth_boundary"]["authorized_optimized_target_exposure"] == 0
assert p["truth_boundary"]["training_executed"] is False
PY

python tools/execute_eval647_current_corpus_overlap_v1.py \
  --training-records-jsonl "$WORK/post-g05-g06/records.jsonl" \
  --materialization-evidence-json "$WORK/post-g05-g06/evidence.json" \
  --manifest configs/evaluation/eval_code_reserve_v1.json \
  --output artifacts/eval647/current_corpus_overlap_v1.json \
  --expected-materialization-identity-sha256 "$EXPECTED_MATERIALIZATION_ID" \
  --expected-training-jsonl-sha256 "$EXPECTED_CURRENT_SHA" \
  --timeout 60

python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("artifacts/eval647/current_corpus_overlap_v1.json").read_text(encoding="utf-8"))
assert p["status"] == "PASS_CURRENT_CORPUS_OVERLAP_ONLY"
assert p["current_corpus_overlap_zero"] is True
assert p["match_evidence_count"] == 0
assert p["selection_validation_records_authorized"] == 0
assert p["current_corpus"]["record_count"] == 255
assert p["current_corpus"]["record_payload_jsonl_sha256"] == "b1ec0433fbd9675645b7e29c1e32b406e638fa081868cad7a56afec8f9a601cc"
print(json.dumps(p, sort_keys=True))
PY