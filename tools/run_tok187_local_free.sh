#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
OUT_ROOT="${2:-tok187-local-free}"
cd "$REPO_ROOT"

SOURCE_SHA="$(git rev-parse HEAD)"
PYTHON="${PYTHON:-python}"

if [[ "$($PYTHON -c 'import platform; print(platform.python_version())')" != "3.11.16" ]]; then
  echo "TOK-187 requires CPython 3.11.16" >&2
  exit 2
fi

rm -rf "$OUT_ROOT"
mkdir -p "$OUT_ROOT"

TOKENIZER_VENV="$OUT_ROOT/tokenizer-env"
ENV_MANIFEST="$OUT_ROOT/tokenizer-environment.json"

$PYTHON tools/execution_bootstrap.py bootstrap \
  --repo-root . \
  --capabilities runtime,tokenizer,tests,lint \
  --command "python -m pytest" \
  --command "python -m ruff" \
  --venv "$TOKENIZER_VENV" \
  --manifest "$ENV_MANIFEST"

TOK_PY="$TOKENIZER_VENV/bin/python"
"$TOK_PY" -m pip check
"$TOK_PY" -m compileall -q \
  src/twelve_six/tok187_bpe_real.py \
  src/twelve_six/tokenization/experiments.py \
  src/twelve_six/vocabulary.py \
  src/twelve_six/research_decision.py
"$TOK_PY" -m ruff check --select E4,E7,E9,F \
  src/twelve_six/tok187_bpe_real.py \
  src/twelve_six/tokenization/experiments.py \
  src/twelve_six/vocabulary.py \
  src/twelve_six/research_decision.py \
  tests/test_tok187_bpe_real.py
"$TOK_PY" -m pytest -q tests/test_tok187_bpe_real.py

# DATA-183 still has a separate, pre-ENV-151 exact intake runtime because the
# universal capability registry intentionally has no canonical DataTrove lock.
DATA_VENV="$OUT_ROOT/data183-env"
$PYTHON -m venv "$DATA_VENV"
DATA_PY="$DATA_VENV/bin/python"
"$DATA_PY" -m pip install \
  --disable-pip-version-check --no-deps --require-hashes \
  -r requirements/locks/linux-x86_64/toolchain.lock.txt
"$DATA_PY" -m pip install \
  --disable-pip-version-check --no-deps --require-hashes \
  -r requirements/locks/linux-x86_64/runtime.lock.txt
"$DATA_PY" -m pip install \
  --disable-pip-version-check --no-deps --require-hashes \
  -r requirements/locks/linux-x86_64/dev.lock.txt

WHEEL_ROOT="$OUT_ROOT/wheels"
mkdir -p "$WHEEL_ROOT/datatrove" "$WHEEL_ROOT/tokenizers"
"$DATA_PY" -m pip download --no-deps datatrove==0.10.0 -d "$WHEEL_ROOT/datatrove"
DATATROVE_WHEEL="$(find "$WHEEL_ROOT/datatrove" -name 'datatrove-0.10.0-*.whl' -print -quit)"
test -n "$DATATROVE_WHEEL"
test "$(sha256sum "$DATATROVE_WHEEL" | awk '{print $1}')" = \
  "c7bb75deed2c3e88fb5138f8ea075a170ee98d6c94fc263829609091ea9c2b5d"
"$DATA_PY" -m pip download --no-deps tokenizers==0.23.1 -d "$WHEEL_ROOT/tokenizers"
TOKENIZERS_WHEEL="$(find "$WHEEL_ROOT/tokenizers" -name 'tokenizers-0.23.1-*.whl' -print -quit)"
test -n "$TOKENIZERS_WHEEL"
test "$(sha256sum "$TOKENIZERS_WHEEL" | awk '{print $1}')" = \
  "5075b405006415ea148a992d093699c66eb01952bf59f4d5727089a98bda45a4"
"$DATA_PY" -m pip install "$DATATROVE_WHEEL" orjson regex "xxhash==3.8.1"
"$DATA_PY" -m pip install --no-deps "$TOKENIZERS_WHEEL"
"$DATA_PY" -m pip check

INTAKE="$OUT_ROOT/data183-external-intake"
EVIDENCE="$OUT_ROOT/data183-evidence"
TOK_EVIDENCE="$OUT_ROOT/tok187-evidence"

"$DATA_PY" tools/run_external_source_intake.py \
  --output "$INTAKE" \
  --max-download-bytes 2000000 \
  --max-normalized-chars 50000 \
  | tee "$OUT_ROOT/data183-external-intake.stdout.json"

"$DATA_PY" -m twelve_six.data183_corpus_v02_real build \
  --repo-root . \
  --source-sha "$SOURCE_SHA" \
  --external-intake "$INTAKE" \
  --output-dir "$EVIDENCE"
"$DATA_PY" -m twelve_six.data183_corpus_v02_real validate \
  "$EVIDENCE/corpus-v0.2-real-candidate.json" \
  --expected-source-sha "$SOURCE_SHA"

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false \
"$TOK_PY" -m twelve_six.tok187_bpe_real run \
  --repo-root . \
  --source-sha "$SOURCE_SHA" \
  --data183-report "$EVIDENCE/corpus-v0.2-real-candidate.json" \
  --build-root "$EVIDENCE/data110/build-a" \
  --manifest "$EVIDENCE/data110/build-a/manifest.json" \
  --environment-manifest "$ENV_MANIFEST" \
  --output-dir "$TOK_EVIDENCE"

"$TOK_PY" -m twelve_six.tok187_bpe_real validate \
  "$TOK_EVIDENCE/tok187-bpe-real-selection.json" \
  --expected-source-sha "$SOURCE_SHA"

"$TOK_PY" - <<PY
import json
from pathlib import Path
p = Path(${TOK_EVIDENCE@Q}) / "tok187-bpe-real-selection.json"
r = json.loads(p.read_text(encoding="utf-8"))
print(json.dumps({
    "source_sha": r["source"]["git_sha"],
    "corpus_identity_sha256": r["corpus"]["corpus_identity_sha256"],
    "selection_validation_identity_sha256": r["protocol"]["selection_validation_identity_sha256"],
    "500K_winner": r["model_probe_summary"]["500K"]["ranked_candidates"][0]["actual_vocab_size"],
    "1M_winner": r["model_probe_summary"]["1M"]["ranked_candidates"][0]["actual_vocab_size"],
    "promotion": r["promotion"],
    "report_sha256": r["report_sha256"],
}, ensure_ascii=False, sort_keys=True, indent=2))
PY
