from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/tok316/authority-gate.json"
VALIDATOR = ROOT / "tools/validate_tok316_authority_gate.py"


def _rehash(value: dict[str, object]) -> None:
    core = dict(value)
    core.pop("gate_sha256", None)
    payload = json.dumps(
        core,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    value["gate_sha256"] = hashlib.sha256(payload).hexdigest()


def _run(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--input", str(path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_tok316_committed_gate_validates() -> None:
    result = _run(EVIDENCE)
    assert result.returncode == 0, result.stderr
    assert "TOK316_AUTHORITY_GATE_PASS" in result.stdout


def test_tok316_rejects_invented_training_even_with_fresh_self_hash(tmp_path: Path) -> None:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    value["execution"]["tokenizer_training_started"] = True
    value["execution"]["independent_training_runs_completed"] = 8
    value["candidate_results"][0]["training_run_1"] = "PASS"
    _rehash(value)
    mutated = tmp_path / "invented-training.json"
    mutated.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = _run(mutated)
    assert result.returncode != 0


def test_tok316_rejects_final_test_exposure_even_with_fresh_self_hash(tmp_path: Path) -> None:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    value["execution"]["final_test_bytes_read"] = True
    _rehash(value)
    mutated = tmp_path / "final-test-exposure.json"
    mutated.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = _run(mutated)
    assert result.returncode != 0


def test_tok316_rejects_grid_or_runtime_substitution(tmp_path: Path) -> None:
    value = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    value["protocol"]["requested_vocab_grid"] = [256, 320, 384, 512]
    value["maintained_bpe"]["library_version"] = "0.23.2"
    value["maintained_bpe"]["self_written_bpe_substitution_allowed"] = True
    _rehash(value)
    mutated = tmp_path / "substitution.json"
    mutated.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = _run(mutated)
    assert result.returncode != 0
