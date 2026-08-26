from __future__ import annotations

import json
from pathlib import Path

from twelve_six import data110_release_candidate as candidate
from twelve_six.data110_entrypoint import json_normalize, normalized_run_manifest
from twelve_six.tokenization import ByteTokenizer


def test_data110_run_manifest_is_fresh_process_json_stable(tmp_path: Path) -> None:
    spec, init, _ = candidate._model(Path("."))
    tok = ByteTokenizer()
    cfg = candidate._trainer_config()
    release = {
        "release_manifest_sha256": "2" * 64,
        "candidate_manifest": {"corpus_identity_sha256": "3" * 64},
    }
    run = normalized_run_manifest(
        "0" * 40,
        spec,
        init,
        tok,
        release,
        cfg,
        {"combined_sha256": "1" * 64},
    )
    path = tmp_path / "run-manifest.json"
    path.write_text(
        json.dumps(run, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == run
    assert run["trainer_config"]["betas"] == [0.9, 0.95]
    assert json_normalize(run) == run
