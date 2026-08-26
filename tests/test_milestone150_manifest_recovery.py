from __future__ import annotations

import json

from twelve_six.checkpoint import hash_json
from twelve_six.tokenization import ByteTokenizer
from twelve_six import milestone150_learned_base_ladder as base
from twelve_six import milestone150_entrypoint as recovery


def test_recovered_run_manifest_survives_json_round_trip() -> None:
    run = recovery.normalized_run_manifest(
        "a" * 40,
        "100k",
        base.model_spec("100k"),
        base.init_spec(),
        ByteTokenizer(),
        {"corpus_identity_sha256": "corpus"},
        {"identity_sha256": "evaluation"},
        base.trainer_config(),
        {"combined_sha256": "environment"},
    )

    persisted = json.loads(json.dumps(run, sort_keys=True))
    assert persisted == run
    assert run["trainer_config"]["betas"] == [0.9, 0.95]

    expected = run["identity_sha256"]
    unsigned = dict(run)
    unsigned.pop("identity_sha256")
    assert hash_json(unsigned) == expected
