from __future__ import annotations

from twelve_six.verify218_consumer_authority import (
    PRODUCER_ARTIFACT_NAME,
    PRODUCER_WORKFLOW_RUN_ID,
    SCHEMA,
    _json_normalize,
)
from twelve_six.verify218_learned_10m import (
    PRODUCER_ARTIFACT_ID,
    PRODUCER_ARTIFACT_ZIP_SHA256,
    PRODUCER_SHA,
    STATE,
    WORKER,
)


def test_consumer_authority_binds_exact_terminal_learn217_transport() -> None:
    assert SCHEMA == "12-6.verify218-learned-10m-independent.v2"
    assert WORKER == "VERIFY-218-LEARNED-10M-INDEPENDENT"
    assert STATE == "VERIFIED_LEARNED_10M"
    assert PRODUCER_SHA == "c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef"
    assert PRODUCER_ARTIFACT_ID == 9602650341
    assert PRODUCER_ARTIFACT_NAME == "learn217-terminal-10m-learned-base"
    assert PRODUCER_WORKFLOW_RUN_ID == 32952787070
    assert PRODUCER_ARTIFACT_ZIP_SHA256 == (
        "8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee"
    )


def test_json_normalization_matches_tuple_and_list_forms() -> None:
    left = {"betas": (0.9, 0.95), "nested": {"x": (1, 2)}}
    right = {"betas": [0.9, 0.95], "nested": {"x": [1, 2]}}
    assert _json_normalize(left) == _json_normalize(right)
