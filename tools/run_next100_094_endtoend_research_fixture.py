from __future__ import annotations

import os
from pathlib import Path

from twelve_six.post_base import (
    CANONICAL_BASE_EVIDENCE_NAMESPACE,
    POST_BASE_EVIDENCE_NAMESPACE,
    CanonicalBasePolicy,
    EvaluationSeparation,
)
from twelve_six.postbase_research_fixture import canonical_json, run_fixture


def main() -> None:
    raw = os.environ.get("NEXT100_094_BASE_CHECKPOINT")
    checkpoint = Path(raw) if raw else None
    trace = run_fixture(checkpoint=checkpoint)

    base_policy = CanonicalBasePolicy()
    separation = EvaluationSeparation()
    trace["communication_boundary"] = {
        "component": "POSTBASE-253",
        "canonical_base_namespace": separation.canonical_base_namespace,
        "post_base_namespace": separation.post_base_namespace,
        "namespaces_distinct": (
            separation.canonical_base_namespace != separation.post_base_namespace
        ),
        "canonical_base_policy": {
            "random_init_pretraining_origin": base_policy.random_init_pretraining_origin,
            "sft_applied": base_policy.sft_applied,
            "rlhf_applied": base_policy.rlhf_applied,
            "dpo_applied": base_policy.dpo_applied,
            "personality_applied": base_policy.personality_applied,
            "chat_template_applied": base_policy.chat_template_applied,
            "external_llm_inference_used_for_base": (
                base_policy.external_llm_inference_used_for_base
            ),
        },
        "expected_namespaces": {
            "base": CANONICAL_BASE_EVIDENCE_NAMESPACE,
            "post_base": POST_BASE_EVIDENCE_NAMESPACE,
        },
        "execution_authorized_by_contract": False,
    }
    assert trace["communication_boundary"]["namespaces_distinct"] is True
    assert separation.canonical_base_namespace == CANONICAL_BASE_EVIDENCE_NAMESPACE
    assert separation.post_base_namespace == POST_BASE_EVIDENCE_NAMESPACE
    print(canonical_json(trace), end="")


if __name__ == "__main__":
    main()
