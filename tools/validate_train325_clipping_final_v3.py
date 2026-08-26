#!/usr/bin/env python3
from __future__ import annotations
import hashlib
import json
from pathlib import Path

P = Path("evidence/train325/final_clipping_v3.json")
d = json.loads(P.read_text(encoding="utf-8"))

assert d["schema"] == "12-6.train325-10m-clipping-final.v3"
assert d["worker_id"] == "TRAIN-325-10M-CLIPPING-FINAL-V3"
assert d["decision"]["selected_gradient_clip_norm"] == 1.0
assert d["decision"]["lr_co_tuned"] is False
assert d["decision"]["new_optimizer_updates_executed_by_train325"] == 0
assert d["consumed_train243"]["terminal_status"] == "SUCCESS"
assert d["consumed_train243"]["paired_seeds"] == [20260825, 20260826, 20260827]
assert d["consumed_train243"]["equal_optimized_token_exposure"]["passed"] is True
assert d["consumed_train243"]["nan_inf_failed_before_clipping"] is True
assert d["research_corpus_v1"]["compatible_for_new_clipping_rerun"] is False
assert d["research_corpus_v1"]["status"] == "BLOCKED_NO_TERMINAL_FROZEN_RESEARCH_CORPUS_IDENTITY"
assert d["scope"]["paid_compute"] is False
assert d["scope"]["universal_clipping_claim"] is False

c = d["candidate_results"]
assert c["unclipped"]["preregistered_gate_evaluation"]["stable"] is False
assert c["clip_q95"]["preregistered_gate_evaluation"]["stable"] is False
assert c["clip_q90"]["preregistered_gate_evaluation"]["stable"] is False
assert c["incumbent_clip_1p0"]["preregistered_gate_evaluation"]["stable"] is True

identity = d.pop("identity_sha256")
raw = json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
assert hashlib.sha256(raw).hexdigest() == identity
print("TRAIN-325 PASS", identity)
