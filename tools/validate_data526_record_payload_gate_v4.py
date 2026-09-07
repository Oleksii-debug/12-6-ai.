from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG = Path("configs/data/research_corpus_v1_predecontam_blocker_v4.json")


def identity(doc: dict[str, object]) -> str:
    projected = dict(doc)
    projected.pop("evidence_identity_sha256", None)
    raw = json.dumps(projected, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def validate(doc: dict[str, object]) -> None:
    if doc.get("evidence_identity_sha256") != identity(doc):
        raise ValueError("evidence identity mismatch")
    if doc.get("status") != "BLOCKED_WAIT_RECORD_GRANULARITY_MATERIALIZATION":
        raise ValueError("record materialization blocker weakened")
    source = doc["source_convergence_terminal"]
    if source["head_sha"] != "991a0b6e939cddeff16c075922f7c407fa1e86cb" or source["run_id"] != 33046314943 or source["conclusion"] != "success":
        raise ValueError("terminal source authority drift")
    dedup = doc["global_dedup_terminal"]
    if dedup["head_sha"] != "d3333ec1b4a508df232a5aefccd6686adda745fb" or dedup["run_id"] != 33045763964 or dedup["conclusion"] != "success":
        raise ValueError("terminal dedup authority drift")
    if dedup["artifact_digest"] != "sha256:cca6921a2093d4e033976b23b0af180e9dc1945b624b82e218780f8d20bafd18":
        raise ValueError("dedup artifact drift")
    freeze = doc["record_materialization_gate"]
    if freeze != {"frozen": False, "record_count": 0, "record_inventory_digest_sha256": None, "payload_inventory_digest_sha256": None, "reason": freeze["reason"]}:
        raise ValueError("record freeze fabricated")
    boundary = doc["claim_boundary"]
    if boundary["authorized_unique_optimized_targets"] != 0 or boundary["corpus_frozen"] is not False or boundary["decontamination_executed"] is not False:
        raise ValueError("downstream scientific gate fabricated")
    if boundary["source_bytes_are_training_tokens"] is not False or boundary["tokenizer_fit_executed"] is not False or boundary["training_executed"] is not False:
        raise ValueError("source bytes promoted to training authority")
    gates = doc["downstream_gates"]
    if gates["record_inventory_freeze"] != "BLOCKED_MISSING_EXACT_RECORD_PAYLOAD_MATERIALIZATION" or gates["reserved_evaluation_decontamination"] != "NOT_PERMITTED_NO_FROZEN_RECORD_INVENTORY":
        raise ValueError("decontamination permitted without payload-bound record inventory")


def main() -> int:
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate(doc)
    print("PASS DATA-526 record-payload gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
