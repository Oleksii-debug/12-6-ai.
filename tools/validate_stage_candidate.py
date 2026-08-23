"""Validate a stage-candidate manifest without mutating repository state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from twelve_six.integration import (
    AuditVerdict,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)


def load_manifest(path: Path) -> StageCandidateManifest:
    raw = json.loads(path.read_text(encoding="utf-8"))
    components = tuple(
        ComponentRef(
            lane=item["lane"],
            source_sha=item["source_sha"],
            disposition=ComponentDisposition(item["disposition"]),
            component_kind=item["component_kind"],
            pr_number=item.get("pr_number"),
            artifact_sha256=item.get("artifact_sha256"),
            contains_behavioral_weights=bool(item.get("contains_behavioral_weights", False)),
            contains_foreign_pretrained_weights=bool(
                item.get("contains_foreign_pretrained_weights", False)
            ),
            notes=item.get("notes", ""),
        )
        for item in raw.get("components", [])
    )
    required = frozenset(
        raw.get(
            "required_lanes",
            ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"],
        )
    )
    return StageCandidateManifest.compose(
        stage=raw["stage"],
        integration_anchor_sha=raw["integration_anchor_sha"],
        status=CandidateStatus(raw["status"]),
        base_lineage=bool(raw["base_lineage"]),
        components=components,
        candidate_sha=raw.get("candidate_sha"),
        audit_a=AuditVerdict(raw.get("audit_a", "NOT_RUN")),
        audit_b=AuditVerdict(raw.get("audit_b", "NOT_RUN")),
        required_lanes=required,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    missing = manifest.missing_required_lanes()
    print(f"stage={manifest.stage}")
    print(f"status={manifest.status.value}")
    print(f"accepted_lanes={','.join(sorted(manifest.accepted_lanes())) or '-'}")
    print(f"missing_required_lanes={','.join(missing) or '-'}")
    print(f"audits_pass={str(manifest.audits_pass()).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
