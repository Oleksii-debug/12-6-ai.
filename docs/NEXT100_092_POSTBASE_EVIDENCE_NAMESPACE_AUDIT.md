# NEXT100-092 Base/Post-Base Evidence Namespace Audit

Worker: `NEXT100-092-POSTBASE-EVIDENCE-NAMESPACE-AUDIT`

Execution profile: `LOCAL_FREE` only. No training, external LLM, teacher API, paid
compute, or model-weight mutation is part of this audit.

## Firewall invariant

A post-Base behavior result, communication evaluation, SFT result, tool trace,
deliberation score, hypothesis-search object, verifier verdict, retrieval result, or
synthetic teacher-factory record is permanently in the `post_base` evidence domain.
It cannot become canonical Base training evidence or canonical Base scientific
evidence by changing a label, a path, a manifest flag, a model-lineage field, or a
parent reference.

Immutable Base checkpoint provenance may be referenced by post-Base components. That
reference is not a promotion surface. A Base provenance container may contain checkpoint,
ModelSpec, tokenizer, dataset/run-manifest, step, token-count, and similar inherited
identity fields. It may not contain post-Base scores, verdicts, final responses, tool
traces, retrieval results, synthetic-data decisions, or other post-Base outputs.

## Current component audit

The machine inventory is frozen in
`configs/post_base/next100_092_evidence_namespace_audit_v1.json` and binds the exact PR
heads inspected by this worker.

| Component | PR | Native status | Audit conclusion |
| --- | ---: | --- | --- |
| model adapter | 428 | EXPLICIT_FIREWALL | Typed `base` checkpoint provenance and `post_base` generation evidence are separate. |
| communication data | 437 | EXPLICIT_FIREWALL | Base-corpus and canonical-Base-training eligibility are hard false. |
| SFT runner | 433 | EXPLICIT_FIREWALL | Checkpoints/evaluations are restricted to post-Base artifact/evidence namespaces. |
| communication eval | 434 | EXPLICIT_FIREWALL | Evaluation namespace is post-Base; Base raw-LM diagnostics and training eligibility are false. |
| tools | 435 | CENTRAL_GATE_REQUIRED | `lineage=BASE` describes the model lineage, not evidence authority; serialized tool behavior lacks a native evidence-namespace field. |
| deliberation | 386 | CENTRAL_GATE_REQUIRED | Scores, verification summaries, and selected candidates are post-Base behavior but the returned trace has no native evidence-namespace field. |
| hypothesis search | 422 | CENTRAL_GATE_REQUIRED | Internal objects named `evidence` are hypothesis-search state, not canonical Base scientific evidence. |
| verifier | 423 | CENTRAL_GATE_REQUIRED | Deterministic verdicts are post-Base verification results and have no native Base/post-Base scientific-evidence namespace. |
| memory/RAG | 436 | CENTRAL_GATE_REQUIRED | `EvidenceObject` is retrieval evidence for post-Base orchestration, not canonical Base evidence. |
| teacher factory | 427 | EXPLICIT_FIREWALL | Synthetic records/manifests hard-code Base eligibility false and Base-corpus export fails closed. |

No current inspected component was observed relabeling a post-Base result as canonical
Base evidence. Five components nevertheless expose namespace-ambiguous output types, so
publication must pass the central envelope gate rather than infer authority from names
such as `Evidence`, `Verification`, `BASE` lineage, or a high score.

## Machine gate

`src/twelve_six/postbase_evidence_firewall.py` implements the reusable fail-closed
policy. `tools/validate_next100_092_postbase_evidence_namespace.py` validates the frozen
ten-component audit and can additionally validate serialized EvidenceEnvelope JSON or
JSONL artifacts.

The gate rejects:

- a post-Base envelope whose evidence namespace is `base`;
- post-Base artifacts written under `evidence/base`, `artifacts/base`, or `data/base`;
- any true Base-corpus, canonical-Base-training, canonical-Base-scientific,
  Base-training, Base-evaluation, or Base-raw-LM-diagnostic claim on post-Base output;
- Base/canonical-Base classifications or training-use relabeling;
- post-Base result fields smuggled inside an otherwise legitimate Base provenance
  container;
- an audit inventory that omits, duplicates, or promotes one of the ten required
  components.

The gate explicitly permits a model `lineage=BASE` field and immutable Base checkpoint
provenance references because neither is an evidence-authority claim.

## Adversarial proof

`tests/test_next100_092_postbase_evidence_namespace.py` includes positive controls for
Base checkpoint provenance, Base model lineage, communication-eval namespace references,
and hypothesis internal evidence. Negative fixtures attempt Base-path crossing,
Base-scientific relabeling, canonical Base training promotion, synthetic-data promotion,
and result-field injection into Base provenance. Every crossing must raise
`NamespaceViolation` before publication.

The scoped workflow is
`.github/workflows/next100-092-postbase-evidence-namespace.yml`. It runs only the
stdlib gate and adversarial unittest surface under the `LOCAL_FREE` execution profile.

## Truth boundary

This audit establishes an evidence-namespace firewall. It does not establish model
quality, communication quality, reasoning quality, retrieval quality, teacher quality,
canonical Base admission, learned-model admission, or repository-wide release status.
