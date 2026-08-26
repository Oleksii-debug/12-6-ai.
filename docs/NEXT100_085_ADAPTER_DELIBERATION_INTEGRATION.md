# NEXT100-085 Base-to-Deliberation Adapter Integration

Worker: `NEXT100-085-ADAPTER-DELIBERATION-INTEGRATION`

Runtime policy: `LOCAL_FREE` only. No external model judge, retrieval service, network model, trainer, optimizer, or checkpoint writer is introduced by this integration.

## Authorities consumed

- POSTBASE-351 adapter source head at integration start: `e805bf715617999209aef88946cea01f3668f583`.
- POSTBASE-255 deliberation controller source head: `486bd91ca03bed41750c638d702f557f320b780a`.
- POSTBASE-256 hypothesis-search source head: `ea1d8fff0d3235660dffe7ba411e192df83f5e1d`.
- LEARN-217 learned-10M producer head: `c02c8aa38e691521ae2ab6a4ff3ea1d643efd6ef`, ModelSpec SHA-256 `61caa5469123e23b9b72fc2024140bfca84c4c480dcb0a7e712ba800a4f22998`.
- POSTBASE-351's terminal learned-10M proof consumes the retained best/common checkpoint `12f9edd88bf5e596ae6f985564a5dcff96033922100ba91678ef9a76c0df3156` from LEARN-217 artifact `9602650341`, whose ZIP SHA-256 is `8631e90417e40365b3fc0d6bc98ee6adda5a4ed24530e675d9a91c93219537ee`. This is the checkpoint identity bound by NEXT100-085.
- EVAL-284 consolidated panel head: `eb6fec707cedb4ae3fa9890d176b5a74d1b7e2d0`. Its panel verdict is `INCOMPLETE_WAVE1_10M_EVIDENCE` and records `VERIFY_218_independent_learned_10M_authority_missing_at_cutoff`. Its separately recorded final/checkpoint role is not substituted for the retained best/common checkpoint consumed by POSTBASE-351.

The dedicated NEXT100-085 workflow therefore consumes the exact learned 10M checkpoint identity already authorized by POSTBASE-351 for immutable Base-generation mechanics. This does **not** upgrade the separate EVAL-284 panel: no independent VERIFY-218 result is inferred, and no reasoning-quality or broad-capability claim is made from this integration proof.

A final live POSTBASE-351/VERIFY-10M authority refresh is required before terminal verdict. Any newer terminal checkpoint-verification authority must be bound exactly rather than silently inheriting the older artifact identity.

## Integration boundary

`DeliberationBaseBridge` implements the existing POSTBASE-255 `ModelAdapter` protocol. Each controller request is copied into a new canonical JSON value and then into a fresh POSTBASE-351 `ControllerGenerationRequest`. Deadline/tool-budget controller state is not serialized into the Base prompt. The Base receives generation input only, not the controller trace or mutable controller bookkeeping.

`HypothesisBaseBridge` is a model-assisted facade around the existing POSTBASE-256 `HypothesisSearch`. Generated text may be used as a proposal, critique, or revision input, but the hypothesis graph remains a post-Base controller object. Deterministic test evidence stored by `HypothesisSearch` is orchestration evidence, not Base evidence.

POSTBASE-351 now admits a distinct `controller="hypothesis"` discriminator so hypothesis calls are not mislabeled as deliberation or tool calls.

## Evidence firewall

Base checkpoint provenance remains the frozen `BaseCheckpointEvidence` record exposed by `PostBaseModelAdapter.base_evidence` with `evidence_namespace="base"`.

Bridge call records use `ControllerCallEvidence` with `evidence_namespace="post_base"` and nest only `PostBaseGenerationEvidence`. They do not copy checkpoint id, Base git SHA, ModelSpec hash, parameter count, dataset/run manifest hashes, or learned-token counters.

The hypothesis graph export similarly contains hypotheses, scores, tests, critiques, evidence and contradictions only. It cannot silently become Base checkpoint evidence.

## Immutability proof

`tests/test_next100_085_adapter_deliberation_integration.py` checks:

1. controller `Request` and Base `ControllerGenerationRequest` are distinct objects and the serialized Base request excludes controller deadline/tool-budget state;
2. hypothesis model calls use the dedicated `hypothesis` discriminator;
3. Base and post-Base evidence namespaces do not absorb each other's fields;
4. two complete deliberation-controller invocations leave every checkpoint file byte-identical;
5. the loaded model state-dict digest is identical before and after repeated controller calls;
6. frozen Base evidence is unchanged after those calls.

The dedicated learned-10M workflow additionally downloads the exact POSTBASE-351-authorized LEARN-217 artifact, verifies its ZIP digest and retained checkpoint identity, runs repeated deliberation calls plus hypothesis proposal/critique calls, compares the in-memory Base-weight digest before/after, and compares the checkpoint tree hash before/after.

The mechanics verifier intentionally returns a deterministic pass so the test terminates cheaply. This proves plumbing, budget handoff, namespace separation and non-mutation only. It is **not evidence of reasoning quality, intelligence, instruction following, factuality, or production readiness**.
