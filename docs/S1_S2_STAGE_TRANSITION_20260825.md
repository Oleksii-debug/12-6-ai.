# S0 engineering saturation and S1/S2 transition — 2026-08-25

This branch is a non-promoting engineering successor to the S0 Product lineage. It exists to move engineering attention out of discretionary S0 hardening while independent audit, governance and release authority remain separate.

## S0 freeze boundary

The last fully terminal-green integrated S0 Product head is PR #132 at `86dbcc0b804da988a34367ff74c49ee00bc05818`.

After that exact-green head, two independently exact-green correctness packages were selectively merged into the same W1 incumbent rather than left as competing implementations:

- PR #117 `9b86590bd1c1d51da2bdcfb21c6bfd47c19903fe`: fail-closed generation numeric/input contracts;
- PR #134 `16b3f510543cd1339eb3d804bdc6125e736bb2d7`: fail-closed parity oracle, including the transferred zero-logit-step authority guard.

The resulting live #132 head is `d07393f6f62b99c8106c0b72e6dd6ee53430e4dd`. It preserves #117 and #134 as real Git parents. Fresh exact-head validation is required because the candidate changed; parent/source PASS results are ancestry evidence only. Until the current head's decisive workflows are terminal, `86db...` remains the last fully validated integrated baseline and `d073...` is the pending S0 successor.

The intended freeze point is the first terminal-green exact head containing these already-landed correctness fixes. Do not add new S0 checkpoint wrappers, inference wrappers, parity layers, portability evidence, profiling, serving adapters or acceptance bundles merely to increase coverage. Reopen S0 Product code only for a reproducible Product defect, a failure on the exact candidate, or an explicit independent-audit finding.

## Preserved S1 ancestry

The transition lineage selectively composes two exact-green S1 mechanics sources as real Git parents:

- PR #106 `8894b44eb8627971b50a734fecd36e74293c1093`: S1 numerical mechanics preflight for the current 107,856-parameter engineering ModelSpec.
- PR #130 `2336e58848de6e30ea6c78f815176d75673f44bc`: S1 checkpoint/resume mechanics preflight, including fail-closed rejection of the S0 byte tokenizer as a canonical S1 tokenizer.

Their changed-path sets are disjoint from each other and from the transition's S0 base. Both remain engineering-preflight evidence only: no S1 architecture, tokenizer, corpus, mixture, optimizer or training recipe is frozen.

The transition branch currently preserves `86db...` plus #106/#130 ancestry. It must absorb the live `d073...` S0 successor only after that exact S0 head is terminal-green; the transition must then rerun its own exact-head workflows. This prevents queued/in-progress S0 evidence from being laundered into a next-stage lineage.

## Concrete S2 boundary

Current `configs/stages/s2_1m.json` describes an engineering S2 geometry with 1,066,112 trainable parameters, 2,048 vocabulary slots, context 512 and random-init Base.

`collect_s2_transition_preflight()` performs only a bounded synthetic-token forward/backward mechanics check against that exact config. It requires finite loss, finite gradients and at least one nonzero gradient while performing zero optimizer steps.

Its authority is `ENGINEERING_S2_MECHANICS_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE`.

The S2 preflight explicitly does not select S2 data or tokenizer, freeze architecture, claim quality/capability, authorize paid compute, or grant candidate/promotion authority. The 2,048-vocabulary geometry remains provisional and should be revisited using current vocabulary-allocation, tokenizer-fertility, context, attention-geometry and scaling experiments before any S2 freeze.

## Work that may continue while S0 administration is pending

S1/S2 engineering may continue on non-promoting branches for model-size/tokenizer allocation, initialization, context length, GQA/MQA experiments, optimizer recipes, stage-aware evaluation, checkpoint-v2/recovery, data-mixture work and LOCAL_FREE scaling experiments. Those activities must preserve random-init Base lineage and must not imply S0 audit or release approval.

S0 promotion remains separately blocked until the exact frozen Product candidate receives independent AUDIT-A/AUDIT-B verdicts and D10 governance/security/release authority is satisfied on that exact release lineage.
