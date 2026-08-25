# S0 engineering saturation and S1/S2 transition — 2026-08-25

This branch is a non-promoting engineering successor to the exact S0 Product candidate. It exists to move engineering attention out of discretionary S0 hardening while independent audit, governance and release authority remain separate.

## S0 freeze boundary

S0 Product source is PR #132 at exact head `86dbcc0b804da988a34367ff74c49ee00bc05818`.

That exact head is terminal SUCCESS for broad CI, real 40-step CPU training, fresh-process determinism/repeatability and strict exact-candidate evaluation. Strict evaluation remains deliberately non-promoting. No additional S0 hardening is required by this transition package.

Do not add new S0 checkpoint wrappers, inference wrappers, parity layers, portability evidence, profiling, serving adapters or acceptance bundles merely to increase coverage. Reopen S0 Product code only for a reproducible Product defect or an explicit independent-audit finding against the exact candidate.

## Preserved S1 ancestry

The transition lineage selectively composes two exact-green S1 mechanics sources as real Git parents:

- PR #106 `8894b44eb8627971b50a734fecd36e74293c1093`: S1 numerical mechanics preflight for the current 107,856-parameter engineering ModelSpec.
- PR #130 `2336e58848de6e30ea6c78f815176d75673f44bc`: S1 checkpoint/resume mechanics preflight, including fail-closed rejection of the S0 byte tokenizer as a canonical S1 tokenizer.

Their changed-path sets are disjoint from each other and from #132. Both remain engineering-preflight evidence only: no S1 architecture, tokenizer, corpus, mixture, optimizer or training recipe is frozen.

## Concrete S2 boundary

Current `configs/stages/s2_1m.json` describes an engineering S2 geometry with 1,066,112 trainable parameters, 2,048 vocabulary slots, context 512 and random-init Base.

`collect_s2_transition_preflight()` performs only a bounded synthetic-token forward/backward mechanics check against that exact config. It requires finite loss, finite gradients and at least one nonzero gradient while performing zero optimizer steps.

Its authority is `ENGINEERING_S2_MECHANICS_PREFLIGHT_ONLY_NOT_STAGE_EVIDENCE`.

The S2 preflight explicitly does not select S2 data or tokenizer, freeze architecture, claim quality/capability, authorize paid compute, or grant candidate/promotion authority. The 2,048-vocabulary geometry remains provisional and should be revisited using current vocabulary-allocation, tokenizer-fertility, context, attention-geometry and scaling experiments before any S2 freeze.

## Work that may continue while S0 administration is pending

S1/S2 engineering may continue on non-promoting branches for model-size/tokenizer allocation, initialization, context length, GQA/MQA experiments, optimizer recipes, stage-aware evaluation, checkpoint-v2/recovery, data-mixture work and LOCAL_FREE scaling experiments. Those activities must preserve random-init Base lineage and must not imply S0 audit or release approval.

S0 promotion remains separately blocked until independent AUDIT-A/AUDIT-B verdicts are reissued against the exact candidate and D10 governance/security/release authority is satisfied on the exact release lineage.
