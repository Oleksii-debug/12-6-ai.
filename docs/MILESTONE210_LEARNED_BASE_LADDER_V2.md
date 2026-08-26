# MILESTONE-210 Learned Base Ladder V2

`SWARM_WORKER_ID: MILESTONE-210-LEARNED-BASE-LADDER-V2`

## Verdict

The strongest terminal, directly comparable learned Base ladder currently defensible is **100K -> 500K -> 1M**. The 3M and 10M candidates are not admitted because their common-recipe execution is not terminal-success. SCALE-141 10M has terminal failed runs and is recipe-confounded relative to the M150 control family. 100M remains mechanics/qualification only.

This milestone is convergence, not a new framework. It binds the terminal MILESTONE-150 producer by exact source/workflow/artifact/report identities, persists the extracted V2 ladder facts in `evidence/milestone210/learned-base-ladder-v2.json`, and re-downloads/revalidates the exact terminal artifact in CI. It does not relabel queued or failed campaigns as learned evidence.

## Terminal producer

- Repository: `Oleksii-debug/12-6-ai.` (the trailing period is canonical).
- Source SHA: `5838cd16869dcfcf762368d8673eddf52d51b7e3`.
- Workflow run: `32937411703` — terminal `success`.
- Artifact: `9595677772` / `milestone150-learned-base-ladder-v1`.
- Artifact archive SHA-256: `c00b7e9006320f8916c739a3311e8cc47ad0d0b16957f8ebd7d19233fd9f1c71`.
- M150 ladder report identity: `1f8350bed574a7b78778f0ebb7854ca5311173006820ec27110122f8965c9a5a`.
- Common evaluation identity: `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`.

## Common truth model

All admitted rungs are random-init scratch Base models trained on the same **project-authored DATA-25** corpus, not an external-real or representative corpus. Corpus identity is `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`; the retained corpus has 20,000,775 train byte tokens and 1,410,473 validation byte tokens. The tokenizer is `s0-byte-v1`, vocab 256, no special tokens. All rungs use the same document-isolated seq-128 packing, FP32 AdamW recipe, seed 1337, InitSpec `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`, and exactly 948,504 optimized byte targets.

The terminal environment identity is CPython 3.11.16 with torch 2.13.0, numpy 2.4.6 and safetensors 0.8.0; environment hash `185db464d08873fb5b389e52e3be16e10e58827a9495f7ba181b9ffcc7af8fbc`, lock hash `48223bb44548b597a820b614dee9301db17d8ea8767568831c70c187f0d9589c`.

## Admitted learned rungs

| Rung | Parameters | Best held-out BPB | UA | EN | code | Best/final checkpoint |
|---|---:|---:|---:|---:|---:|---|
| 100K | 95,568 | 1.852985317050 | 2.025425976738 | 1.685492693075 | 1.681235376438 | `a8b5b2d2106a63a10a85e6ebba0b1bd5ea77fc9faf3836fb3319fac0ad0a6cbb` |
| 500K | 467,808 | 0.264545596881 | 0.379332978336 | 0.125796454995 | 0.178116269981 | `8d13262139f3fcb89c7efb141c7a449e4faba9048166d0dd2f49eba4288b4524` |
| 1M | 1,037,696 | 0.126517570964 | 0.129468030247 | 0.102005105122 | 0.145736104269 | `2292b43f0114479965d71e910185396af989738da0776a59bf6badb86990bf98` |

Direct quality ranking is 1M, then 500K, then 100K under the identical evaluation identity. Each rung retains the complete 0/250/500/750/1000 selection-validation trajectory, exact random-init state, best/final checkpoint identity, real fresh-process 500 -> 501 -> 1000 resume, fresh checkpoint reload, first-party logits fingerprints, evaluation non-mutation, reproducibility-manifest verification, and raw random/best/final greedy Base generation.

## Memorization / generalization boundary

The terminal M150 artifact supports a train-vs-heldout exposure/generalization proxy for every rung. M210 converts the terminal last-100 online byte-token NLL to BPB and records heldout-minus-train BPB without changing the source evidence. This is not a privacy-leakage or dedicated memorization-canary claim. RECOVER-178 is non-terminal at this convergence point, so none of its numerical canary results are imported.

## Excluded targets

- **3M:** RESEARCH-192 defines a 3,221,184-parameter common-recipe bridge, but run `32940278650` is queued rather than terminal-success. No terminal `LEARN-191` artifact was found. Status: `NOT_ADMITTED_NO_TERMINAL_LEARNED_EXECUTION`.
- **10M:** SCALE-141 has terminal failed learned-continuation runs (`32902872519`, `32937546047`, `32937882847`, `32937925825`). RESEARCH-192's exact 10,000,640-parameter fixed-control arm is in queued run `32940278650`. Status: `NOT_ADMITTED_NO_TERMINAL_SUCCESS`.
- **100M:** qualification/mechanics only unless genuine learned training evidence becomes terminal-success. It is not part of the learned ladder.

## Project-authored versus external-real evidence

The admitted ladder uses only project-authored DATA-25. DATA-183 explicitly separates `EXTERNAL_REAL` and `PROJECT_AUTHORED` origins, but its exact-head workflow run `32938788016` is queued and its candidate remains external-real UA/EN plus project-authored code. M210 therefore does not use DATA-183 for learned ranking and makes no representative-corpus claim.

## Execution gate

The M210 branch is based on ENV-151 and invokes `./.github/actions/execution-bootstrap` with `runtime,tests`. Dependency, executable, import, Python or hash-lock preflight failure prevents evidence validation. After bootstrap, the workflow downloads artifact `9595677772`, checks the exact archive SHA-256, validates the terminal M150 report self-identity, and cross-checks source/evaluation/corpus/model/checkpoint/optimized-token/fresh-verification identities against the V2 manifest.

## Claim boundary

No foreign pretrained weights. No SFT, RLHF, DPO, or paid compute. No claim of intelligence, production readiness, alignment, instruction following, external representativeness, or stage promotion.
