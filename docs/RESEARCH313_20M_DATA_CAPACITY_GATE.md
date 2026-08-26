# RESEARCH-313 — 20M Data Capacity Gate

Worker: `RESEARCH-313-20M-DATA-CAPACITY-GATE`

Mode: `LOCAL_FREE`.

Verdict: **BLOCKED_NO_TERMINAL_FINAL_CORPUS_LEDGER**

Evidence identity: `d8aa1bbd9446c4ca881a97e9218891ee4223d6de60a5ee5534462978b5f2e970`

## Scope

This gate asks one narrow question: how many **actual unique nonignored causal loss positions** may a future approximately 20M-parameter Base optimize before any replay, using the final external-real corpus candidate.

It does not equate source bytes, padded slots, packed windows, epochs, optimizer steps, or a token/parameter heuristic with unique training exposure. It does not claim a universal compute-optimal token/parameter ratio.

## Authority cutoff and final candidate

DATA-300 PR #392, exact head `8ea7f830e50a23754d189dd4134f4afad76a7ee9`, freezes contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5` while explicitly keeping corpus state `NOT_BUILT_NOT_FROZEN_NOT_TERMINAL`.

DATA-301 PR #408, exact head `8820ba1b255f6bb95c7db0531fd846078a1aae01`, is the terminal execution result for that exact candidate. Its evidence identity is `939065abeefff8aed924415589608ff3fc721fe4b0a57fc200146a4b6a137e81` and status is `TERMINAL_BLOCKED`: corpus identity and shard identity are null, full five-source loss ledger is absent, and `authorized_balanced_no_replay_capacity` is **0**.

The exact candidate inventory is five training-admitted source objects, four independent families and 183,061 source bytes:

- UK: 88,565 bytes / 1 family;
- EN: 84,793 bytes / 1 family;
- code: 9,703 bytes / 2 families.

The selected DATA-295 policy remains 45% UK / 35% EN / 20% code with a 20,000,000 source-byte target, but its hard minimum is two independent families in every stratum. Current family counts are 1/1/2, so the family-constrained no-replay source-byte budget is **0**.

## Actual unique-loss evidence

DATA-294 is the terminal exact causal-position ledger available inside this evidence chain. It covers the three DATA-229 text objects only, under exact `s0-byte-v1`, document isolation, no BOS/EOS, and padding excluded.

Its one-pass total is exactly **173,355 unique nonignored causal loss positions**:

| scope | positions |
|---|---:|
| UK | 88,564 |
| EN | 84,791 |
| code | 0 |
| total text | 173,355 |

Ledger identity: `9a1cd57c52459bdc6e4bb2d46047a47713e10d9a5be7b0a4b86f041ba6f62bd0`.

This is real measured exposure capacity for its exact three-object text scope. It is **not** authority for the DATA-301 five-source final candidate. The full post-split, post-dedup, post-reservation Wave-3 train ledger does not exist, so 183,061 source bytes are not converted into a guessed loss-position count.

## Budget classes

### Mechanics

Mechanics qualification may continue on bounded fixtures or synthetic data because it is a trainer/runtime/checkpoint claim, not a learned-quality claim. If mechanics must stream the final external-real candidate as an authorized training corpus, current allowed learned exposure is **0 positions**. Mechanics evidence cannot raise corpus capacity.

### Short science

The project has a prior no-replay short-science precedent of **2,000,060 actual nonignored causal loss positions** on the 10M Base (RESEARCH-251 consuming LEARN-217). That is descriptive evidence that a bounded learned trajectory can be useful; it is not an optimum and is not automatically portable to 20M.

The final external-real candidate cannot authorize that budget. The only exact external-real ledger available in the chain is 173,355 text-only positions, while DATA-301 terminally blocks the five-source corpus.

### Meaningful learned campaign

RESEARCH-251 used a project-specific decision band of roughly 0.5–2 unique positions per parameter for 10M/100M scale comparisons, explicitly as a research-sufficiency envelope rather than a universal law. Applied only as a local planning reference, a nominal 20M Base corresponds to **10M–40M unique positions**.

This gate does **not** use that band to set the no-replay ceiling. The ceiling comes only from the terminal final-corpus train ledger. Current authorized exposure is **0 positions**.

## Maximum safe preregistered exposure before replay

**0 unique nonignored causal loss positions, now.**

This is a fail-closed authorization result, not a claim that the candidate sources contain zero learnable material. DATA-301 terminally retains these blockers:

- `G05_QUALITY`;
- `G06_PRIVACY`;
- `G09_BALANCE_DIVERSITY`;
- `G10_SELECTION_VALIDATION`;
- `G12_UNIQUE_LOSS`;
- `G14_TWO_CLEAN_BUILDS`.

Future unlock rule:

> Once a terminal final corpus exists, the maximum preregistered no-replay exposure is exactly the immutable train ledger's one-pass count of unique nonignored causal loss positions.

Nothing else may raise that number: not source-byte totals, padding, epochs, sampling with replacement, duplicated documents, source aliases, or a token/parameter ratio.

## 20M implication

A future ~20M Base can be mechanically qualified before the corpus gate opens, but a learned external-real campaign remains blocked. A meaningful 20M comparison has a project-local planning lower edge of 10M unique positions, but this is not an authorization threshold. The actual maximum remains whatever a successor terminal corpus ledger proves, capped at one pass.

No model training or optimizer updates were executed by RESEARCH-313.
