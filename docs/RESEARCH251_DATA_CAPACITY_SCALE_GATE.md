# RESEARCH-251 Data Capacity Scale Gate

Worker: `RESEARCH-251-DATA-CAPACITY-SCALE-GATE`

Verdict: **DATA_LIMITED_BEYOND_10M_SHORT_SCIENCE**

This is a data-sufficiency gate only. No model training was executed. All decisions are LOCAL_FREE and intentionally avoid treating padded sequence capacity as training data.

## Authority consumed

The strongest terminal learned ladder is MILESTONE-221 (head `66b6c432f78042565bc09aaca5b3d4a95a3f358d`, PR #358). Its directly comparable 100K/500K/1M rungs each optimized 948,504 actual loss targets on DATA-25; all three selected the final checkpoint, so no held-out overfit onset was observed through that budget.

The strongest terminal 10M learned evidence is LEARN-217. It optimized 2,000,060 actual non-ignored causal loss targets with `corpus_replay=false`, covering about 10% of DATA-25. Held-out BPB continued to improve at every scheduled point:

| optimized loss tokens | held-out BPB |
|---:|---:|
| 500,026 | 0.1086155908 |
| 1,000,133 | 0.0939694709 |
| 1,500,237 | 0.0902736792 |
| 2,000,060 | 0.0858092047 |

Therefore the observed small-model evidence does **not** justify an early-overfit cutoff below 2M unique loss tokens.

The only terminal assembled train corpus is DATA-25, identity `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`: 43,238 train documents and 20,000,775 byte/source tokens (UK 9,000,418; EN 7,000,137; code 4,000,220). Those bytes are project-authored. DATA-25 explicitly has no external-real representativeness claim.

The latest immutable external-real authority found is DATA-229, registry identity `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`. It is **not** a terminal assembled research corpus. At its cutoff it contains three external-real text snapshots, two independent source families, zero code sources, and 173,358 normalized UTF-8 bytes. The family byte shares are approximately 48.9% Standard Ebooks manual and 51.1% Verkhovna Rada. No terminal DATA-230 or MILESTONE-238 research-corpus authority exists at this cutoff.

## Unique-token accounting

Two numbers must not be conflated:

- **2,000,060** is the strongest execution-proven no-replay unique-loss exposure. LEARN-217 counts only actual `labels[:, 1:] != -100` targets; padding is excluded.
- **20,000,775** is the terminal DATA-25 source-byte-token ceiling. The repository has not separately published an exact whole-corpus count of unique non-ignored causal loss positions.

Accordingly, 2,000,060 is the verified unique-loss floor and 20,000,775 is an upper source-token ceiling. Any experiment extending beyond the verified 2,000,060 should first materialize an exact source-position/unique-loss ledger. No experiment may exceed one pass and still call the excess unique data.

At the source-token ceiling, current supply is about 2.00 source tokens/parameter for the exact 10M model, 0.20 for the exact 100M candidate, 0.05 for the exact 400M candidate, and 0.02 for a nominal 1B model.

## Scale envelopes

These are project decision envelopes, not universal compute-optimal ratios. Where the repository has an explicit scale plan, that plan is used directly. Where it does not, the extrapolation is labeled.

| scale | minimum meaningful unique-token range | current unique availability | maximum exposure before uncontrolled recycling | data-limited? | next missing corpus requirement |
|---|---:|---|---|---|---|
| 10M | 5M–20M | 2,000,060 verified no-replay loss tokens; ≤20,000,775 source-token ceiling | 2,000,060 immediately verified; hard ceiling one DATA-25 pass after exact unique-loss ledger | **YES** for meaningful research/full pretraining; **NO** for mechanics and short science | terminal external-real UA/EN/code corpus, real code, multiple independent source families, exact no-repeat loss ledger ≥5M |
| 100M | 50M–200M | same 2,000,060 verified / ≤20,000,775 source ceiling | same one-pass ceiling; current source supply is only ~0.20 token/parameter | **YES** | ≥50M–200M unique external-real loss tokens for meaningful science; larger serious campaign should grow toward the existing SCALE-04 2.0B planning point |
| 400M | 250M–1B | same 2,000,060 verified / ≤20,000,775 source ceiling | current supply reaches only the lower SCALE-05 mechanics band | **YES** beyond mechanics | ≥250M–1B unique external-real tokens for meaningful comparison; 2B–8B for full-pretraining ambition, optional 12B only if held-out curves support it |
| 1B | 625M–2.5B | same 2,000,060 verified / ≤20,000,775 source ceiling | one pass only; current supply is ~0.02 source token/parameter | **YES** | ≥625M–2.5B unique diverse external-real loss tokens for meaningful learning; 5B–20B full-pretraining planning envelope |

### Mechanics qualification

Mechanics qualification is not a learned-quality claim and may use bounded fixtures/synthetic data. The existing 400M project plan explicitly uses 10M–50M byte tokens for mechanics. A 1B mechanics band of roughly 25M–125M is only a linear token/parameter extrapolation of that 400M plan and is not a pretraining recommendation.

### Short scientific training

The 10M terminal learned run at 2.0M no-replay loss tokens is a valid short scientific trajectory. The 100M SCALE-04 pilot is 50,003,968 scheduled byte tokens. SCALE-05 places ~250M at the 400M intermediate gate. For 1B, ~625M is a labeled extrapolation of that project-native 400M intermediate ratio.

### Meaningful learned campaigns

The 10M/100M decision band of roughly 0.5–2 unique tokens/parameter is intentionally conservative and project-specific: 10M is still improving at 0.20 token/parameter while the 1M ladder continues improving through ~0.91 token/parameter. This band is a gate for “enough data to make a scale comparison meaningful,” not a claim of optimal pretraining.

For 400M the repository already defines stronger project-native gates: ~250M intermediate and up to ~1B for serious runtime/optimizer/data comparison. The 1B 625M–2.5B band is the corresponding linear token/parameter extrapolation and must be replaced by empirical 1B evidence once available.

### Full pretraining ambition

The repository's 400M plan uses 2B → 4B → 8B as primary scratch-pretraining planning points, with 12B optional only if held-out curves support continued training. SCALE-04's serious 100M profile uses 2,000,027,648 tokens. For 10M and nominal 1B, 5–20 tokens/parameter is retained only as a planning envelope consistent with those project plans. It is **not** asserted as a Chinchilla-style universal optimum.

## Source-family sufficiency

Quantitative token count is not enough. Current terminal learned data are project-authored. DATA-229 contributes only two external-real source families, both text, and no admitted real code. The intended UA/EN/code research program therefore has no terminal external-real corpus authority yet.

A next corpus must add real code and multiple independent source families across UA and EN rather than scaling one family until the byte count looks large. Any family dominating most tokens must be reported explicitly. The current DATA-229 snapshot registry is roughly split 49/51 between its two families; that balance does not compensate for having only two families and zero code.

## Gate decision

- **10M:** allow mechanics and bounded short scientific training. Do not call the present data sufficient for a meaningful external-real research campaign.
- **100M:** do not start a meaningful learned campaign on current data. Mechanics-only work remains legitimate.
- **400M:** current data support only mechanics qualification. Meaningful learning is blocked.
- **1B:** architecture/runtime mechanics may be qualified, but learned scaling is data-blocked.

Global rule: **do not increase parameter count for meaningful learned campaigns above 10M until unique external-real data grows first.** The immediate corpus milestone is not “more repeated bytes”; it is a terminal, deterministic, rights-cleared, decontaminated, split-safe UA/EN/code research corpus with exact unique-loss accounting and multiple independent source families.
