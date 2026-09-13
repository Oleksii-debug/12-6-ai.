# 12-6 AI data-capacity acquisition plan

## Decision

The primary ~20M model mechanics are no longer the limiting surface. The limiting surface is unique, rights-cleared, decontaminated training data.

This control artifact converts the current terminal source authorities into a conservative acquisition gap without pretending that source bytes are optimized loss positions.

## Live evidence consumed

The legacy DATA-300 inventory contributes 183,061 normalized bytes across four independent families but DATA-301 remains `TERMINAL_BLOCKED` with no corpus identity and zero authorized balanced no-replay capacity.

Three exact-head source authorities are additionally terminal-green:

- NEXT100-022 Ukrainian Wikisource: 1,479 bytes, separate Ukrainian public-domain edition lineage;
- NEXT100-027 Verba/Nomis1864: 1,659 normalized bytes, separate Ukrainian public-domain corpus lineage;
- NEXT100-037 CPython documentation: 14 accepted chunks from a 17,901-byte normalized source. The accepted chunk hashes are terminal, but their exact retained byte sum is not published, so this plan gives them **zero conservative capacity credit** until record-level materialization.

This yields a conservative lower bound of 186,199 training-source bytes and seven observed independent families: UK 3, EN 2, code 2. Family-count diversity is therefore no longer the main acquisition problem. Volume, exact record materialization, decontamination, balance, and unique-loss accounting are.

## Primary ~20M gap

LEARN-345 requests 20,000,000 actual unique optimized causal positions, with a 10,000,000 meaningful activation floor. The 20,000,000-byte number used here is only a **necessary planning floor**: it is not a claim that one source byte equals one loss position.

At the project mixture target 45% Ukrainian / 35% English / 20% code, the planning floor is:

- Ukrainian: 9,000,000 bytes; conservative observed 91,703; gap at least 8,908,297.
- English: 7,000,000 bytes; conservative observed 84,793; gap at least 6,915,207. CPython retained-chunk bytes can reduce this only after exact materialization.
- Code: 4,000,000 bytes; conservative observed 9,703; gap at least 3,990,297.

Total conservative gap: **19,813,801 bytes**.

The family caps imply that at this 20M planning floor no single UK family may supply more than 5.0M bytes, no single EN family more than 4.2M, and no single code family more than 2.4M. Acquisition must therefore expand multiple independent lineages rather than finding one giant corpus and replaying it.

## Scale path

The immediate target remains the learned ~20M campaign. ~100M meaningful learned work stays blocked until ~20M has real learned evidence and the data plane grows to at least the project planning band of 50M–200M unique positions. The 1B stage remains planning-only at 625M–2.5B unique positions.

Parameter mechanics may be explored independently, but learned-scale promotion is data-gated.

## Next engineering order

1. Let the already-reserved Research Corpus V1 successor compose the exact terminal record inventory and produce a pre-decontamination corpus identity.
2. Materialize CPython's 14 accepted chunks with exact retained byte counts.
3. Acquire large, rights-clear source expansions in all three strata under the family caps; prioritize UK, then EN, then code by absolute deficit.
4. Run exact/near-match evaluation decontamination on the composed identity.
5. Run quality, privacy, exact/near dedup, cluster-safe split, balance, two clean builds, deterministic shards, and the post-pack one-pass unique-loss ledger.
6. Only after those gates pass may LEARN-345 be refreshed and material compute authorization requested.

No long training or paid compute is authorized by this plan.
