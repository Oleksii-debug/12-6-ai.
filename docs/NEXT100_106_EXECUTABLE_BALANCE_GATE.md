# NEXT100-106 Executable Successor Balance Gate V3

## Purpose

NEXT100-106 turns the frozen DATA-295 / NEXT100-069 balance policy into an executable handoff gate for the successor post-global-dedup source vector.

It does **not** freeze Research Corpus V1 and does not authorize training. Its only job is to answer one deterministic question: given a terminal post-dedup set of independent family capacities, how many source bytes can be selected without replay while preserving the preregistered 45% Ukrainian / 35% English / 20% code mixture and family caps?

## Frozen policy

- planning target: 20,000,000 source bytes;
- Ukrainian / English / code: 45% / 35% / 20%;
- minimum two independent families in every stratum;
- one family may contribute at most 25% of the complete mixture;
- one family may contribute at most 60% of its own stratum;
- no replay, duplication, source-alias inflation, or BPB-guided mixture retuning.

The calculation uses a 100-byte budget quantum. This makes every frozen mixture fraction and both family-cap fractions exact integers, so the gate never relies on floating-point rounding.

## Input contract

`tools/next100_106_balance_gate.py evaluate` accepts only `12-6.next100-106-post-dedup-family-vector.v1`.

The input must be terminal and must bind:

- the exact dedup authority worker;
- a 40-hex exact head SHA;
- a 64-hex evidence identity;
- terminal verdict `PASS`;
- unique positive post-global-dedup source-byte capacity for every independent family;
- exact declared totals and family counts matching recomputation.

Duplicate family IDs are rejected. That prevents the same canonical lineage from being counted twice as quota capacity.

## Feasibility proof

For a candidate total `T`, the gate computes exact stratum targets. Each family's usable capacity is capped by all three limits:

1. its post-dedup unique capacity;
2. 25% of `T`;
3. 60% of its stratum target.

A stratum is feasible only when the sum of those capped independent-family capacities can cover the whole frozen stratum target. The gate then binary-searches the monotone feasible interval in exact 100-byte units and emits a deterministic constructive allocation for the maximum feasible total.

## Claim boundary

Even a 20,000,000-byte PASS means only that the **source-mixture balance gate** can be satisfied. The output deliberately leaves:

- authorized training loss positions at `0`;
- corpus identity `null`;
- shard identity `null`;
- tokenizer fitting unauthorized;
- model training unauthorized;
- paid compute unauthorized.

Quality/privacy revalidation, evaluation decontamination, cluster-safe split, deterministic pack/shard materialization, two-clean-build identity, and the full post-pack unique-loss ledger still have to pass before learned exposure can become nonzero.

## Commands

```bash
python tools/next100_106_balance_gate.py validate-policy
python tools/next100_106_balance_gate.py evaluate path/to/post_dedup_family_vector.json
python -m unittest tests.test_next100_106_balance_gate -v
```

Execution class: `LOCAL_FREE`.
