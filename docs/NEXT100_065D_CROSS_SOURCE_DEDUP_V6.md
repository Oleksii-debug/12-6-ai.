# NEXT100-065D — Cross-Source Dedup V6

## Purpose

This stacked successor closes the known composition gap between NEXT100-065C / PR #632 and the stronger terminal source evidence already available for NumPy and Project Gutenberg.

It does not create a competing source registry. It re-materializes the inherited V5 graph, then consumes exact terminal source payloads under their existing training/evaluation/rights boundaries and runs the inherited lineage-aware global dedup engine over the complete successor graph.

## Exact authority additions

NumPy / NEXT100-049:

- PR #468 head `bca7a4c8afc5cb2546c35e3a0ebad9619cd3a4a8`.
- Dedicated run `32998548535 = success`.
- Authority identity `e9d2ce633915d6b6844b35e4abb0188974ef4791b208362c4f106ec0ad79ca70`.
- Five exact first-party Python source files.
- 36,898 identity-preserved UTF-8 bytes.
- One independent code family `github:numpy/numpy`.
- Evaluation remains not separately admitted.

Project Gutenberg / NEXT100-107 terminal seal:

- PR #627 head `c50b3f9cf871792c03886bdc1ccdc144812be88f`.
- Parent qualification head `3f4ad26e1e8f3406a1274418cf5f485814ce3032`.
- Dedicated parent run `32998859164 = success`.
- Authority identity `1b1bad11b688826ee4f73701c08e3b5af76ba16e8d8a806e008d5b84bee0b97b`.
- Three exact normalized book bodies totaling 1,672,110 UTF-8 bytes.
- All three records remain one independent English family.
- Model training is allowed only for the exact admitted normalized bodies; evaluation is not authorized; no worldwide public-domain claim is introduced.

## Expected pre-global-dedup vector

The inherited V5 vector is exactly 336,172 source-capacity bytes after the accepted-only CPython ledger is materialized.

V6 adds 36,898 NumPy bytes and 1,672,110 Gutenberg bytes, producing:

- 31 source objects.
- 14 independent families: UA 4, EN 5, code 5.
- UA: 100,856 bytes.
- EN: 1,838,293 bytes.
- code: 106,031 bytes.
- total: 2,045,180 bytes.
- remaining acquisition-planning gap to the 20,000,000-byte Research Corpus V1 milestone: 17,954,820 bytes before successor dedup collapse.

These values are source-capacity accounting, not optimized tokens and not authorized unique causal-loss positions.

## Verification design

V6:

1. re-materializes the full V5 object graph instead of trusting moving PR prose;
2. re-fetches NumPy files at the immutable upstream commit and verifies exact Git blob identities and byte counts;
3. re-fetches Gutenberg transports, verifies raw byte/hash/blob identities and reproduces `NEXT100_033_PG_BODY_NFC_LF_V1` exactly;
4. verifies exact terminal normalized Gutenberg identities;
5. binds live PR heads and completed dedicated workflow conclusions for the successor authorities;
6. runs the inherited exact/normalized/near-copy/fragment/code-skeleton/lineage dedup engine over all 31 objects;
7. materializes the report twice and requires byte-identical output;
8. keeps evidence text-free and retains all training/evaluation/compute claim boundaries as false.

The existing legacy NEXT100-065C workflow is extended rather than creating a new workflow, in accordance with `docs/CI_SWARM_POLICY.md`.

## Truth boundary

Even a green V6 result is not Research Corpus V1 release and not learned-20M authority. It does not authorize tokenizer fitting, optimizer updates, long training, final-test access or paid compute.

After V6, the critical chain remains: exact record inventory freeze, reserved-evaluation decontamination, final quality/privacy and balance/diversity revalidation, cluster-safe split, deterministic packing and two-clean-build proof, exact post-pack unique causal-loss ledger, tokenizer/FLOP calibration, D05 checkpoint-integrity terminal proof, bounded 20M training requalification, then explicit material-compute authorization.
