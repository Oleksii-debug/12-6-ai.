# MILESTONE-221 — Learned Base Ladder V3

## Verdict

MILESTONE-221 supersedes MILESTONE-210 using terminal evidence available at the execution cutoff.

The directly comparable equal-budget ladder remains **1M > 500K > 100K** by minimum held-out BPB. All three use the same M150 DATA-25/byte-tokenizer/InitSpec/AdamW/seq128/batch8 recipe and exactly 948,504 optimized targets.

- ~1M: 1,037,696 params; best/final BPB 0.12651757096387536; UA 0.12946803024656128; EN 0.10200510512203087; code 0.1457361042689415.
- ~500K: 467,808 params; best/final BPB 0.2645455968814711; UA 0.37933297833635227; EN 0.12579645499499625; code 0.1781162699814322.
- ~100K: 95,568 params; best/final BPB 1.8529853170496395; UA 2.0254259767376777; EN 1.685492693074772; code 1.6812353764375643.

Each admitted rung retains its exact ModelSpec, InitSpec, corpus/tokenizer/environment identity, complete 0/250/500/750/1000 selection-validation trajectory, best/final checkpoint identity, fresh 500→501 resume, final reload, train-v-heldout generalization proxy, raw Base generation, first-party UA/EN/code logits fingerprints, and evaluation non-mutation evidence in the machine record.

## New terminal memorization evidence

RECOVER-178 is now terminal-success at source `fc4b3a1ed39216ee8e4cc938283ece2bd44f4d68`, run `32938943596`, artifact `9597643947`, SHA-256 `bba24085b45f1c73f7f4735b7cbef9994d4c1a6f1d585921641a2a271375b665`.

It is a **separate canary-injection diagnostic experiment**, not the exact M150 ranking weights. At its terminal diagnostic checkpoint, all three scales trigger the frozen disproportionate-memorization stop rule. The 100K signal is NLL/rank only (exact-recovery lift 0); 500K and 1M have exact-recovery lift 1.0 in addition to NLL/rank signals. The diagnostic explicitly makes no privacy-leakage claim and preserves evaluation non-mutation.

## ~3M: learned producer exists, ladder admission remains gated

LEARN-191 is genuinely terminal-success: source `a75920cef8bde37a8c590e34095be83c97b75f1d`, run `32940842372`, artifact `9597788382`, SHA-256 `f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee`.

Exact model: 3,213,120 parameters, D192/L7/H12/KV12/head16/FF528/context256/vocab256, ModelSpec `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`; same DATA-25 identity, canonical byte tokenizer, InitSpec and fixed AdamW family.

Actual checkpoint boundaries are 17,125 / 66,417 / 131,938 optimized targets at steps 18 / 70 / 139. Aggregate selection BPB improves monotonically 7.995210405185286 → 5.697811934759831 → 3.486825260650889 → 2.2859499700392583. At the final/best checkpoint: UA 2.4833089811651017, EN 2.184148268823763, code 2.0560070470884018; train-probe BPB 2.289821102395208; train-minus-validation gap +0.0038711323559499355. Midpoint fresh-process resume and final fresh load pass; best=final checkpoint `920283a052c6c7fbd2b66f8ce5f775e4747c3d459e539b3625e4bebf1b9a7a59`. First-party greedy generation is retained.

However, the LEARN-191 artifact contains no separately retained logits fingerprint/verification record. A repository-wide current check found no published VERIFY-219 authority. Therefore the 3M producer is recorded as **different-token-budget learned evidence, not an admitted rung**. It must not be ranked directly against M150's 948,504-token models; VERIFY-219 remains the mandatory admission authority.

## ~10M

LEARN-217 exists as PR #355, but no terminal VERIFY-218 authority is published at this cutoff. CHECKPOINT-211 remains mechanics-only and explicitly records `full 10M retraining performed: false`. Therefore no 10M rung or quality number is admitted.

## External-real boundary

No separately verified terminal learned run using an external-real corpus satisfied this milestone contract at the cutoff. The admitted ladder remains project-authored DATA-25 evidence. No external-representativeness claim is made.

## Execution boundary

The MILESTONE-221 validation workflow uses the accepted ENV-151 universal bootstrap, re-downloads the exact M150, RECOVER-178 and LEARN-191 artifacts by immutable IDs, verifies their SHA-256 digests, and cross-checks the V3 record against retained reports. No foreign weights, SFT, RLHF, DPO or paid compute are admitted. This is learned Base evidence, not stage-promotion authority.
