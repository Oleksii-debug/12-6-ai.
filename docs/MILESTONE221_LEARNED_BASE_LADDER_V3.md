# MILESTONE-221 — Learned Base Ladder V3

## Terminal verdict

MILESTONE-221 supersedes the temporal state recorded by MILESTONE-210.

The authoritative equal-budget learned ladder remains:

1. ~1M — 1,037,696 parameters — best/final held-out BPB 0.12651757096387536.
2. ~500K — 467,808 parameters — best/final held-out BPB 0.2645455968814711.
3. ~100K — 95,568 parameters — best/final held-out BPB 1.8529853170496395.

All three are the exact terminal MILESTONE-150 family at 948,504 optimized targets, with one DATA-25 identity, one byte tokenizer, one evaluation identity, one InitSpec family, one AdamW recipe, document-isolated seq128 packing, batch 8, and the retained locked environment.

## What changed after MILESTONE-210

RECOVER-178 is now terminal-success. Exact source `fc4b3a1ed39216ee8e4cc938283ece2bd44f4d68`, run `32938943596`, artifact `9597643947`, artifact SHA-256 `bba24085b45f1c73f7f4735b7cbef9994d4c1a6f1d585921641a2a271375b665`. This gives the 100K/500K/1M family a terminal dedicated memorization diagnostic in addition to the existing train-v-heldout generalization proxy.

LEARN-191 is also now a genuine terminal learned ~3M producer. Exact source `a75920cef8bde37a8c590e34095be83c97b75f1d`, dedicated run `32940842372` = success, artifact `9597788382`, artifact SHA-256 `f57bf36113a68fffd4bfcf877bf08762393479b9c09e6fd0fd613fbb91f044ee`. Its exact model is 3,213,120 parameters, D192/L7/H12/KV12/head16/FF528/context256/vocab256, ModelSpec `462c85da80a3c0d7d6a4f1a570b87d208b1847d8a57b12a4d9be7e36846b65dc`.

The ~3M producer is not admitted into the authoritative ladder until VERIFY-219 independently validates the retained artifact. Its preregistered token budget, 16,632 / 65,772 / 131,292 targets, is also materially different from the 948,504-target M150 comparison. It is therefore recorded as different-token-budget learned evidence and is not ranked as if it were an equal-budget fourth rung.

No ~10M rung is admitted. CHECKPOINT-211 is terminal recovery-mechanics evidence and explicitly records that full 10M retraining was not performed. No terminal VERIFY-218 authority establishing a genuine learned ~10M artifact existed at this execution cutoff.

## Exact equal-budget rungs

### ~100K

ModelSpec: D48/L3/H4/KV4/head12/FF128/context256/vocab256; 95,568 parameters; SHA-256 `4f1aaa6821360f0d22033356e011843646c8c14a6b4d20a3ad5b2ad125867470`.

Selection BPB trajectory: 7.965927514139019 → 4.302755434172232 → 3.0430575555052655 → 2.448567062685034 → 1.8529853170496395 at steps 0/250/500/750/1000. Final strata BPB: UA 2.0254259767376777, EN 1.685492693074772, code 1.6812353764375643. Best=final step 1000, checkpoint `a8b5b2d2106a63a10a85e6ebba0b1bd5ea77fc9faf3836fb3319fac0ad0a6cbb`. Fresh-process 500→501 resume and final reload passed. Last-100 train BPB 1.9794069112304282; heldout-minus-train -0.1264215941807887. Generation and first-party UA/EN/code logits fingerprints are retained in the machine record.

### ~500K

ModelSpec: D96/L4/H6/KV6/head16/FF256/context256/vocab256; 467,808 parameters; SHA-256 `208ac8ca113388e76f280d0154cae815785bee7705546f4d854d9447b9dd1f4a`.

Selection BPB trajectory: 7.939260616864241 → 2.7221165399832223 → 1.3000818654645316 → 0.7293227535119913 → 0.2645455968814711. Final strata BPB: UA 0.37933297833635227, EN 0.12579645499499625, code 0.1781162699814322. Best=final step 1000, checkpoint `8d13262139f3fcb89c7efb141c7a449e4faba9048166d0dd2f49eba4288b4524`. Fresh-process 500→501 resume and final reload passed. Last-100 train BPB 0.313944417144953; heldout-minus-train -0.04939882026348191. Generation and first-party logits fingerprints are retained in the machine record.

### ~1M

ModelSpec: D128/L5/H8/KV8/head16/FF352/context256/vocab256; 1,037,696 parameters; SHA-256 `ff3cee542a1f75bb4e1eff8d7d24d72533af8f4f3d82bd064fb1cbfeba8c8d07`.

Selection BPB trajectory: 7.801359708814326 → 1.860984214520523 → 0.611874879789735 → 0.17778162236855835 → 0.12651757096387536. Final strata BPB: UA 0.12946803024656128, EN 0.10200510512203087, code 0.1457361042689415. Best=final step 1000, checkpoint `2292b43f0114479965d71e910185396af989738da0776a59bf6badb86990bf98`. Fresh-process 500→501 resume and final reload passed. Last-100 train BPB 0.1298213442958011; heldout-minus-train -0.0033037733319257467. Generation and first-party logits fingerprints are retained in the machine record.

## Shared contract

DATA-25 identity: `422f545da93526ceba2c0ff738e0b9deac65f35bfb31d87d2aab068cda091fc8`. Tokenizer: `s0-byte-v1`, vocab 256, no special tokens. Evaluation identity: `7189e6df053574beb686727c94e684cdbaf08a34ef33aa953eff7cdae0320113`. InitSpec: `86483c6df623e80cab2f73aba718863fce18af6fe3b12430c1348414d92b48a5`. FP32 AdamW lr 3e-4, betas .9/.95, eps 1e-8, weight decay 0, constant schedule, no warmup, clip 1.0, batch 8, seed 1337.

Environment: Python 3.11.16, torch 2.13.0, numpy 2.4.6, safetensors 0.8.0; environment hash `185db464d08873fb5b389e52e3be16e10e58827a9495f7ba181b9ffcc7af8fbc`.

## External-real boundary

No external-real evidence is imported merely because a real-source corpus pipeline exists. No separately verified terminal learned run using that corpus met the MILESTONE-221 admission contract at the cutoff. The admitted ladder remains project-authored DATA-25 evidence.

## Execution boundary

The validation workflow uses the accepted universal execution bootstrap and verifies retained artifacts by immutable source/run/artifact/digest identity. No foreign or pretrained weights, SFT, RLHF, DPO, or paid compute are used or admitted. This is learned Base evidence, not stage-promotion authority.
