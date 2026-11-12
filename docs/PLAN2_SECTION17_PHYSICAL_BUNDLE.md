# Plan 2 Section 17 physical data bundle

The bundle builder uses the physical S3–S9 partition accepted in Section 16
and the frozen 32,768-token byte BPE from Section 15. The tokenizer file
contains the original 32,508 ordered merge rules. Those bytes were restored
from the exact S15 source checkout and verified against the manifest identity
retained by [the accepted Actions run](https://github.com/Oleksii-debug/12-6-ai./actions/runs/38292511894).
The S17 builder loads those rules directly.

Generate the training JSONL using
[Section 16's build command](PLAN2_SECTION16_SOURCE_ADMISSION.md), then run:

```bash
PYTHONPATH=src python -m tools.plan2_s17_physical_bundle_v1 \
  --root . --train "$ADMITTED_CORPUS/train.jsonl" --out-dir "$BUNDLE"
```

The builder checks the admitted payload and source rights before publishing
bundle files. It preserves the source and language of each document, applies
the incumbent S13 packer and produces the S14 causal exposure chain.
The train partition has 22,852 records, 71,256 packed blocks and 1,985,122
next-token targets across 1,114 shards. The targets include 918,536 Ukrainian,
554,758 English and 511,828 Python code positions. The 32-token block is the
packing geometry inherited from S13.

The validation and test manifests preserve EVAL-303 and EVAL-233 custody.
They bind 10 validation and 16 test records, including Ukrainian and English
in each role. The bundle carries their identities and hashes. Evaluation
consumers use the original reservations to access the physical sets.

The tokenizer and bundle identities are shared across the seven scale
profiles. The measured target count is for this exact corpus. Model
configuration, effective training exposure and launch qualification belong
to Plan 9. Section 18 handles the bundle's release and Plan 9 handoff.

Two independent processes produced byte-identical bundles from the two
Section 16 rebuilds. One process ran in a clean checkout. The full inventory
contains 2,237 files and 1,010,006,881 bytes. It was checked against the
per-file digests in the packing and exposure manifests and independently
read from both output directories. The
[qualification receipt](../evidence/plan2_s17/physical_bundle_receipt_v1.json)
binds the inventory SHA-256
`8c51f6a227908effa2c2ba0bddeb303c612b1623fc361b95c999c98841277d7d`
and the bundle identity
`3c357304e4db661bb74398ba4aa75c61c19b7727bca5817f8ec51296ee8765cd`.
