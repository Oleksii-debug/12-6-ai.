# D03 Rada_Trees HF snapshot -> archive inventory bridge

## Defect closed

The low-level archive inventory engine retains explicit expected-size/hash/object arguments for mechanics testing, but the authoritative bridge must not let an operator choose any source identity.

`tools/run_d03_rada_trees_inventory_from_hf_snapshot.py` is the authority path for real acquisition evidence. It derives exact byte size, Git/Xet object identity, and full content SHA-256 from pinned authorities only.

## Pinned content authority

For `uacorpus/Rada_Trees` at immutable dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`, the primary `Rada_Trees.7z` authority is bound to:

- Xet identity `a31d24710d417246fb7e48028baaf6b9efb9a199983d78264f7997c69e42a801`;
- full content SHA-256 `5e53939cd255276c58190569aebfaa6c90fb085fb10063e3e5f661747749719d`.

The config marks the provider content hash as pinned while keeping archive download and independent downloaded-byte verification false until execution.

## Fail-closed handoff

The bridge verifies:

- schema `12-6.d03-rada-trees-hf-object-identity.v1`;
- snapshot self SHA-256;
- dataset `uacorpus/Rada_Trees`;
- immutable revision `1b994a5804dcda122721e8d33a03fd172cf8d867`;
- exact ordered archive inventory `Rada_Trees.7z`, `rada_xtag_texts.7z`;
- positive exact primary byte size;
- 40-hex primary Git blob OID;
- 64-hex primary Xet identity and equality with the independently pinned Xet authority;
- terminal immutable-tree/Git/Xet/resolve-header verification vector;
- zero training authority and no false claim that the metadata-only parent already downloaded the archive.

Only after those checks does the bridge call the archive inventory engine. Exact size comes from the self-hashed parent snapshot; content SHA-256 and Xet authority come from the pinned inventory config. No content identity is accepted from the command line.

The inner inventory then requires the local archive bytes to rehash exactly to the pinned SHA-256 before extraction and requires the upstream object identity to equal the pinned Xet hash. The outer report binds parent snapshot identity, Git blob OID, Xet identity, pinned content SHA-256, and the complete deterministic member inventory under a new self-hash.

## Usage

First execute the PR #638 metadata probe at the exact dataset revision, then acquire the primary archive with the exact downloader. Run the bridge without any operator hash argument:

```text
python tools/run_d03_rada_trees_inventory_from_hf_snapshot.py \
  --archive evidence/d03-rada-trees/download/Rada_Trees.7z \
  --object-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --output evidence/d03-rada-trees/hf-snapshot-inventory-bridge-v1.json
```

`--extractor` remains optional and is constrained by the existing inventory policy.

## Downstream truth boundary

A successful bridge proves only:

`immutable HF metadata snapshot -> pinned Xet/content authority -> exact downloaded archive -> deterministic safe member inventory`.

It does not prove that any member is admissible training text. Original plain-text transcripts still need classification away from UD/nlp_uk/other derivatives, member-level attribution and provenance, period-level provenance/quality stratification, Ukrainian language/quality/privacy, global lineage dedup against overlapping Rada/ParlaMint/GRAC material, evaluation decontamination, family caps, deterministic corpus construction, tokenizer selection and post-pack unique causal-loss accounting.

`training_authorized_bytes = 0`. No tokenizer fit, optimizer update, model training, final-test access or paid compute is authorized by this bridge.
