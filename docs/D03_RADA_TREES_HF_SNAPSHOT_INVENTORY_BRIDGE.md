# D03 Rada_Trees HF snapshot -> archive inventory bridge

## Defect closed

The low-level archive inventory engine accepts `--expected-size` and `--upstream-object-identity` as operator inputs. That is useful for mechanics testing, but by itself it does not prove that those values came from the exact self-hashed Hugging Face object snapshot produced by PR #638.

`tools/run_d03_rada_trees_inventory_from_hf_snapshot.py` is therefore the recommended authority path for real acquisition evidence. It removes those two free-form inputs from the operator surface and derives them only from the exact parent snapshot.

## Fail-closed handoff

The bridge verifies:

- schema `12-6.d03-rada-trees-hf-object-identity.v1`;
- snapshot self SHA-256;
- dataset `uacorpus/Rada_Trees`;
- immutable revision `1b994a5804dcda122721e8d33a03fd172cf8d867`;
- exact ordered archive inventory `Rada_Trees.7z`, `rada_xtag_texts.7z`;
- positive exact primary byte size;
- 40-hex primary Git blob OID;
- 64-hex primary Xet identity;
- terminal immutable-tree/Git/Xet/resolve-header verification vector;
- zero training authority and no false claim that the metadata-only parent already downloaded or content-hashed the archive.

Only after that verification does the bridge call the existing inventory engine. The exact primary byte size and Xet identity are derived from the snapshot, never copied from command-line arguments. The independently supplied archive content SHA-256 remains a separate transport/content proof.

The outer report binds the parent snapshot identity, Git blob OID, Xet identity and the full inner deterministic member-inventory report under a new self-hashed bridge identity.

## Usage

First execute the PR #638 metadata probe at the exact dataset revision and retain its self-hashed JSON snapshot. After the primary archive is acquired and an independent full-content SHA-256 is known:

```text
python tools/run_d03_rada_trees_inventory_from_hf_snapshot.py \
  --archive /path/to/Rada_Trees.7z \
  --object-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --expected-sha256 <64-lowercase-hex> \
  --output evidence/d03-rada-trees/hf-snapshot-inventory-bridge-v1.json
```

`--extractor` remains optional and is constrained by the existing inventory policy.

## Downstream truth boundary

A successful bridge proves only the chain:

`immutable HF metadata snapshot -> exact archive object metadata -> independently content-hashed local archive -> deterministic safe member inventory`.

It does not prove that any member is admissible training text. Original plain-text transcripts still need classification away from UD/nlp_uk/other derivatives, member-level attribution and provenance, and period-level provenance/quality stratification. Ukrainian language/quality/privacy, global lineage dedup against overlapping Rada/ParlaMint/GRAC material, evaluation decontamination, family caps, deterministic corpus construction, tokenizer selection and post-pack unique causal-loss accounting remain downstream.

`training_authorized_bytes = 0`. No tokenizer fit, optimizer update, model training, final-test access or paid compute is authorized by this bridge.