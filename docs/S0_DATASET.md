# D03 S0 deterministic dataset contract

This package is the first training-data vertical for the S0 ~10K factory. It is deliberately tiny and controlled. It proves data lineage and deterministic consumption; it is not a quality corpus and it does not support a claim that the project has globally eliminated benchmark contamination.

## Pipeline order

`immutable source registry -> extraction/read -> NFKC normalization -> language ID -> quality/PII filtering -> exact dedup -> tiny near dedup -> contamination checks -> deterministic train/validation split -> JSONL package + manifest`

The S0 source is a purpose-written project fixture. It is explicitly tagged synthetic in provenance metadata. It is not silently mixed with external data. The source registry records an immutable SHA-256 over the raw JSONL bytes and the build fails closed if those bytes change without a new registry identity.

## Licensing/provenance state

The fixture has no external source. Its metadata is intentionally `NOASSERTION` rather than a claim of general license cleanliness. Before D03 admits any external corpus, source-level origin, retrieval snapshot, applicable license/terms state, allowed use, and content hashes must be added and reviewed.

## D04 consumption contract

D04 should consume only:

- `data/s0/packaged/train.jsonl`
- `data/s0/packaged/validation.jsonl`
- `data/s0/packaged/manifest.json`

Every record contains `id`, normalized `text`, detected language, `source_id`, normalized content SHA-256, and explicit synthetic provenance tags. D04 must bind tokenizer/shard identity to `dataset_identity_sha256` from the manifest and must not silently reshuffle documents between train and validation.

Current deterministic package:

- dataset id: `s0-tiny-controlled-v1`
- accepted documents: 12
- train documents: 10
- validation documents: 2
- normalized text UTF-8 bytes: 2326
- train normalized text UTF-8 bytes: 1920
- validation normalized text UTF-8 bytes: 406
- dataset identity SHA-256: `bab60119d49e93303c972b77900fcb5553817f754cbc5d9a58019228cfa0ca89`

For a byte-level S0 tokenizer, normalized text UTF-8 byte counts are the closest pre-special-token token estimate. D04 owns the exact token count once tokenizer/special-token policy is fixed.

## Dedup strategy

S0 uses normalized-text SHA-256 for exact dedup. Near dedup is a deterministic word-shingle Jaccard implementation strictly bounded to at most 5,000 documents. Crossing that limit fails closed with an instruction to use DataTrove/MinHash rather than scaling the quadratic S0 implementation. The manifest schema is intended to remain stable when that backend changes.

## Contamination policy

The build rejects sources marked `benchmark`, `evaluation_test`, or `heldout_test`, and removes documents whose normalized SHA-256 appears in the contamination registry. The current registry contains a deterministic sentinel used to prove the mechanism. Before external data ingestion, D03 and D06 must replace/extend this with a real contamination-safe registry and benchmark-source policy.

## Rebuild

From the repository root:

```bash
python -m twelve_six.data.pipeline \
  --source-registry data/s0/source_registry.json \
  --contamination-registry data/s0/contamination_registry.json \
  --output-dir /tmp/12-6-s0-data
```

Tests rebuild into a temporary directory and require byte-for-byte equality with the committed train, validation, and manifest artifacts.

## Scaling seam

The source registry and packaged JSONL/manifest contracts are backend-neutral. At larger stages, extraction/filtering/statistics/dedup should move to DataTrove, Parquet/fsspec, Hugging Face Datasets, or equivalent mature streaming infrastructure where it is operationally better. The D03 contract remains provenance-first; backend changes require a new corpus identity/version and hashes.
