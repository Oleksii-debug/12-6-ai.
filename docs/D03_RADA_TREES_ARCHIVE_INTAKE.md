# D03 Rada_Trees archive intake

This step turns the pinned Hugging Face object identity for `uacorpus/Rada_Trees` into a local archive-content and member-inventory authority without granting corpus or training capacity.

## Exact inherited authority

The tool consumes the machine snapshot produced by `tools/probe_d03_rada_trees_hf_objects.py`. That snapshot is bound to dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867` and to the exact Hugging Face Git/Xet identities for `Rada_Trees.7z`.

The local archive must be named exactly `Rada_Trees.7z`, its byte length must equal the pinned Hugging Face object size, and its full streaming SHA-256 must equal an independently supplied lowercase 64-hex digest. An Xet object identity is not silently relabelled as the archive content SHA-256.

## Fail-closed member inventory

`7z l -slt` is parsed only after the member separator. The intake rejects malformed records, traversal, absolute or backslash paths, Unicode normalization drift, unsafe portable-name forms, duplicate/case-colliding members, symbolic/hard-link metadata, encrypted members, single-member expansion above the inherited 50 MB ceiling and total uncompressed expansion above the inherited 10 GB ceiling.

With `--hash-members`, extraction occurs only after the full listing has passed. The extracted regular-file set must equal the accepted listing exactly. Symbolic links, hard-linked files, other non-regular objects and byte-size drift fail closed. Every accepted file then receives a streaming SHA-256.

## Example execution

```bash
python tools/materialize_d03_rada_trees_archive.py \
  --archive /path/to/Rada_Trees.7z \
  --hf-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --expected-sha256 <independently-obtained-content-sha256> \
  --hash-members \
  --output evidence/d03-rada-trees/archive-member-inventory-v1.json
```

The command does not download the archive. It verifies bytes already obtained by an operator or acquisition process.

## Truth boundary

A successful report means only that the exact local primary archive matches the pinned object size, matches the supplied content SHA-256, has a fail-closed member inventory, and optionally has exact member hashes. It does not determine which members are original plain-text transcripts versus annotation/derived material. It does not terminalize attribution, member provenance, Ukrainian quality/privacy, cross-lineage deduplication, evaluation decontamination or family-cap feasibility.

Therefore `training_authorized_bytes` remains exactly `0`; tokenizer fitting, optimizer updates, learned-model training and paid compute remain unauthorized.

The next data step is deterministic plain-text-vs-annotation member classification plus member-level provenance/attribution binding, followed by quality/privacy, global lineage deduplication, evaluation decontamination and mixture/family-cap recomputation.
