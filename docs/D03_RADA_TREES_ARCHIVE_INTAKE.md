# D03 Rada_Trees archive intake

This stacked successor to PR #638 implements the first executable content-intake boundary for the primary `Rada_Trees.7z` object at dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`.

The public dataset card reports Ukrainian Verkhovna Rada plenary transcripts from 1990-2024, approximately 88 million tokens, and CC BY 4.0. Those are discovery facts only and are not converted into training capacity.

## Metadata-to-content provenance chain

The parent lane now pins the exact Hugging Face tree revision, 40-hex Git blob OID, 64-hex Xet object identity, exact archive byte size, and matching `X-Xet-Hash` resolve header into a deterministic self-hashed metadata snapshot.

`tools/materialize_d03_rada_trees_archive.py` requires that snapshot through `--object-snapshot`. It revalidates the snapshot self-hash, dataset/revision, parent verification vector, zero-training boundary and primary object identities. The local file must then be named exactly `Rada_Trees.7z`, and its byte size must equal the immutable object snapshot before content hashing begins.

A separate independently supplied lowercase SHA-256 is required for the full downloaded archive. The archive is fully hashed before 7-Zip is invoked. The report preserves the parent snapshot identity, Git blob OID, Xet identity, exact byte size and content SHA-256 as distinct evidence fields rather than treating one identifier as a substitute for another.

## Safe archive boundary

The 7-Zip technical listing rejects absolute/traversal/backslash paths, duplicate normalized paths, symbolic or hard links, malformed sizes, oversized members and excessive total expansion. Optional `--hash-members` extraction happens only after the complete listing passes, inside an isolated temporary directory. The extracted file set must match the listed regular-file set exactly; symlinks, byte-size drift and unexpected files fail closed. Every extracted regular file then receives SHA-256.

Example after the parent metadata snapshot and real archive SHA-256 are independently available:

```bash
python tools/materialize_d03_rada_trees_archive.py /path/to/Rada_Trees.7z \
  --object-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --expected-sha256 <64-lowercase-hex> \
  --hash-members \
  --output evidence/d03-rada-trees/archive-intake-v2.json
```

## Capacity and provenance firewall

Even a fully hashed archive/member report keeps `training_authorized_bytes = 0`. A later successor must identify the original plain-text transcript layer versus UD/nlp_uk/other derivatives, bind member-level attribution/provenance, and preserve period-level provenance/quality strata rather than treating 1990-2024 as automatically homogeneous.

The downstream sequence remains Ukrainian language/quality/privacy checks, exact+near lineage dedup against Rada laws/ParlaMint/GRAC/live corpus, evaluation decontamination, family-cap/mix recomputation, deterministic split/shard/pack, and exact post-pack unique causal-loss accounting. Only a later terminal authority may propose nonzero source-capacity credit.

No standalone Actions workflow is added while repository CI is saturated. LOCAL_FREE only; no tokenizer fit, model training, optimizer update, final-test access or paid compute.