# D03 Rada_Trees archive intake

This stacked successor to PR #638 implements the first executable content-intake boundary for the primary `Rada_Trees.7z` object at dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`.

The public dataset card reports Ukrainian Verkhovna Rada plenary transcripts from 1990-2024, approximately 88 million tokens, and CC BY 4.0. Those are discovery facts only and are not converted into training capacity.

## Executable boundary

`tools/materialize_d03_rada_trees_archive.py` requires a local file named exactly `Rada_Trees.7z` and an independently pinned lowercase SHA-256. The archive is fully hashed before 7-Zip is invoked.

The technical listing rejects absolute/traversal/backslash paths, duplicate normalized paths, symbolic or hard links, malformed sizes, oversized members and excessive total expansion. Optional `--hash-members` extraction happens only after the complete listing passes, inside an isolated temporary directory. The extracted file set must match the listed regular-file set exactly; symlinks, byte-size drift and unexpected files fail closed. Every extracted regular file then receives SHA-256.

Example after the real archive SHA-256 is independently known:

```bash
python tools/materialize_d03_rada_trees_archive.py /path/to/Rada_Trees.7z \
  --expected-sha256 <64-lowercase-hex> \
  --hash-members \
  --output evidence/d03/rada_trees_archive_intake.json
```

## Capacity firewall

Even a fully hashed archive/member report keeps `training_authorized_bytes = 0`. A later successor must classify original plain-text transcripts versus annotation derivatives, bind member-level attribution/provenance, run Ukrainian language/quality/privacy checks, execute exact+near lineage dedup against Rada laws/ParlaMint/GRAC/live corpus, apply evaluation decontamination, and recompute family-cap/mix feasibility before any source-capacity credit is proposed.

No standalone Actions workflow is added while repository CI is saturated. LOCAL_FREE only; no tokenizer fit, model training, optimizer update, final-test access or paid compute.