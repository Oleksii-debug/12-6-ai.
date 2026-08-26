# D03 Rada_Trees member classification

## Purpose

This stacked successor to PR #708 implements the next data-plane seam for the high-capacity Ukrainian `uacorpus/Rada_Trees` source: classify an exact, fully hashed primary archive into original plain-text transcript candidates versus annotation, metadata and unknown-format holds.

The public dataset card describes approximately 88 million Ukrainian parliamentary tokens from 1990–2024 and explicitly says the corpus is available as original plain text, UD annotation and `nlp_uk` annotation under CC BY 4.0. It also identifies ParlaMint-UA and GRAC as related surfaces of the same parliamentary material. Therefore a safe pretraining path must not count every representation as independent text or independent family capacity.

## Exact parent

The classifier is stacked on D03 archive intake PR #708 exact head `e74afbd4a9883dab348c8698a748dc9003b79192`.

It refuses to run unless the parent report:

- has schema `12-6.d03-rada-trees-archive-intake.v1`;
- binds dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`;
- has a valid self-hash;
- verifies the exact `Rada_Trees.7z` archive identity;
- contains a complete hash for every regular archive member;
- has internally consistent paths, member counts and byte totals.

The classifier then re-hashes the archive, extracts to an isolated temporary directory, requires the extracted file set to equal the parent inventory exactly, and rechecks every member byte count and SHA-256 before classification.

## Classification contract

Only `.txt` members can become `PLAIN_TEXT_CANDIDATE`. The label is deliberately a candidate label, not admission.

Fail-closed holds include:

- known UD suffixes such as `.conllu`;
- XML/JSON/TSV/CSV annotation-like formats;
- metadata/readme/license material;
- NUL/binary payloads;
- unknown suffixes;
- `.txt` files whose content has CoNLL-U 10-column structure;
- `.txt` files that begin with configured markup prefixes;
- highly tabular `.txt` payloads consistent with annotation output;
- empty or undecodable text.

Text decoding is deterministic: strict UTF-8 with optional BOM first, then strict Windows-1251. The selected encoding is recorded per member. Raw member text and content previews are never written to the public report.

The report also records text-free metrics useful for the next Ukrainian language/quality filter and collapses exact duplicate SHA-256 values for diagnostic candidate-byte accounting. This exact collapse is not represented as global lineage dedup.

## Truth boundary

A successful run proves member classification mechanics only.

It does not prove:

- member-level rights or attribution completeness;
- source provenance for each transcript;
- Ukrainian language, quality or privacy PASS;
- independence from Rada laws, ParlaMint-UA, GRAC or mirrors;
- global exact/near/fragment/lineage dedup;
- evaluation decontamination;
- family-cap or 45/35/20 mixture feasibility;
- Research Corpus V1 release;
- tokenizer fit or nonzero unique causal-loss positions;
- optimizer updates, learned 20M/100M/1B weights, or paid compute authorization.

`training_authorized_bytes` and `unique_causal_loss_positions_authorized` remain exactly zero.

## Intended execution

After PR #708 has a real immutable archive SHA-256 and complete member-hash report:

```bash
python tools/classify_d03_rada_trees_members.py run /path/to/Rada_Trees.7z \
  --parent-report evidence/d03/rada_trees_archive_intake.json \
  --output evidence/d03/rada_trees_member_classification.json
```

Then verify the emitted report independently:

```bash
python tools/classify_d03_rada_trees_members.py verify \
  --report evidence/d03/rada_trees_member_classification.json
```

## Next successor

Bind attribution and member-level transcript provenance for only the plain-text candidates, then run Ukrainian language/quality/privacy filtering. After that, compare the surviving exact transcripts against the incumbent Rada laws family, ParlaMint-UA, GRAC and the live composed corpus under global lineage dedup before any nonzero source-capacity proposal.
