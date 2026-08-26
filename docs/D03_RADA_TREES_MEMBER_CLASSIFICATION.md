# D03 Rada_Trees member classification

## Decision

`MEMBERS_CLASSIFIED_PLAIN_TEXT_CANDIDATES_REQUIRE_PROVENANCE_QUALITY_AND_LINEAGE_DEDUP`

This stacked successor to PR #708 implements the next executable seam for the high-capacity Ukrainian `uacorpus/Rada_Trees` lane: classify a fully hashed primary archive into original plain-text transcript candidates versus annotation, metadata and unknown-format holds.

The public dataset card reports approximately 88 million Ukrainian parliamentary tokens from 1990–2024 and explicitly exposes the corpus as original plain text, UD annotation and `nlp_uk` annotation under CC BY 4.0. It also identifies ParlaMint-UA and GRAC as related parliamentary surfaces. One transcript represented multiple ways therefore cannot be counted repeatedly or promoted to multiple independent families.

## Exact parent authority

The classifier binds D03 archive intake PR #708 exact head `ff50eb1e3b9b264ac713e248d01e2342a9784156` and report schema `12-6.d03-rada-trees-archive-intake.v2`.

Execution refuses to proceed unless the parent report proves:

- exact Hugging Face dataset revision and object identity;
- exact Git blob OID and Xet object identity for `Rada_Trees.7z`;
- verified full archive content SHA-256 and immutable object-size match;
- a safe archive inventory;
- complete SHA-256 for every regular member;
- zero training/tokenizer/model authority.

The classifier re-hashes the archive, extracts to an isolated temporary directory, requires exact file-set equality with the parent inventory, and rechecks every member byte count and SHA-256 before classification.

## Classification policy

Only `.txt` members can become `PLAIN_TEXT_CANDIDATE`, and that label is not training admission.

Fail-closed holds cover known CoNLL-U suffixes, XML/JSON/TSV/CSV annotation formats, metadata, NUL/binary payloads, unknown suffixes, `.txt` payloads with CoNLL-U 10-column structure, configured markup prefixes, strongly tabular annotation-like text, empty text and undecodable text.

Text decoding is deterministic: strict UTF-8 with optional BOM first, then strict Windows-1251. Encoding and text-free structural metrics are recorded per member. Raw text and content previews are never emitted.

Exact SHA-256 duplicate plain-text candidates are collapsed for diagnostic candidate-byte accounting. That is not represented as global lineage dedup. Path-derived 1990–2024 year hints are recorded only as diagnostics and do not terminalize period provenance.

## Why period provenance remains open

Related UD_Ukrainian-ParlaMint documentation reports that parliamentary transcript characteristics vary over time, including grammatical corrections in older material and stronger normalization in recent speech-to-text-era transcripts. Therefore the 1990–2024 source must not be treated as automatically homogeneous merely because it is one archive. A later successor must bind actual member/session provenance and stratify relevant transcript-generation regimes before final quality/mix decisions.

## Truth boundary

A successful classification run still leaves all of the following blocked:

- member-level attribution and rights terminality;
- period/session provenance terminality;
- Ukrainian language, quality and privacy PASS;
- independence from Rada laws, ParlaMint-UA, GRAC or mirrors;
- global exact/near/fragment/lineage dedup;
- evaluation decontamination;
- family-cap and 45/35/20 mixture release;
- Research Corpus V1 release;
- tokenizer fit and unique non-ignored causal-loss authority;
- optimizer updates, learned 20M/100M/1B weights and paid compute.

`training_authorized_bytes = 0` and `unique_causal_loss_positions_authorized = 0` remain hard invariants.

## Intended execution

After PR #708 has a real immutable archive evidence report with complete member hashes:

```bash
python tools/classify_d03_rada_trees_members.py run /path/to/Rada_Trees.7z \
  --parent-report evidence/d03-rada-trees/archive-intake-v2.json \
  --output evidence/d03-rada-trees/member-classification-v1.json
```

Independent verification:

```bash
python tools/classify_d03_rada_trees_members.py verify \
  --report evidence/d03-rada-trees/member-classification-v1.json
```

## Required successor

Bind member/session attribution and source provenance for only the plain-text candidates, stratify period/transcript-generation provenance, run Ukrainian language/quality/privacy filtering, then execute global lineage dedup against Rada laws, ParlaMint-UA, GRAC and the live composed corpus before any nonzero source-capacity proposal.
