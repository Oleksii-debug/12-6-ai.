# NEXT100-032 — English Wikisource source qualification

Worker: `NEXT100-032-DATA-EN-WIKISOURCE`

Terminal result: **ADMIT**, but only for the exact bounded, rights-clear scan-backed subset in
`evidence/next100_032/en_wikisource_qualification.json`.

The whole English Wikisource project/dump is **REJECTED as a single blanket training authority**.
The platform mixes public-domain works, compatibly licensed works, contributor-created material,
translations, annotations, and edition-specific material. Public availability does not satisfy the
project's explicit model-training rights gate.

## Admitted bounded object

The admitted pilot is exactly three validated Page-namespace revisions from William James,
*The Varieties of Religious Experience*, Longmans, Green, and Co., 1902:

- scan page 20 — revision `8450353`;
- scan page 21 — revision `8450364`;
- scan page 22 — revision `6931309`.

The work page is pinned to revision `14857116`; the scan Index is pinned to revision `13271826`.
The index records the 1902 Longmans edition and that all work pages are validated.

No category crawl, current-page alias, dump-wide import, talk/user/project content, modern
translation, modern annotation, or unreviewed illustration is authorized by this source authority.

## Rights separation

1. **Underlying work.** The pinned work page marks this 1902 William James work PD-old and public
   domain worldwide.
2. **Platform contribution layer.** English Wikisource's copyright policy states that, unless
   otherwise noted, user contributions are released under CC BY-SA 4.0 and unversioned GFDL.
   This authority selects CC BY-SA 4.0 as the redistribution compliance path.
3. **Edition-specific layer.** The admission is limited to the pinned 1902 scan transcription.
   A different translation, introduction, annotation, image set, or edition requires a fresh
   rights decision.

Source extraction/training is allowed. Redistribution of the source-derived text must retain the
required attribution/license/change information and satisfy ShareAlike when applicable. This
authority deliberately does not decide whether trained model weights are Adapted Material.

Evaluation remains `NOT_SEPARATELY_ADMITTED`.

## Attribution

A downstream source ledger must retain the work/edition citation, English Wikisource, all selected
permanent revision URLs, a contributor-history locator for each selected page, the CC BY-SA 4.0
license URI, and a description of extraction/normalization changes.

## Determinism and quality

The bounded selection is revision-ID exact and contains 3/3 validated pages. The deterministic
normalization is UTF-8 + Unicode NFKC + horizontal-whitespace collapse while preserving paragraph
boundaries, ascending scan-page order, and a terminal LF.

Sealed extracted snapshot identity:

- extracted bytes before normalization: `5535`;
- extracted SHA-256: `263404f43b0c3c6964bf266a34694e06db96bbb0d1dbba37f29e475480d4e46d`;
- normalized bytes: `5536`;
- normalized SHA-256: `1c4ec6e66b425e517a17fb865dabaf3aeddfb1a16cb7e40bca2984be56dce0e7`;
- words: `930`;
- Unicode replacement characters: `0`;
- ASCII/Latin letter share among letters: `1.0`.

The raw text itself is not committed by this qualification PR; the immutable Page revision IDs and
hashes are the acquisition authority. A later corpus materializer must reacquire only those exact
revisions and reproduce the sealed hashes before storage/promotion.

## Dedup and family lineage

Within the bounded subset there are no duplicate revision IDs or normalized page hashes, and the
maximum pairwise word-5-gram Jaccard in this three-page pilot is `0.0`.

Against the sealed live external registry, exact normalized hashes do not collide. This is not a
replacement for the repository's cross-source near-copy/mirror audit: that audit remains mandatory
before canonical corpus inclusion.

Family lineage is explicitly parented to `wikimedia`. The source receives **zero automatic
independent-family diversity credit**. A later registry convergence worker must reconcile it against
any English/Ukrainian Wikipedia, Wikisource, Wikibooks, Commons-derived, or other Wikimedia sibling
authorities and against shared underlying editions.

## Truth boundary

This is source-qualification authority, not a corpus freeze, not an evaluation authority, not a
representativeness claim, and not permission to ingest English Wikisource wholesale.
