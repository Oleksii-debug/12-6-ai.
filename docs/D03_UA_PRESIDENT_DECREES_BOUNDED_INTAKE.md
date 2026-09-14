# D03 — Ukrainian Presidential decrees bounded intake

## Purpose

Open one additional Ukrainian source family on the learned-20M data-capacity critical path without
bypassing D03 rights, privacy, deduplication, evaluation-decontamination, split, packing, or
unique-loss gates.

The source is the official decrees catalogue of the President of Ukraine:
`https://www.president.gov.ua/documents/decrees/`.

## Rights boundary

The authority used here is narrow. Article 8(1)(3) of Ukrainian Law No. 2811-IX excludes acts of
state authorities and official political, legislative, administrative and judicial documents,
including decrees, from copyright protection. The source contract therefore covers only the text
of an official Presidential decree and official text embedded as part of that act.

It does **not** treat the whole website as training-rights-clear. Site navigation, news, press
materials, photos/video/design, linked third-party material, and non-official attachments remain
outside this lane. The site's general CC BY-NC-ND notice is recorded but is not used as blanket
training authority.

## Bounded acquisition

The LOCAL_FREE probe:

1. discovers only same-origin decree links from the official catalogue;
2. reads each catalogue page twice and requires the exact discovered decree URL list to match;
3. follows only HTTPS same-origin redirects;
4. reads each decree twice and requires the canonical extracted decree text to match exactly;
5. records raw-response SHA-256 values but does not require dynamic page chrome to be byte-identical;
6. stores no decree body or subject text in durable evidence;
7. records only URLs, decree number, byte counts, hashes, acceptance state, and machine rejection reason.

Default execution is two catalogue pages and at most 40 decree URLs. Hard limits are eight pages
and 120 documents, with at least one second between requests.

## Privacy and quality

The first intake deliberately prefers long normative/policy acts and fails closed on likely
personal-data-heavy material. It rejects personnel, awards, citizenship, sanctions and related
subjects; email, phone, street-address and long-identifier patterns; high initialized-name density;
short payloads; low Ukrainian-letter ratio; control characters; and exact normalized duplicates.
The fixed Presidential signatory line is removed from the candidate payload before hashing.

These gates are conservative source-intake guards, not a claim that privacy review is finished.
A survivor still requires the normal project post-composition privacy gate.

## Scientific boundary

Observed surviving bytes are **pre-global-dedup source-intake evidence only**. This package grants:

- canonical corpus capacity credit: 0 bytes;
- training-authorized bytes: 0;
- authorized unique causal-loss positions: 0;
- tokenizer-fit authority: false;
- model training / optimizer updates: false / 0;
- final-test access: false;
- paid compute: false.

Before any credit or training exposure, the exact surviving URL/hash graph must pass the incumbent
global exact/near/fragment/lineage deduplication, reserved-evaluation decontamination,
post-composition quality/privacy/balance/family caps, cluster-safe split, deterministic tokenizer and
packing double-build, and positive exact unique causal-loss ledger.

Presidential decrees can quote or incorporate Rada laws, Cabinet acts and other official material,
so cross-family fragment and lineage deduplication is mandatory. The entire source remains one
conservative family: `ua.president.official-decrees`.

## Reproduction

Static tests and validator are network-free. The temporary specialist workflow exists only to
obtain the bounded external evidence on a networked LOCAL_FREE runner. It performs two independent
materializations and requires the stable materialization identity to match before uploading the
body-free evidence artifact. After terminal evidence is consumed, the temporary workflow should be
removed rather than retained as permanent Actions fanout.
