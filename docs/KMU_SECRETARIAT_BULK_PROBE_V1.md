# KMu Secretariat bulk capacity probe V1

## Purpose

Measure whether the already-admitted `ua.kmu.portal.secretariat-news` family can supply a materially useful share of the learned-20M Ukrainian source floor without inventing new rights, family credit, corpus identity, or training capacity.

This is a **PROBE**, not a corpus admission.

## Parent authority

The probe consumes the terminal NEXT100-026 / PR #449 boundary exactly:

- head `40950a950b60921fd856af2719e1ae2486d9e892`;
- manifest identity `1f068e6cc5ce3fc4a51d8477acee31fab5a0178e15f49225b57de94c5178f7d9`;
- family `ua.kmu.portal.secretariat-news`;
- `CC-BY-4.0`, training allowed with attribution;
- Secretariat-authored `kmu.gov.ua/news/` prose only;
- ministry/agency syndication, normative acts, contact/request/submission material excluded;
- evaluation not separately admitted; final-test use prohibited.

The successor does not create a second Ukrainian family. It measures the same lineage and therefore adds `0` family-count credit.

## Why this matters

The frozen learned-20M acquisition policy targets 9,000,000 Ukrainian source bytes. One family may supply at most 60% of its stratum, so this family has a hard planning ceiling of 5,400,000 bytes regardless of how much content exists. Rada_Trees is a separate active lineage; scaling KMu in parallel can reduce the risk that one Ukrainian family becomes the sole capacity bottleneck.

Source bytes remain planning units. They are not tokenizer tokens or causal-loss positions.

## Network execution

The scoped LOCAL_FREE workflow:

1. verifies the exact parent authority at its immutable Git commit;
2. reads `https://www.kmu.gov.ua/sitemap.xml` and recursively follows only same-origin sitemap children;
3. considers only canonical `/news/` URLs and rejects paths disallowed by the site's robots policy;
4. sorts candidate URLs deterministically by `(lastmod, URL)` descending;
5. fetches a bounded sample with at least 1.0 second between page requests;
6. accepts only pages containing the exact Secretariat byline used by NEXT100-026;
7. requires the site CC BY 4.0 marker and rejects contrary NC/ND/all-rights-reserved markers;
8. rejects short, weak-Ukrainian, email/phone-bearing, contact-like, normative and ministry-syndicated material;
9. records response and normalized probe hashes, duplicate counts, observed bytes and the fraction of the 5.4 MB family cap reached by the sample;
10. publishes only evidence. It does not commit fetched page payloads to the repository or mark them training-eligible.

A successful workflow means the capacity probe executed reproducibly enough to produce a hash-bound current-site report. It does **not** mean those sampled bytes are admitted into Research Corpus V1.

## Successor if yield is useful

A locked bulk materializer must freeze an exact URL inventory and per-page source/normalized hashes, preserve attribution, rerun quality/privacy, exact+near lineage dedup against all terminal corpus families, run evaluation decontamination, and only then hand retained records to the canonical source registry/global-dedup successor. The later family may contribute at most one family identity and at most the applicable family cap.

## Truth boundary

Always false/zero in V1 probe evidence:

- training-authorized bytes;
- added family credit;
- tokenizer fit;
- optimizer updates/model training;
- final-test access;
- paid compute;
- canonical source-registry mutation.

LOCAL_FREE only.
