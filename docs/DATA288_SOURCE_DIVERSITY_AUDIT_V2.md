# DATA-288 — Source diversity audit V2

Worker: `DATA-288-SOURCE-DIVERSITY-AUDIT-V2`.

## Verdict

`BLOCKED_SOURCE_DIVERSITY` for the next external-real corpus.

This is a source-diversity audit only. It performs no model training and makes no model-quality,
representativeness, production-readiness, or benchmark-clean claim.

## Terminal scope

Counted authorities are frozen to the evidence available at the worker cutoff.

- DATA-229 head `90bc0b7f8b696ec35202532b13edf6ab29a662fe`: dedicated run
  `32957147036` = `success`; registry
  `1357a343eb4ea973950d8991913109cbea53fe4fa891f0be9745ab497eb59486`.
- DATA-227 head `8ebdb2e132ed7bae5245e9d4c140752640ab9885`: dedicated run
  `32956209865` = `success`; artifact `9602093542`,
  digest `sha256:080f073327020cb3bbb05c7348f658223804684d23012d9b66ab9b798c4fed5d`.
- DATA-228 head `46a70c990dab6ff72bb84ddb54cff1156b491b40`: dedicated immutable-source
  probe run `32957120454` = `failure`; its Kubernetes/CPython candidates are excluded.

## Exact current diversity

Five admitted source objects collapse to four independent source families:

| stratum | independent families | normalized unique bytes |
|---|---:|---:|
| Ukrainian text | 1 | 88,565 |
| English text | 1 | 84,793 |
| code | 2 | 9,703 |
| total | 4 | 183,061 |

Raw source bytes total 448,214. Raw-byte weighting is reported only as a diagnostic because DATA-229
text normalization/extraction changes the consumable byte mass; diversity gates use normalized unique
training bytes.

Family-normalized byte shares are:

- `ua.rada.open-data.laws-texts`: 88,565 / 183,061 = 48.380048%.
- `en.standardebooks.manual`: 84,793 / 183,061 = 46.319533%.
- `github:encode/httpx`: 8,161 / 183,061 = 4.458077%.
- `github:psf/requests`: 1,542 / 183,061 = 0.842342%.

The top-family share is 48.380048%. Shannon entropy is 0.886660119575 nats
(1.279180157465 bits), giving an effective-family count of 2.427010176022.

## File-length distribution

The exact admitted raw file sizes, ordered, are:

`1,542 / 8,161 / 37,299 / 68,812 / 332,400` bytes.

The exact normalized unique sizes are:

`1,542 / 8,161 / 36,791 / 48,002 / 88,565` bytes.

Using nearest-rank quartiles, raw min/p25/median/p75/max are
`1,542 / 8,161 / 37,299 / 68,812 / 332,400`; mean is 89,642.8 bytes.
Normalized min/p25/median/p75/max are
`1,542 / 8,161 / 36,791 / 48,002 / 88,565`; mean is 36,612.2 bytes.

## Hidden mirrors and false independence

The audit treats a family as an upstream provenance unit, not a file.

The two Standard Ebooks objects, `8-typography.rst` and `9-metadata.rst`, have distinct raw and
normalized hashes but the same canonical `standardebooks/manual` upstream and the same declared
parent family. They are therefore **two files in one EN family**, not two independent families.

The two code objects remain separate families. DATA-227 proves both selected repositories are
canonical non-forks with no mirror URL. A fresh cross-object scan over the retained terminal bytes
found no raw or normalized exact duplicate and no cross-family mirror/copy edge. Under DATA-232-style
normalization and thresholds, the highest natural-text pair was the two Standard Ebooks siblings
(Jaccard 0.025219487003; containment 0.061399832355), far below 0.80/0.88. The HTTPX/Requests code
pair has ordinary token-shingle Jaccard 0 and code-skeleton Jaccard 0.021197007481, far below 0.82.

No independent-family credit is granted to a mirror, fork, vendored/generated derivative, sibling
file in one canonical repository, or translation with the same upstream document lineage unless
independent origin is positively demonstrated.

## Exact diversity gates for the next corpus

All gates are hard and conjunctive:

1. At least 6 independent terminal training-eligible external-real families in total.
2. At least 2 independent families in each stratum: Ukrainian text, English text, and code.
3. No family may exceed 25% of normalized unique selected bytes.
4. No family may exceed 60% of its own stratum's normalized unique selected bytes.
5. Shannon effective-family count, byte-weighted on normalized unique selected bytes, must be at least 4.0.
6. Cross-family exact/normalized duplicate, near-mirror, fragment-containment, or code-copy clusters
   that survive lineage resolution must be zero.
7. Only terminal rights-approved sources count. Failed or nonterminal candidates contribute zero.
8. Replay, repeated documents, mirrors, forks, or extra files from an existing family never increase
   unique-byte capacity or independent-family count.

These thresholds are consistent with a minimum 2/2/2 UA/EN/code structure and prevent four large
families plus nominal tiny files from masquerading as a balanced six-family corpus.

## Current gate failures

The terminal Wave-1 inventory fails family count (4 < 6), UA family count (1 < 2), EN family count
(1 < 2), top-family share (48.38% > 25%), effective-family count (2.427 < 4.0), and within-stratum
dominance (UA 100%, EN 100%, code 84.108% > 60%). It passes the observed cross-family
duplicate/mirror-edge gate.

Machine evidence: `evidence/data288/source_diversity_audit_v2.json`.
Audit identity: `daffef7aeb5da2994f71b8d7dad26c609704b8ee096c564a3f398be0d9d3423d`.
