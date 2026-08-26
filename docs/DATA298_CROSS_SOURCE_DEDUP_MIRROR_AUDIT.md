# DATA-298 Cross-Source Dedup / Mirror Audit

Worker: `DATA-298-CROSS-SOURCE-DEDUP-MIRROR-AUDIT`

DATA-298 is an independent, LOCAL_FREE capacity audit over the externally observed UA/EN/code source objects. It does **not** grant source rights, admit a new corpus, replace DATA-230, or execute model training.

## Authority boundary

The inventory keeps source evidence states separate:

- `REGISTRY_TERMINAL`: the three DATA-229 registry objects.
- `DEDICATED_TERMINAL`: the two DATA-227 code objects whose dedicated exact-head admission workflow completed successfully, but which were not consumed by the earlier DATA-229 cutoff.
- `PROBE_NONTERMINAL`: the two DATA-228 UA/EN probe objects. They are observable for redundancy analysis but are not promoted by DATA-298.

Declared pre-dedup capacity boundaries are therefore:

| scope | source IDs | declared families | declared capacity bytes before dedup |
| --- | ---: | ---: | ---: |
| canonical_registry | 3 | 2 | 173,358 |
| terminal_evidence | 5 | 4 | 183,061 |
| all_observed | 7 | 6 | 207,771 |

The seven exact source objects total 480,273 raw bytes before any exact-byte collapse. The all-observed 207,771-byte figure is an **upper bound**, not a unique-capacity claim.

## Deduplication semantics

DATA-298 deliberately reuses DATA-232 normalization and overlap thresholds rather than creating a weaker parallel definition. It detects:

- identical raw bytes;
- identical content after DATA-232 normalization;
- aliases pointing at the same canonical origin identity;
- high-Jaccard near copies;
- document-fragment containment;
- code forks/copies through the DATA-232 identifier/string/number/comment-insensitive code skeleton;
- repeated publisher edge boilerplate.

Publisher boilerplate is evidence for review but is not capacity-collapsing by itself. A shared site/publisher header must not make otherwise unrelated documents duplicates.

All capacity-collapsing pair edges are converted into connected components. This prevents A≈B and B≈C chains from being counted as three independent objects merely because A and C are not a direct threshold match.

## Capacity rule

Within every connected duplicate cluster, DATA-298 counts at most the **largest declared-capacity member**. The resulting field is:

`conservative_unique_capacity_bytes_after`

That field, and not source count, raw byte sum, publisher count, or pre-dedup capacity, is the permitted capacity metric for the audited scope.

Cross-family collapse edges also merge source families for `effective_independent_family_count`. Multiple files from one publisher remain one declared source family unless independent provenance says otherwise.

## Live verification

The exact-head workflow:

1. checks out the exact PR SHA on Python 3.11.16;
2. runs the universal LOCAL_FREE bootstrap;
3. runs adversarial exact/normalized/fragment/boilerplate/code-fork tests;
4. retrieves all seven exact upstream objects;
5. verifies immutable raw SHA-256 or Git blob identity and exact raw byte count;
6. computes a deterministic hash-only report;
7. verifies that deduplication never increases capacity;
8. uploads the report and execution environment as immutable CI evidence.

No raw source text is written to the report or committed by DATA-298. If a supposedly immutable object changes identity, acquisition fails closed instead of silently accepting replacement bytes.

## Interpretation

Before the live DATA-298 report is green, the safe claims remain:

- DATA-229 canonical registry: 173,358 declared normalized/admitted bytes across 2 declared families;
- DATA-227 adds 9,703 bytes of dedicated-terminal code evidence across 2 additional declared families, not yet converged into DATA-229;
- DATA-228 contributes 24,710 normalized probe bytes across 2 candidate families, but its current exact-head probe is nonterminal/failing and therefore cannot be counted as terminal admitted capacity.

After a green exact-head DATA-298 workflow, use each scope's `conservative_unique_capacity_bytes_after` and `effective_independent_family_count` for redundancy-adjusted capacity reporting. DATA-298 still does not supersede a future terminal DATA-230 corpus registry.
