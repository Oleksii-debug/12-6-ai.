# D03 Rada_Trees secondary full text stream scan V1

## Purpose

This package closes the exact next scientific seam after the terminal secondary-role
probe on PR #820. It does not create another archive transport, classifier, dedup
engine, corpus registry, tokenizer path, or training campaign.

The exact immutable object is:

- dataset: `uacorpus/Rada_Trees`;
- revision: `1b994a5804dcda122721e8d33a03fd172cf8d867`;
- archive: `rada_xtag_texts.7z`;
- compressed bytes: `697,768,591`;
- content SHA-256:
  `737fa5df061c55555a8c761023559a206cbcabf0299ae52d1ccae387f685803e`;
- terminal listing: `8,782` members / `19,711,802,635` unpacked bytes;
- `.txt` members: `4,391`;
- terminal listing identity:
  `9e5bad28b2455a0b1682b7c91d0f81f4113a928223898570c9800cf8af7ad15a`.

PR #820's bounded probe classified 16/16 sampled `.txt` members as
`PLAIN_TEXT_CANDIDATE`, but it intentionally granted zero capacity because all
4,391 text members had not been scanned.

## Reuse boundary

`tools/scan_d03_rada_trees_secondary_full_txt_v1.py` reuses:

- the existing Rada_Trees 7z lister and canonical member-path checks;
- the existing exact secondary archive member streamer;
- the existing member classifier and its UTF-8-SIG -> Windows-1251 decode order;
- the existing classifier's derivative/markup/tabular/empty/NUL holds.

It does not change those semantics.

## Full scan contract

Before scanning, the tool requires the exact canonical terminal #820 evidence
identity and recomputes that evidence's self-hash. It binds the successful
workflow run, job, artifact digest, exact source revision/object identities,
archive content SHA, complete listing identity, member counts, and zero-credit
claim boundary.

The local archive is independently required to match the exact compressed byte
count and content SHA-256.

The scanner then:

1. recreates the exact complete 7z listing;
2. recomputes the terminal listing identity;
3. selects exactly all 4,391 `.txt` members in canonical path order;
4. streams each selected member without full archive extraction;
5. requires streamed bytes to match the listed member size;
6. reuses the canonical classifier;
7. records only text-free hashes, sizes, encoding labels, path year hints, and
   aggregate classification facts;
8. collapses only exact raw-content duplicates among
   `PLAIN_TEXT_CANDIDATE` members.

The resulting exact-duplicate-collapsed raw byte total is still a source-role
diagnostic. It is not global-deduplicated corpus capacity and is never converted
to training exposure by this tool.

## Truth boundary

Even after a successful complete scan:

- training-authorized bytes remain `0`;
- unique causal-loss positions remain `0`;
- tokenizer fit remains unauthorized;
- optimizer/model training remains not executed;
- final-test payload access remains false;
- paid compute remains unused;
- Research Corpus V1 remains unreleased.

Required successors remain period/session provenance, rights-scope
revalidation, language/quality/privacy, project global exact/near/fragment/
lineage dedup including overlap with ParlAment/GRAC, reserved-evaluation
decontamination, and the already-merged balance/family-cap gate.

## Execution strategy

The initial PR contains no new workflow and does not modify `ci.yml`. Shared CI
qualifies the deterministic scanner and adversarial tests first. A network-capable
LOCAL_FREE execution may then use the exact #820 archive/evidence. Because the
archive is about 19.7 GB unpacked and the scan deliberately avoids full extraction,
runtime duration is treated as measured evidence rather than guessed in advance.

The full result must be retained as text-free evidence and independently verified
by its report identity before it can feed the provenance/privacy/global-dedup
successor.
