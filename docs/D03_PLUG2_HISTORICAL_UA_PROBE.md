# D03 PluG/PluG2 historical Ukrainian acquisition probe

## Decision

Treat `Dandelliony/pluperfect_grac` as a high-leverage Ukrainian discovery source, but credit exactly zero training bytes until rights, member provenance, quality strata, lineage deduplication, and evaluation decontamination are terminal.

Pinned upstream authority:

- repository: `Dandelliony/pluperfect_grac`
- commit: `27e503ecc2b553d52bfc121b8320144bb25294d8`
- root tree: `99dec65503cdb9a0b0865901145b7a4311a8bdd2`
- pinned roots: `PluG_metadata.psv`, `PluG_texts`, `PluG2_metadata.psv`, `PluG2_texts`

The upstream README at that commit reports PluG as 42,000 files / 58,676,313 tokens covering 1816–1954, and PluG2 as 73,900,596 tokens. Those are discovery statistics only. They are not source-capacity bytes, tokenizer tokens, unique causal-loss positions, or training authorization.

## Why this source is useful

The current Research Corpus V1 plan requires a 45/35/20 Ukrainian/English/code balance after global deduplication. Ukrainian is the binding stratum. PluG is potentially large enough to materially reduce that deficit while diversifying away from the modern parliamentary Rada_Trees family.

However, diversity is only useful if it is controlled. PluG contains historical literature, OCR-derived text, translations, and older orthographies. PluG2 expands Western Ukrainian material from roughly the 1880s–1920s and explicitly warns that this historical variant can complicate models oriented toward the modern standard.

## Rights boundary

The pinned README says only that the corpus is available under `CC-BY`; it does not state an exact CC-BY version in the text pinned by this probe. Therefore:

- do not infer CC BY 4.0 from the license family name;
- do not infer dataset rights from the journal article's publication license;
- do not admit any member until the exact dataset-license terms and attribution payload are pinned;
- evaluation/final-test use remains separately unauthorized.

## One-lineage rule

PluG and PluG2 are versions/expansions of one underlying corpus lineage and receive at most one family credit. Repository versions, mirrors, formatting changes, GRAC-derived surfaces, or annotations cannot create additional independent-family credit. Exact and near lineage deduplication against the live corpus is mandatory before any capacity claim.

## Quality strata

Do not mix all text into a homogeneous Ukrainian bucket. Before admission, each accepted record must carry enough provenance to support at least:

- period/date stratum;
- orthography stratum;
- OCR quality status;
- original-vs-translation status when available;
- source/author attribution;
- language-quality/privacy result;
- lineage cluster identity.

The default for PluG2 is HOLD until an explicit historical-orthography cap or measured ablation justifies its proportion in a modern Ukrainian model.

## Metadata-only live probe

`tools/probe_d03_plug2_github_tree.py` queries the exact pinned commit/tree and emits no corpus text. It fails closed if GitHub reports a truncated recursive tree. The snapshot contains only root identities, text-blob counts/sizes, execution-boundary flags, and a deterministic SHA-256 self-hash.

A successful metadata snapshot is still not admission evidence. The successor must bind metadata rows to members, pin exact rights, download bounded members, verify SHA-256, perform quality/privacy and cross-source deduplication, then run evaluation decontamination and 45/35/20 family-cap recomputation.

## Truth boundary

LOCAL_FREE only. No corpus materialization, tokenizer fit, optimizer update, model training, paid compute, learned-20M claim, or Research Corpus V1 promotion is authorized by this probe.
