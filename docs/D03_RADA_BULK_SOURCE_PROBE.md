# D03 Rada bulk source probe

Status: `PROBE_ONLY / ZERO_TRAINING_AUTHORIZATION`

## Why this exists

The live DATA-287 registry already admits one bounded object from the family
`ua.rada.open-data.laws-texts` with 88,565 normalized bytes. The official
Verkhovna Rada Open Data Portal exposes the same dataset as a frequently
updated bulk ZIP containing thousands of primary-law HTML objects. That makes
bulk acquisition a potentially high-leverage Ukrainian-capacity path, but the
mutable archive cannot be treated as an immutable training source by URL alone.

This package therefore adds an acquisition/inventory boundary, not a corpus
admission.

## Discovery observation

Observed on 2026-08-26 from the official portal page:

- dataset id: `laws-texts`;
- publisher: Staff of the Verkhovna Rada of Ukraine;
- portal publication timestamp: `2026-08-07T16:58:14+03:00`;
- displayed file count: `5926`;
- displayed `texts.zip` bytes: `46,682,933`;
- displayed MD5: `06f239fd182e580ce22ab00dce867e31`;
- update frequency: frequent / mutable upstream.

These fields are discovery observations only. The probe fails closed if the
archive no longer matches them unless `--accept-current-upstream` is supplied
to create a fresh observation for successor review.

## Rights boundary

The portal's terms state that open data may be copied, published, distributed,
used and reused, including commercially, with source attribution. The portal
also states CC BY 4.0 as its default content license unless otherwise specified.
DATA-287 already records project model-training permission for the bounded Rada
snapshot in this same source family.

This probe does **not** automatically extend that bounded admission to every
object in the bulk archive. A successor source authority must re-bind exact
archive bytes, purpose rights, attribution obligations and exclusions before
any bulk records receive training credit. Evaluation remains separately gated.

## Probe behavior

`tools/probe_d03_rada_bulk_source.py`:

1. downloads `texts.zip` or reads a supplied local archive;
2. enforces an archive-size ceiling;
3. optionally requires the discovery-time byte count and MD5;
4. rejects path traversal, symlinks, oversized entries and duplicate canonical
   basenames;
5. selects only `d[0-9]+.htm` canonical objects;
6. SHA-256 hashes every selected object;
7. emits a deterministic inventory identity over sorted basename/size/SHA-256
   tuples;
8. reports exact raw archive/inventory facts while leaving every downstream
   data gate `NOT_RUN` and training-authorized bytes at zero.

Example strict observation:

```bash
python tools/probe_d03_rada_bulk_source.py \
  --output evidence/d03_rada_bulk/source_probe.json
```

If the frequently updated upstream has changed, record a new observation
without pretending the old identity remains current:

```bash
python tools/probe_d03_rada_bulk_source.py \
  --accept-current-upstream \
  --output evidence/d03_rada_bulk/source_probe.json
```

A report produced with `--accept-current-upstream` is not terminal authority;
its SHA-256 must be pinned and reviewed by a successor.

## Required downstream sequence

The acquisition observation must be followed by exact source-manifest sealing,
canonical HTML extraction/normalization, quality and privacy filtering, global
cross-source exact/near deduplication, evaluation decontamination,
balance/diversity and family-cap retest, deterministic split/shard/packing,
unique causal-loss accounting, tokenizer-fit authorization, and only then a
learned-20M compute decision.

Until those gates close, this package authorizes zero training bytes, zero
tokenizer-fit bytes and no paid compute.
