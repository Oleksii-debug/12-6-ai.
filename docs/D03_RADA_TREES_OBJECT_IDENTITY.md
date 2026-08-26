# D03 Rada_Trees immutable archive object discovery

## Purpose

This is the first executable successor to PR #638's Rada_Trees acquisition probe. It solves only the immutable-object seam: prove which exact large-file object is referenced by the pinned Hugging Face dataset commit before downloading hundreds of megabytes.

The source remains `uacorpus/Rada_Trees` at exact revision `1b994a5804dcda122721e8d33a03fd172cf8d867`. The primary acquisition candidate is `Rada_Trees.7z`; `rada_xtag_texts.7z` remains a zero-credit annotation/derivative hold.

## Discovery algorithm

`tools/discover_d03_rada_trees_object_identity.py`:

1. verifies the exact parent PR #638 probe config by Git blob SHA-1;
2. rejects branch names or shortened revisions;
3. queries the Hugging Face dataset tree at the exact 40-hex revision;
4. requires both archive paths exactly once, positive byte sizes and exact Git blob IDs;
5. requires at least one immutable large-file identity (Xet hash or LFS SHA-256);
6. makes a no-redirect request to the exact-revision `resolve` URL and captures `X-Xet-Hash` when supplied;
7. fails closed if tree and resolve Xet identities disagree;
8. writes a deterministic self-hashed metadata report.

The no-redirect request is intentional: this stage must not accidentally follow the large-file redirect and download the 536 MB archive merely to discover object metadata.

## Truth boundary

A successful report means only `IMMUTABLE_OBJECT_METADATA_PINNED_DOWNLOAD_BODY_NOT_EXECUTED`.

It does not prove archive content SHA-256 unless an LFS SHA-256 is explicitly returned. It does not inventory archive members, establish member-level provenance, grant family independence, grant source-capacity credit, fit a tokenizer, run an optimizer step, or authorize paid compute.

The next stage is an explicitly bounded download of the exact primary object, streaming SHA-256 verification, safe 7z member inventory, and plain-text-versus-annotation classification before any capacity arithmetic.

## Local verification

The regression suite is network-free:

`python -m unittest tests/test_d03_rada_trees_object_identity.py`

The live discovery command is intentionally separate:

`python tools/discover_d03_rada_trees_object_identity.py discover --report evidence/d03/rada_trees_object_identity.json`

Because repository Actions are currently runner-saturated, this package does not add another dedicated workflow. Generic CI remains the regression authority; the live network report must be treated as nonterminal until its exact execution evidence is recorded.
