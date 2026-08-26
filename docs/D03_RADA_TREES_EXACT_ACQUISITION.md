# D03 Rada_Trees exact archive acquisition

## Purpose

This layer removes the remaining manual transport-provenance gap between the immutable Hugging Face object snapshot produced by PR #638 and the deterministic archive/member inventory in PR #697.

`tools/download_d03_rada_trees_from_hf_snapshot.py` acquires only `Rada_Trees.7z` from the exact dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`. It consumes the self-hashed parent object snapshot through the canonical HF-snapshot validator, so archive size, Git blob OID and Xet identity are not operator-supplied.

## Transport binding

The downloader starts from one hard-coded immutable resolve URL. Before it follows the first redirect, it requires the documented HTTP 302 Xet handoff and requires `X-Xet-Hash` to equal the primary archive Xet identity in the parent snapshot. If `X-Repo-Commit` or `X-Linked-Size` are present, they must equal the exact dataset revision and snapshot byte size.

Redirects and the final response must remain on HTTPS Hugging Face/HF/Xet storage origins. Content encoding is rejected, a present `Content-Length` must equal the exact snapshot size, and streaming aborts if the body is either shorter or longer than that exact size. The destination is written as a new `.partial` file, fsynced, and atomically renamed only after the byte count is exact. Existing archive or partial files are never overwritten.

The resulting transfer report binds:

- parent object snapshot identity;
- exact dataset revision;
- exact resolve URL;
- validated first redirect evidence;
- Git blob OID;
- Xet file identity;
- exact downloaded byte count;
- computed full-content SHA-256.

The computed SHA-256 is then the input to `run_d03_rada_trees_inventory_from_hf_snapshot.py`. This separates transport acquisition from archive decoding and member classification while keeping the evidence chain exact.

## Execution

```text
python tools/download_d03_rada_trees_from_hf_snapshot.py \
  --object-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --archive-output evidence/d03-rada-trees/download/Rada_Trees.7z \
  --report-output evidence/d03-rada-trees/exact-acquisition-v1.json
```

After acquisition, pass the reported `archive.content_sha256` to the existing HF-snapshot inventory bridge.

## Truth boundary

A successful acquisition means only that exact-revision transport was bound to the parent Xet identity and that the resulting archive bytes were counted and SHA-256 hashed. It does not mean the archive members are safe or admissible training text.

`training_authorized_bytes` remains `0`. No tokenizer fit, model training, optimizer update, corpus admission, final-test access, GPU provisioning or paid compute is authorized. The next gate is deterministic member inventory, followed by plain-text classification, period/member provenance, rights, Ukrainian language/quality/privacy checks, global lineage deduplication, evaluation decontamination and family-cap recomputation.
