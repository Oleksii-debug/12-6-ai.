# D03 Rada_Trees exact archive acquisition

## Purpose

This layer closes the transport-provenance gap between the immutable Hugging Face object snapshot from PR #638 and deterministic archive/member inventory in PR #697.

`tools/download_d03_rada_trees_from_hf_snapshot.py` acquires only `Rada_Trees.7z` from exact dataset revision `1b994a5804dcda122721e8d33a03fd172cf8d867`. Archive size and Git object identity come from the self-hashed parent snapshot. Xet and full-content SHA-256 are independently pinned in the #697 inventory config.

## Pinned identity

The primary file authority is:

- Xet identity `a31d24710d417246fb7e48028baaf6b9efb9a199983d78264f7997c69e42a801`;
- full content SHA-256 `5e53939cd255276c58190569aebfaa6c90fb085fb10063e3e5f661747749719d`.

This provider-published hash is pinned before transfer. It is not itself evidence that this project has downloaded and independently rehashed the file.

## Transport binding

The downloader starts from one hard-coded immutable resolve URL. Before following the first redirect, it requires the HTTP 302 Xet handoff and requires `X-Xet-Hash` to equal both the parent snapshot and the independently pinned Xet authority. If `X-Repo-Commit` or `X-Linked-Size` are present, they must equal the exact dataset revision and snapshot byte size.

Redirects and the final response must remain on HTTPS Hugging Face/HF/Xet storage origins. Content encoding is rejected, a present `Content-Length` must equal the exact snapshot size, and streaming aborts if the body is shorter or longer than that exact size.

The destination is written as a new `.partial` file while SHA-256 is computed. A wrong full-content digest fails closed and the partial is removed. The file is fsynced and atomically renamed only after both exact byte count and pinned SHA-256 equality pass. Existing archive or partial files are never overwritten.

The transfer report binds:

- parent object snapshot identity;
- exact dataset revision;
- exact resolve URL;
- validated first redirect evidence;
- Git blob OID;
- pinned Xet file identity;
- exact downloaded byte count;
- expected pinned content SHA-256;
- independently computed downloaded content SHA-256;
- explicit equality of the two hashes.

## Execution

```text
python tools/download_d03_rada_trees_from_hf_snapshot.py \
  --object-snapshot evidence/d03-rada-trees/hf-object-identity-v1.json \
  --archive-output evidence/d03-rada-trees/download/Rada_Trees.7z \
  --report-output evidence/d03-rada-trees/exact-acquisition-v1.json
```

After acquisition, run the HF-snapshot inventory bridge. The bridge takes no operator-supplied content hash; it derives the expected digest from the pinned config and independently rehashes the archive before extraction.

## Truth boundary

A successful acquisition means only that exact-revision transport was bound to the parent object snapshot and the downloaded archive matched the independently pinned byte count/Xet/content identities. It does not mean archive members are safe or admissible training text.

`training_authorized_bytes` remains `0`. No tokenizer fit, model training, optimizer update, corpus admission, final-test access, GPU provisioning or paid compute is authorized. The next gate is deterministic member inventory, followed by plain-text classification, period/member provenance, rights, Ukrainian language/quality/privacy checks, global lineage deduplication, evaluation decontamination and family-cap recomputation.
