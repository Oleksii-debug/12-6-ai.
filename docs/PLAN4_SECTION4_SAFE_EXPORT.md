# Plan 4 / Section 4 — Safe weight format and transfer export

## Canonical authority and reuse

`tools/plan4_safe_export.py` is a narrow, versioned transfer adapter. It
**does not** serialize a second copy of weights or own checkpoint semantics:
it invokes the incumbent verified D05 checkpoint loader and
`checkpoint.hf_export.export_hf_directory`. Weight bytes remain SafeTensors,
not Python pickle or executable deserialization. The existing HF-style export
continues to report that Transformers architecture and inference parity are
NOT CLAIMED / NOT TESTED.

### Output (immutable, create-only, atomic)

- `hf/model.safetensors` — exact incumbent weight bytes, verified by incumbent hashes;
- `hf/config.json` — canonical bound model spec/checkpoint/tokenizer identity;
- `hf/12-6-checkpoint-manifest.json` and incumbent attestation/checksum/parity request;
- `tokenizer.json` — exact canonical S0 UTF-8 byte tokenizer config;
- `tokenizer-vocab.json` — complete 256-entry exact token-ID mapping;
- `12-6-safe-bundle.json` — versioned checkpoint/provenance, SHA-256 of tokenizer
  files, HF model config, SafeTensors and incumbent export attestation;
- `12-6-safe-bundle.sha256` — exact self-hash of canonical bundle manifest.

Only canonical `s0-byte-v1` is accepted in v1; wrong tokenizer identity,
corrupt or missing files, non-canonical JSON, symlinks, unexpected artifacts,
weight tampering, mismatched configuration, independent hash pins, and
concurrent destination writes fail closed. The original checkpoint is never
modified. Staging is private and publication is atomically no-replace on
supported Windows/Linux systems; staging is strictly cleaned on ordinary
failures. A simulated interrupted publish is covered by tests.

### Contract and evidence

API:
`export_safe_bundle(checkpoint_dir, destination)`;
`verify_safe_export(destination, expected_checkpoint_id=...,
expected_manifest_sha256=...)`.

The independent digest pin is supplied by a trustworthy consumer; bundle
self-hash alone detects accidental corruption but is **not a digital signature**.
The compatibility namespace is explicitly HF_STYLE_ONLY: this does not
grant alternate backend parity, trained/champion authority, external provenance
approval, final-test access, paid cloud/GPU authorization or whole-product
release. Production tokenizer/model substitution must pass a new versioned
compatibility contract, not silently change S0 semantics.

LOCAL_FREE fixture qualification:
```sh
python -m pytest -q tests/test_plan4_safe_export.py \
  tests/test_hf_export.py tests/test_hf_export_transactional.py \
  tests/test_inference_runtime_plan4.py
python -m py_compile tools/plan4_safe_export.py
ruff check tools/plan4_safe_export.py tests/test_plan4_safe_export.py
```
