# EVAL-303 Selection-Validation Composite V1

Worker: `EVAL-303-SELECTION-VALIDATION-COMPOSITE`

## Result

This authority composes the terminal Wave-2 selection-validation components into one immutable, selection-only authority. EVAL-303 does not duplicate source text; it commits a canonical hash/provenance membership registry bound to the exact component byte identities. The composite contains 10 external-real records: 8 Ukrainian records from EVAL-290 and 2 English records from the later non-empty EVAL-291 authority. The EVAL-292 code authority is bound exactly and remains an immutable zero-record, fail-closed component.

Composite selection identity:

`7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd`

Composite hash-only membership JSONL SHA-256:

`e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6`

## Terminal component bindings

- UA: EVAL-290 exact head `029514654829cebc149cff6fc1fea2a8ba4fa566`, dedicated run `32968339064` = `success`, set identity `c32320a706a283049e35eb537eb20a1e7f5865b86c24397c8b73d1e3d2014164`.
- EN: EVAL-291 exact head `fb268061300127b62cc2a262664b30c614559dac`, dedicated run `32967119568` = `success`, authority identity `727f229c091f86748a4eee9ea5aec72bb65347b68d6b687fabbf33166b0eca1e`. This is the later purpose-reserved non-empty successor; the earlier empty EVAL-291 branch is not used as the EN payload.
- Code: EVAL-292 exact head `2cbe2f2d9c74984baa69e49e520e2280fc76421b`, dedicated run `32967204390` = `success`, zero-record set identity `9fd52e879c388f06f0b103afa02d68678388867c81cfb0f27ddbf0ca18867054`.

The component workflow artifact digests, generated/committed authority hashes, rights bindings, reservations, provenance, revisions, source families, content hashes and rejected code candidates are all bound in the composite manifest.

## DATA-300 exclusion proof

The proof is run against frozen DATA-300 contract identity `07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5` at head `8ea7f830e50a23754d189dd4134f4afad76a7ee9`.

All 10 selected content SHA-256 values are disjoint from every frozen DATA-300 raw training SHA-256 and every text normalized SHA-256. The selected EN documentation Git blobs are also distinct from the HTTPX and Requests code Git blobs in training. HTTPX and Requests are shared source families, but the exact pinned objects, paths, Git blobs and SHA-256 values are different.

Proof identity:

`ac9a0e2c3beab26c0d664b0006b11ec9fd155fa78be9f46d56ecb3ed336f2621`

This proves exact byte/object separation only. It does not replace DATA-300 G07/G08 near-copy, mirror, connected-component or cluster decontamination, which still must run on the eventual Wave-3 materialization.

## Final-test firewall

EVAL-303 does not read the EVAL-233 final-test payload or any final-test outcome. It binds only the terminal component separation proofs and final-test provenance identities. No final-test bytes are copied into the composite. EVAL-290 reports zero exact content-hash and source-family overlap with its bound EVAL-233 final-test identity; EVAL-291 binds disjoint final-test provenance/source IDs and explicitly records that neither payload nor outcomes were read for construction.

## Usage boundary

The 10 bound member records are eligible only for tokenizer-configuration, checkpoint, model and hyperparameter selection. They are prohibited from tokenizer fitting, model updates, training, and final-test reporting. Code-aware selection remains unavailable because EVAL-292 has zero admitted code records. Any claim of non-empty code selection requires a successor authority with explicit evaluation rights and pre-training reservation.

This authority does not mutate the DATA-300 training plan and does not claim that the DATA-300 corpus is built, frozen, terminal or release-ready.

`LOCAL_FREE` only. No model training. No final-test outcome access.
