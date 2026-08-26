# NEXT100-087 independent communication dataset audit

Worker: `NEXT100-087-COMM-DATA-INDEPENDENT-AUDIT`

Target: POSTBASE-352 exact head `d83fe9f7227112615da1f8f6e7a10f56531dbb35` from PR #437.

Target producer evidence: CI run `32997692106` completed `success` on that exact head. The audit does not inherit that producer result as its own authority.

## Frozen dataset reviewed

The audit reviews the unchanged POSTBASE-352 seed only:

- 20 project-authored dialogues;
- train: 12;
- selection: 4;
- final: 4;
- English and Ukrainian in every split;
- foreign-model outputs: 0;
- manifest SHA-256: `51a927c40b4274f8b8f992b8dd83b4dbddac1e925a45834832a79ee6be18d3d6`;
- train SHA-256: `ddafe61ce3255dd30d207ec1ee811efa59a2da37288368a9bbc3fa0602cb2ba7`;
- selection SHA-256: `e36f7c560c44fd2812935b5382dd628fabbd7af0e79c080c295d1332de13309f`;
- final SHA-256: `f50262994089f276fb7d3f4c644180d0854273b51d7f7920aca1d2048031d039`.

The audit code never writes these files. Tests snapshot all four SHA-256 identities before and after the audit and require byte identity.

## Independent review findings

Role alternation: PASS. Every dialogue starts with `user`, alternates exactly, and ends with `assistant`. System/tool/other roles are absent.

Language: PASS for the frozen seed. Language tags are `en` or `uk`, every split contains both, and an independent Unicode-script check rejects obvious English/Ukrainian relabeling.

Provenance: PASS. Every row is `project_authored`, source `project:postbase352-manual-v1`, rights `project_owned`, `foreign_model_output=false`, and has no synthetic authority. Message-content SHA-256 is independently recomputed.

Answer quality: PASS for this narrow 20-row seed. The answers are simple direct facts, arithmetic, transformations, summaries, clarification questions, structured steps, and context-carryover responses. The manual review found no incorrect answer in the frozen bytes. This is not a learned-model quality claim.

Formatting: PASS. Message text is non-empty, has no edge whitespace, and does not inject `User:`, `Assistant:`, or `System:` role prefixes at line starts. The manifest formatter remains `postbase352.plain-role-v1` with no special tokens.

Duplicate families: PASS. All 20 `family_id` values are globally unique under the independent audit. This is stricter than merely keeping one family inside a single split.

Near duplicates: PASS. The independent auditor recomputes normalized character-5-gram Jaccard across all record pairs and requires every score below the frozen `0.85` threshold.

Train/selection/final separation: PASS. The root must contain exactly `manifest.json`, `train.jsonl`, `selection.jsonl`, and `final.jsonl`; split files must be distinct regular non-symlink filesystem objects; each row must name its physical split; the frozen split hashes and counts must match.

Tokenizer compatibility: PASS at the dataset boundary. The logical profile remains `s0-byte-v1`, vocabulary 256, UTF-8 bytes, no added special tokens, and no Base chat template. Exact config/vocab hash equality remains a separate POSTBASE-253 `TokenizerCompatibility` gate before any later training authority; the independent tests prove exact hash drift is rejected when that authority pair is supplied.

Foreign-model output count: PASS at 0. A manifest/count mismatch fails closed.

Hidden-reasoning field: PASS. The required `no_hidden_reasoning=true` gate is present on every row. The independent auditor additionally rejects explicit `analysis`, `reasoning`, `chain_of_thought`, `hidden_reasoning`, or `scratchpad` fields anywhere in a row.

Base-training firewall: PASS. `base_corpus_evidence`, `canonical_base_training_eligible`, `training_authorized`, `selection_for_training`, `final_for_training`, and `final_for_selection` are all frozen `false`.

## Important trust boundary found by the audit

`data_contract.validate_dataset()` is a structural/provenance/attestation validator. It does not itself understand whether a newly rewritten natural-language answer is factually correct. A coordinated attacker can change an answer, recompute that row's content hash, recompute the split hash, update the manifest, and still satisfy the generic attestation contract if `answer_verified=true` is dishonestly retained.

The independent audit therefore does not treat a self-consistent rewritten manifest as the already reviewed POSTBASE-352 seed. It hard-binds the exact reviewed manifest and split identities above. Any answer/language/format content mutation must receive a new audit identity and manual review; it cannot silently inherit this audit result.

This is intentional separation of responsibilities rather than a claim that deterministic code can semantically judge arbitrary natural-language answer quality.

## Adversarial fixture mutations

The scoped regression suite attacks:

- broken role alternation;
- English/Ukrainian relabeling;
- wrong answer with recomputed content/split/manifest hashes;
- role-prefix formatting injection;
- same-split duplicate family IDs;
- near duplicates;
- split relabeling;
- provenance source substitution;
- foreign-model count mismatch;
- hidden-reasoning field insertion and gate disablement;
- canonical Base-training eligibility enablement;
- logical tokenizer profile drift;
- exact Base tokenizer config-hash drift.

Each attack must be rejected by either the POSTBASE-352 contract gate or the independent frozen-seed audit gate. No fixture mutation is written back to the dataset.

## Truth boundary

This audit establishes dataset-contract integrity and a manual review of the exact 20 frozen seed examples. It does not authorize SFT/RLHF/DPO/RL, does not execute optimizer updates, does not call a foreign model, does not modify canonical Base, and does not claim broad assistant quality or production readiness.

`LOCAL_FREE` only.
