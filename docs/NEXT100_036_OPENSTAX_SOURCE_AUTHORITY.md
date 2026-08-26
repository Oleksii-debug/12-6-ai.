# NEXT100-036 OpenStax Source Authority

Worker: `NEXT100-036-DATA-EN-OPENSTAX`

Terminal decision: **REJECT**

## Bounded candidate

- Provider: OpenStax
- Title: `Physics`
- Authors: Paul Peter Urone; Roger Hinrichs
- Publication release date: `2020-03-26`
- Edition identity: `Physics|published=2020-03-26|source_revision=dfdfd7a5356ecdd42e504de3df50d9153e33ea49`
- No numeric edition number is asserted because the authoritative title page does not label this release with one.
- Official source repository: `openstax/osbooks-physics`
- Pinned commit: `dfdfd7a5356ecdd42e504de3df50d9153e33ea49`
- Pinned tree: `b99750d59120e03f65f15da8ae012c5d5bdcfaa7`
- Bounded text object: `modules/m54467/index.cnxml`
- Object Git blob: `a1b45a7c27067e950a112a3746b911c5a620c01c`
- License Git blob: `409b0eada6569b9844b8ab1958c7e2f6d1359e3f`
- Source locator SHA-256: `bf79459b52cf90ab50abb8dba70385bfaa8fbe36463a1ad079174b8bec623817`
- Family: `en.openstax.physics`
- Family identity SHA-256: `fc9e09155fc316c2199bc51d403791bc791ba8271b7aa419d0c043d819e3fe4a`

The bounded object was selected only to establish a deterministic candidate identity. It is not admitted to the training corpus.

## License and attribution

The pinned repository license is `CC-BY-4.0` (`Creative Commons Attribution 4.0 International`). This candidate therefore avoids NC and ND material.

For reuse of the CC BY text, attribution must preserve the applicable creator/licensor attribution, copyright/license/notices and source URI where reasonably practicable, identify modifications, and include or reference CC BY 4.0. The current Physics title page additionally directs redistribution attribution to Texas Education Agency (TEA), identifies the Texas Gateway source, identifies OpenStax changes, and requires the title-specific access-for-free notice on redistributed physical pages or digital page views.

Images, media, iframe content, and separately permissioned art are outside this candidate. The title page explicitly warns that some art is used under permission and can carry additional limitations.

## Training decision

`REJECT_NO_OPENSTAX_PERMISSION`.

The current OpenStax `Physics` title page states that OpenStax permission is required for LLM/generative-AI training or ingestion. No separate OpenStax permission covering this exact title/revision and this project's model-training use is evidenced.

This decision is fail-closed. It does not attempt to resolve a general legal conflict between a Creative Commons grant and a provider's later AI-specific notice. For this repository, the missing explicit provider permission is sufficient to reject model-training admission. DATA-21/22 previously used the same fail-closed boundary for an OpenStax candidate.

CC BY text redistribution is recorded separately as `ALLOWED_WITH_ATTRIBUTION_FOR_CC_BY_TEXT`; that redistribution permission does not bootstrap model-training admission.

## Normalization, quality, privacy, and dedup

No training payload is materialized after the rights rejection. Consequently:

- raw training-payload SHA-256: not computed;
- normalization: `NOT_RUN_RIGHTS_REJECT`;
- normalized bytes: `0`;
- quality admission: not credited;
- privacy-clean admission: not asserted;
- exact/near dedup against admitted corpus: not run;
- admitted capacity delta: `0`;
- admitted family-credit delta: `0`.

A deterministic CNXML text-only normalization policy is preregistered in the machine authority solely for a future retest after rights are cleared. It excludes figures/media/images/iframes and performs deterministic XML-order extraction and byte normalization without OCR, translation, paraphrase, or generated text.

## Live registry binding

This authority was stacked on the live DATA-287 external snapshot registry:

- PR: `#404`
- head SHA: `b0523ccbc4b957615aac849d476cfa851be87578`
- registry path: `data/registry/external_snapshots.v2.json`
- committed registry identity field: `917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c`
- sources: `5`
- independent source families: `4`
- English independent families: `1`
- unique normalized bytes: `183061`

This rejected OpenStax candidate changes none of those counts. It is not evidence that the existing corpus, English slice, or OpenStax itself is representative.

## Retest gate

Retest only if immutable evidence is available for one of the following:

1. OpenStax grants permission covering model training/ingestion for this exact title/revision and intended project use, including any redistribution needed by the project; or
2. authoritative OpenStax terms materially change and a new rights review confirms the training use.

Until then, terminal training status is `REJECT`.
