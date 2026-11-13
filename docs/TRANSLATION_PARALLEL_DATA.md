# English–Ukrainian translation data

The initial corpus uses the November 7, 2026 Tatoeba weekly export. It joins English and Ukrainian detailed text exports through the direct English to Ukrainian links export. The materializer verifies the compressed originals before parsing. It keeps each selected sentence unchanged and attaches both contributors, sentence IDs and original sentence URLs to every pair.

Tatoeba publishes these text and link exports under CC BY 2.0 France. The generated bundle includes the license URL and an attribution notice. It retains the source-specific attribution in each row. [Downloads](https://tatoeba.org/en/downloads) · [Terms](https://tatoeba.org/en/terms_of_use) · [Contributor guide](https://en.wiki.tatoeba.org/articles/show/how-to-be-a-good-contributor-in-tatoeba) · [License](https://creativecommons.org/licenses/by/2.0/fr/)

Place the exact compressed sources in a local directory. The program lists their official URLs, sizes and SHA-256 values in the EXPORTS mapping. The URLs are rolling exports, so keep the downloaded originals to reproduce this snapshot.

    python -m twelve_six.data.translation_parallel build --sources /path/to/pinned-exports --out /path/to/new-bundle
    python -m twelve_six.data.translation_parallel verify --directory /path/to/new-bundle --manifest-sha256 127d1b5e62899049b9f4060242e6c249a58c08eb6ed6c433a9d97f6bc1a9136d

The default bundle contains 20,000 train, 1,000 validation and 1,000 test pairs. The split groups connected components of shared normalized sentence text before selection, so alternate translations and normalized text duplicates stay in one split. Training reads the frozen train member by manifest SHA-256. It can use either translation direction by swapping the source and target in a row without changing its pair ID or split.

The bundled test partition measures performance on held-out Tatoeba sentences. FLORES-200 devtest can provide separate evaluation in a different domain under CC BY-SA 4.0. Keep any FLORES references and attribution separate from the training data and from model selection on the Tatoeba validation split. [FLORES-200](https://github.com/facebookresearch/flores/tree/main/flores200) · [Release licensing](https://github.com/facebookresearch/flores#license) · [License](https://creativecommons.org/licenses/by-sa/4.0/)

The optional evaluator accepts the original archive from the publisher's FLORES-200 download. The local devtest builder screens both languages in all 1,012 aligned rows against exact normalized sentences in the pinned Tatoeba training split. It picks 100 rows by a stable hash of their original line identities and records the source member hashes and article URLs. The November 7 snapshot had zero matches in the screen.

    python -m twelve_six.data.translation_parallel flores-devtest --archive-directory /path/to/archive-directory --train-directory /path/to/new-bundle --train-manifest-sha256 127d1b5e62899049b9f4060242e6c249a58c08eb6ed6c433a9d97f6bc1a9136d --out /path/to/new-evaluation-bundle

The expected evaluation JSONL SHA-256 is ed090446f653dbb6a1555c7deb52849b724faa374f7b0650f7a6b3a182f1088d.
