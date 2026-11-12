"""Build source notices for the local Section 18 training handoff."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s17_physical_bundle_v1 as s17

SCHEMA = "12-6.plan2-s18-source-notices.v1"
MAX_NOTICE_BYTES = 20_000
S15_D03 = "data/external/snapshots/plan2-s15-d03-ua-v1"
S15_SOURCE_AUTHORITY = "configs/data/plan2_s15_d03_physical_source_candidates_v1.json"
ARXIV_RIGHTS = "configs/data/d03_common_pile_arxiv_abstracts_source_rights_v1.json"
LANGUK_CARD = (
    "https://huggingface.co/datasets/lang-uk/court-decisions-uk/raw/"
    "2dcac4c941b87bf9c242bdc919cef4b40f4a4813/README.md"
)
EXTERNAL = {
    "licenses/cpython-LICENSE.txt": (
        "https://raw.githubusercontent.com/python/cpython/7f0ccd6c0e3f85fbaeceb2f67b06ab3631db0480/LICENSE",
        "b0e25a78cffb43f4d92de8b61ccfa1f1f98ecbc22330b54b5251e7b6ba010231",
        "20cf39097c68baa17cc566b64e76d34ebf034044",
    ),
    "licenses/CC0-1.0.txt": (
        (
            "https://raw.githubusercontent.com/MurzikVasilyevich/ukr-proverbs-franko/"
            "7f62a9d8f0673d325b0a565d508461f4b44dae5b/LICENSE"
        ),
        "a2010f343487d3f7618affe54f789f5487602331c0a8d03f49e9a7c547cf0499",
        "0e259d42c996742e9e3cba14c677129b2c1b6311",
    ),
    "licenses/werkzeug-LICENSE.txt": (
        "https://raw.githubusercontent.com/pallets/werkzeug/0005c79e09bae5f4cc2bd8ccd468d7dafe24a455/LICENSE.txt",
        "3b49dcee4105eb37bac10faf1be260408fe85d252b8e9df2e0979fc1e094437b",
        "c37cae49ec77ad6ebb25568c1605f1fee5313cfb",
    ),
    "licenses/pytest-LICENSE.txt": (
        "https://raw.githubusercontent.com/pytest-dev/pytest/28549a5f6b82bc916bb2ec5cb9fbfffe9b79fc66/LICENSE",
        "ca836a5f9ecca3b2f350230faa20a48fb8b145653b5568d784862df864706b9b",
        "c3f1657fce94589bd1ec7cead810639047f3d359",
    ),
    "licenses/languk-dataset-card.md": (
        LANGUK_CARD,
        "4a83c3140b81b1647bc73a9e70958ab487be42b17894ff4a2c0b9bad0fd26d95",
        "d6a6aabb212004f8ab7f2daff00e214cae89dcc3",
    ),
    "licenses/MIT-standard-text.txt": (
        "https://raw.githubusercontent.com/spdx/license-list-data/v3.27.0/text/MIT.txt",
        "b05785f9f18e6716bab63424b11454513b9943a222595b70411009202fc592b5",
        "d817195dad53ec992418c28ffca5fbd1cd86502a",
    ),
}
REPOSITORY = {
    "licenses/gutenberg-original-works-rights.txt": (
        "data/external/rights-evidence/plan2-books/public-domain-original-works-v1.txt",
        "f71d79a7d8e347825d84234db7acd46f079e12fb",
    ),
    "licenses/tokenizer-d03-ATTRIBUTION.txt": (f"{S15_D03}/ATTRIBUTION.txt", "147706a12a2cbea14f94edc6947c5ef420dae24e"),
    "licenses/php-uk-license.xml": (f"{S15_D03}/licenses/php/license.xml", "d8b388c25c87a12208e173a0f8e9f10754884dda"),
    "licenses/rust-book-LICENSE-APACHE": (f"{S15_D03}/licenses/rust/LICENSE-APACHE", "96e9f0458b55a4047927de5bf04ceda89d772b2b"),
    "licenses/rust-book-LICENSE-MIT": (f"{S15_D03}/licenses/rust/LICENSE-MIT", "5a56e6e8ed1909b4e4800aa8d2a0e7033ab4babe"),
    f"source-authorities/{Path(S15_SOURCE_AUTHORITY).name}": (S15_SOURCE_AUTHORITY, "fb88fc9379427f781589711d66d2cdaff9d98dad"),
    f"source-authorities/{Path(ARXIV_RIGHTS).name}": (ARXIV_RIGHTS, "31544961a99310487b2b2310adec10069065ac1c"),
    **{f"source-authorities/{Path(path).name}": (path, blob) for path, blob in s17.SOURCE_AUTHORITY_BLOBS.items()},
}
FAMILY_NOTICES = {
    "common-pile/arxiv_abstracts": ("licenses/CC0-1.0.txt", f"source-authorities/{Path(ARXIV_RIGHTS).name}"),
    "en.project-gutenberg.public-domain-books": ("licenses/gutenberg-original-works-rights.txt",),
    "github:pallets/werkzeug": ("licenses/werkzeug-LICENSE.txt",),
    "github:pytest-dev/pytest": ("licenses/pytest-LICENSE.txt",),
    "python.cpython.documentation": ("licenses/cpython-LICENSE.txt",),
    "ua.languk.supreme-court-decisions": ("licenses/languk-dataset-card.md", "licenses/MIT-standard-text.txt"),
    "ua.verba.public-domain.franko1901": ("licenses/CC0-1.0.txt",),
}


class SourceNoticesDenied(ValueError):
    """A source term or its exact notice bytes could not be verified."""


def _external_bytes(name: str, notice_cache: Path | None) -> bytes:
    url, sha256, blob = EXTERNAL[name]
    if notice_cache is not None:
        raw = books.read_checked(notice_cache, name)
    else:
        try:
            request = Request(url, headers={"User-Agent": "12-6-plan2-license-verifier/1"})
            with urlopen(request, timeout=20) as response:
                raw = response.read(MAX_NOTICE_BYTES + 1)
        except (OSError, URLError) as exc:
            raise SourceNoticesDenied(f"Unable to read pinned source notice {name}") from exc
    if not (0 < len(raw) <= MAX_NOTICE_BYTES and books.sha(raw) == sha256 and books.git_blob(raw) == blob):
        raise SourceNoticesDenied(f"Pinned source notice differs for {name}")
    return raw


def _repository_bytes(root: Path, name: str) -> bytes:
    path, blob = REPOSITORY[name]
    raw = books.read_checked(root, path)
    if books.git_blob(raw) != blob:
        # A Windows checkout may render committed text with CRLF.
        raw = raw.replace(b"\r\n", b"\n")
    if books.git_blob(raw) != blob:
        raise SourceNoticesDenied(f"Accepted repository source notice changed for {path}")
    return raw


def build_source_notices(root: Path, notice_cache: Path | None = None) -> dict[str, bytes]:
    """Return immutable release-relative notices, without reading corpus or evaluation payloads.

    A cache is the notices directory produced by an earlier build. Supplying one
    verifies the cached official notice bytes and performs no network access.
    """
    raw_admission = books.read_checked(root, s17.S16_MANIFEST).replace(b"\r\n", b"\n")
    if books.sha(raw_admission) != s17.S16_MANIFEST_FILE_SHA256:
        raise SourceNoticesDenied("Accepted corpus source terms changed")
    admission = json.loads(raw_admission)
    selected = admission["selected_source_terms"]
    if set(selected) != FAMILY_NOTICES.keys():
        raise SourceNoticesDenied("Accepted training source families differ from the notice inventory")

    files = {name: _repository_bytes(root, name) for name in REPOSITORY}
    files.update({name: _external_bytes(name, notice_cache) for name in EXTERNAL})
    families = {
        family: {
            **terms,
            "bundled_source_authority": f"source-authorities/{Path(terms['evidence']).name}",
            "bundled_notices": list(FAMILY_NOTICES[family]),
        }
        for family, terms in sorted(selected.items())
    }
    families["common-pile/arxiv_abstracts"]["license_scope_url"] = "https://info.arxiv.org/help/api/tou.html"
    families["ua.languk.supreme-court-decisions"]["license_declaration_url"] = LANGUK_CARD
    families["ua.languk.supreme-court-decisions"]["notice_scope"] = (
        "The pinned dataset card supplies the MIT declaration and curator attribution. "
        "The generic MIT text supplies the terms. The pinned source has no separate copyright notice."
    )
    families["python.cpython.documentation"]["changes"] = (
        "The tutorial introduction was normalized to UTF-8 with NFKC, LF and whitespace cleanup. "
        "It was split into chunks, screened for privacy, selected for the corpus, and tokenized and packed."
    )
    attribution = {
        "schema": SCHEMA,
        "source_admission_manifest_sha256": admission["manifest_sha256"],
        "training_families": families,
        "additional_frozen_tokenizer_sources": {
            "php.manual.documentation": ["licenses/tokenizer-d03-ATTRIBUTION.txt", "licenses/php-uk-license.xml"],
            "rust-book.documentation.uk-translation": [
                "licenses/tokenizer-d03-ATTRIBUTION.txt", "licenses/rust-book-LICENSE-APACHE",
                "licenses/rust-book-LICENSE-MIT",
            ],
        },
        "handoff_scope": "LOCAL_PROJECT_PLAN9_INPUT",
        "heldout_payloads_included": False,
        "external_dataset_distribution_authorized_by_this_receipt": False,
    }
    files["source-attribution.json"] = books.canonical(attribution)
    files["NOTICE.txt"] = (
        b"Plan 2 local training handoff source notices\n\n"
        b"See source-attribution.json for the exact corpus families, source authority records, "
        b"license files, and transformation details. Each source retains its own terms.\n\n"
        b"The literary records are the admitted original texts. arXiv's CC0 scope covers descriptive "
        b"metadata and abstracts. Preserve the supplied PSF, BSD and MIT notices for the corresponding "
        b"sources. The frozen tokenizer also carries the PHP and Rust Book documentation notices.\n\n"
        b"LangUK court decisions use the pinned pseudonymized dataset. Its card declares MIT. "
        b"That copyright declaration does not settle the privacy requirements for public distribution.\n\n"
        b"Validation and final-test data remain under their original evaluation reservations.\n"
    )
    inventory = {
        "schema": SCHEMA,
        "source_admission_manifest_sha256": admission["manifest_sha256"],
        "files": {
            name: {"bytes": len(raw), "sha256": books.sha(raw)}
            for name, raw in sorted(files.items())
        },
    }
    files["inventory.json"] = books.canonical({**inventory, "inventory_sha256": books.sha(books.canonical(inventory))})
    return {f"notices/{name}": raw for name, raw in sorted(files.items())}
