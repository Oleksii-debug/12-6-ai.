"""Load the 32K tokenizer recorded by the accepted Section 15 run."""
from __future__ import annotations

import json
from gzip import GzipFile
from io import BytesIO
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_real_32k_bpe_v1 as incumbent
from tools.plan2_physical_materialization_v1 import _read_destination

ARTIFACT = "evidence/plan2_s17/frozen_32k_byte_bpe.json.gz"
S15_ACCEPTED_MAIN = "896f2643774d272c6120c4677fdd55849d99692b"
S15_ACTIONS_RUN = 38292511894
S15_IDENTITY_ARTIFACT = 11767462069
S15_MANIFEST_SHA256 = "798322c30e9a5396d218fbb696313996dd963827ca7d6c8124fea39f0269bb8e"
S15_TOKENIZER_SHA256 = "52fa76562890ef382c71dacfa79f4d1cd036efa4a581cf9ceb1f050ce95e5526"
S15_IDENTITY_ARCHIVE_SHA256 = "c472688d40c56272c15e3a7582ea5784e219c205606609e035cc6fed63e1bf00"
MODEL_SCALES = ("500M", "750M", "1B", "2B", "3B", "4B", "5B")


def load_frozen(root: Path) -> tuple[incumbent.ByteBPE32k, dict[str, Any]]:
    """Read the exact historical merge rules and verify the resulting vocabulary."""
    path = root / ARTIFACT
    incumbent.need(
        not any(member.is_symlink() for member in (path, *path.parents))
        and path.is_file() and 0 < path.stat().st_size <= 200_000,
        "invalid frozen tokenizer artifact",
    )
    try:
        with GzipFile(fileobj=BytesIO(_read_destination(path))) as stream:
            raw = stream.read(1_000_001)
        incumbent.need(books.sha(raw) == S15_TOKENIZER_SHA256,
                       "recovered tokenizer bytes differ from the accepted artifact")
        report = json.loads(raw)
        incumbent.need(isinstance(report, dict) and raw == books.canonical(report),
                       "invalid canonical tokenizer artifact")
        core = dict(report)
        identity = core.pop("manifest_sha256")
        incumbent.need(
            identity == S15_MANIFEST_SHA256 == books.sha(books.canonical(core))
            and report["schema_version"] == incumbent.SCHEMA
            and report["actual_vocab_size"] == incumbent.VOCAB
            and report["heldout_payloads_fitted"] is False,
            "tokenizer artifact differs from the accepted Section 15 identity",
        )
        tokenizer = incumbent.ByteBPE32k(report["fitted_merges"])
        incumbent.need(tokenizer.identity.to_dict() == report["tokenizer_identity"],
                       "ordered merge rules do not reproduce the frozen vocabulary")
    except (EOFError, KeyError, OSError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise incumbent.Real32kBpeDenied("malformed frozen tokenizer artifact") from exc
    return tokenizer, {
        "source_accepted_main": S15_ACCEPTED_MAIN,
        "source_actions_run": S15_ACTIONS_RUN,
        "source_identity_artifact": S15_IDENTITY_ARTIFACT,
        "source_identity_archive_sha256": S15_IDENTITY_ARCHIVE_SHA256,
        "source_manifest_sha256": S15_MANIFEST_SHA256,
        "artifact_sha256": books.sha(raw),
        "tokenizer_identity": tokenizer.identity.to_dict(),
        "compatible_scale_profiles": list(MODEL_SCALES),
    }
