"""Release authority and default Plan 4 decoding regressions."""
from pathlib import Path

import pytest

from tools import plan2_s17_frozen_tokenizer_v1 as frozen
from tools import plan2_s18_physical_release_v1 as release
from tools.plan2_s15_real_32k_bpe_v1 import Real32kBpeDenied
from tools.plan2_s18_release_integrity_v1 import IntegrityDenied

ROOT = Path(__file__).resolve().parents[1]


def test_a_locally_resealed_release_authority_cannot_replace_the_trusted_pin(tmp_path):
    file = tmp_path / release.AUTHORITY
    file.parent.mkdir(parents=True)
    core = {
        "schema": release.AUTHORITY_SCHEMA,
        "decision": "ACCEPT_LOCAL_PLAN9_DATA_RELEASE",
        "s17_accepted_main": release.S17_ACCEPTED_MAIN,
        "campaign_authorized": False,
        "release_manifest_sha256": "0" * 64,
    }
    file.write_bytes(release.books.canonical({
        **core, "authority_sha256": release.books.sha(release.books.canonical(core)),
    }))
    with pytest.raises(IntegrityDenied, match="missing accepted Plan 2 release authority"):
        release.open_release(tmp_path, tmp_path / "no-bundle")


def test_default_plan4_decode_replaces_invalid_byte_without_changing_strict_decoding():
    tokenizer, proof = frozen.load_frozen(ROOT)
    assert tokenizer.decode([255], errors="replace") == "�"
    with pytest.raises(Real32kBpeDenied, match="invalid token byte sequence"):
        tokenizer.decode([255], errors="strict")
    assert proof["source_manifest_sha256"] == frozen.S15_MANIFEST_SHA256
