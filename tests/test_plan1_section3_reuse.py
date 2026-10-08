import copy
import json
from pathlib import Path

import pytest

from twelve_six.third_party_reuse import (
    require_reviewed_code,
    validate_base_lineage,
    validate_reuse_catalog,
)

RAW = (Path(__file__).resolve().parents[1] /
       "configs/control/plan1_reuse_assets_v1.json").read_bytes()
GENESIS = dict(genesis_id="a"*64, model_spec_sha256="b"*64,
               init_spec_sha256="c"*64, origin="local_random_init")


def wire(obj):
    return json.dumps(obj, sort_keys=True).encode()


def lineage():
    return dict(schema_version=1, genesis_id="a"*64, model_spec_sha256="b"*64,
                init_spec_sha256="c"*64, head_id="d"*64, checkpoints=[
                    dict(id="a"*64, parents=[], origin="canonical_base"),
                    dict(id="d"*64, parents=["a"*64], origin="canonical_base"),
                ])


def test_reuse_catalog_is_quarantined_and_restart_stable():
    assert len(validate_reuse_catalog(RAW)["assets"]) == 13
    assert validate_reuse_catalog(RAW) == validate_reuse_catalog(bytes(RAW))
    with pytest.raises(ValueError, match="unqualified"):
        require_reviewed_code(RAW, "pytorch", "e"*64)
    with pytest.raises(ValueError, match="unknown external"):
        require_reviewed_code(RAW, "ghost", "e"*64)


def test_no_second_authority_and_no_untrusted_catalog():
    obj = json.loads(RAW)
    obj["canonical_authorities"]["model"] = "vendor.model"
    with pytest.raises(ValueError, match="second canonical"):
        validate_reuse_catalog(wire(obj))
    obj = json.loads(RAW)
    obj["assets"].append(copy.deepcopy(obj["assets"][0]))
    with pytest.raises(ValueError, match="duplicate asset"):
        validate_reuse_catalog(wire(obj))
    with pytest.raises(ValueError, match="duplicate JSON"):
        validate_reuse_catalog(b'{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="invalid strict"):
        validate_reuse_catalog(b'{"schema_version":NaN}')


def test_exact_code_license_security_rights_and_hash_check():
    obj = json.loads(RAW)
    item = obj["assets"][0]
    item.update(status="REVIEWED_CODE_ONLY", upstream_url="https://example.org",
                version="1.0", source_sha256="e"*64, license_spdx="MIT",
                license_evidence_sha256="f"*64, security_posture="REVIEWED",
                data_rights="NOT_APPLICABLE_CODE")
    assert require_reviewed_code(wire(obj), "pytorch", "e"*64)["version"] == "1.0"
    with pytest.raises(ValueError, match="source hash drift"):
        require_reviewed_code(wire(obj), "pytorch", "d"*64)
    for key, bad in [("model_weights", "PRETRAINED"), ("license_spdx", "UNKNOWN"),
                     ("data_rights", "ALLOWED"), ("source_sha256", "z"*64),
                     ("security_posture", "UNKNOWN")]:
        other = copy.deepcopy(obj)
        other["assets"][0][key] = bad
        with pytest.raises(ValueError):
            validate_reuse_catalog(wire(other))


def test_closed_canonical_lineage_is_deterministic():
    assert validate_base_lineage(lineage(), GENESIS) == validate_base_lineage(
        copy.deepcopy(lineage()), copy.deepcopy(GENESIS)
    )


def test_foreign_root_unknown_parent_cycle_and_duplicate_are_rejected():
    cases = []
    m = lineage()
    m["checkpoints"][1]["origin"] = "pretrained"
    cases.append(m)
    m = lineage()
    m["checkpoints"][1]["parents"] = ["e"*64]
    cases.append(m)
    m = lineage()
    m["checkpoints"][0]["parents"] = ["d"*64]
    cases.append(m)
    m = lineage()
    m["checkpoints"][1]["parents"] = []
    cases.append(m)
    m = lineage()
    m["checkpoints"].append(copy.deepcopy(m["checkpoints"][0]))
    cases.append(m)
    m = lineage()
    m["head_id"] = "0"*64
    cases.append(m)
    m = lineage()
    m["schema_version"] = True
    cases.append(m)
    for bad in cases:
        with pytest.raises(ValueError):
            validate_base_lineage(bad, GENESIS)
    with pytest.raises(ValueError):
        validate_base_lineage(lineage(), dict(GENESIS, origin="foreign_pretrained"))
