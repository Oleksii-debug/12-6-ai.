import copy
import hashlib

import numpy as np
import json
from pathlib import Path

import pytest

from twelve_six.third_party_reuse import (
    prepare_trusted_base_checkpoint,
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


class TinyModel:
    def __init__(self, value=1.0):
        self.weights = np.array([value, value + 1], dtype=np.float64)
        self.mutations = 0

    def state_dict(self):
        return {"weights": self.weights.copy()}

    def load_state_dict(self, state, strict=True):
        self.mutations += 1
        self.weights = state["weights"].copy()


def real_checkpoint_case(tmp_path):
    from twelve_six.checkpoint import CheckpointIdentity, save_checkpoint

    init_spec_hash = "c" * 64
    identity = CheckpointIdentity(
        git_sha="f" * 40,
        model_spec={"kind": "tiny-reuse-contract", "width": 2},
        parameter_count=2,
        tokenizer_hash="1" * 64,
        tokenizer_vocab_hash="2" * 64,
        dataset_manifest_hash="3" * 64,
        run_manifest_hash="4" * 64,
        training_config={"init_spec_sha256": init_spec_hash},
        seed=17,
        precision="float64-test",
        step=0,
        tokens_seen=0,
        optimizer={"name": "test-no-optimizer"},
        scheduler=None,
    )
    root = tmp_path / "scratch"
    manifest = save_checkpoint(root, model=TinyModel(), identity=identity)
    genesis = dict(
        genesis_id="a" * 64,
        model_spec_sha256=manifest["identity"]["model_spec_hash"],
        init_spec_sha256=init_spec_hash,
        origin="local_random_init",
    )
    chain = dict(
        schema_version=1,
        genesis_id=genesis["genesis_id"],
        model_spec_sha256=genesis["model_spec_sha256"],
        init_spec_sha256=genesis["init_spec_sha256"],
        head_id=manifest["checkpoint_id"],
        checkpoints=[
            dict(id=genesis["genesis_id"], parents=[], origin="canonical_base"),
            dict(
                id=manifest["checkpoint_id"],
                parents=[genesis["genesis_id"]],
                origin="canonical_base",
            ),
        ],
    )
    genesis_bytes = wire(genesis)
    return root, chain, genesis_bytes, hashlib.sha256(genesis_bytes).hexdigest()


def test_real_checkpoint_ingress_and_canonical_restore(tmp_path):
    from twelve_six.checkpoint import load_verified_checkpoint

    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    verified = prepare_trusted_base_checkpoint(
        root,
        lineage_bytes=wire(chain),
        trusted_genesis_bytes=genesis_bytes,
        expected_genesis_sha256=pin,
    )
    target = TinyModel(99.0)
    result = load_verified_checkpoint(verified, model=target, restore_rng=False)
    assert target.mutations == 1
    np.testing.assert_array_equal(target.weights, [1.0, 2.0])
    assert result.manifest["checkpoint_id"] == chain["head_id"]


def test_unpinned_foreign_or_drifted_base_is_rejected_pre_mutation(tmp_path):
    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    target = TinyModel(99.0)
    bad_cases = []
    forged = copy.deepcopy(chain)
    forged["head_id"] = "d" * 64
    forged["checkpoints"][1]["id"] = "d" * 64
    bad_cases.append((forged, genesis_bytes, pin))
    foreign = copy.deepcopy(chain)
    foreign["checkpoints"][1]["origin"] = "pretrained"
    bad_cases.append((foreign, genesis_bytes, pin))
    wrong_model = copy.deepcopy(chain)
    wrong_model["model_spec_sha256"] = "b" * 64
    bad_cases.append((wrong_model, genesis_bytes, pin))
    bad_cases.append((chain, genesis_bytes, "f" * 64))
    for lineage_value, trusted_bytes, expected in bad_cases:
        with pytest.raises(ValueError):
            prepare_trusted_base_checkpoint(
                root,
                lineage_bytes=wire(lineage_value),
                trusted_genesis_bytes=trusted_bytes,
                expected_genesis_sha256=expected,
            )
    assert target.mutations == 0


def test_actual_checkpoint_tamper_and_init_drift_fail_closed(tmp_path):
    root, chain, genesis_bytes, pin = real_checkpoint_case(tmp_path)
    init_drift = copy.deepcopy(chain)
    init_drift["init_spec_sha256"] = "0" * 64
    trusted_init_drift = json.loads(genesis_bytes)
    trusted_init_drift["init_spec_sha256"] = "0" * 64
    changed_genesis = wire(trusted_init_drift)
    with pytest.raises(ValueError, match="InitSpec"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(init_drift),
            trusted_genesis_bytes=changed_genesis,
            expected_genesis_sha256=hashlib.sha256(changed_genesis).hexdigest(),
        )
    with (root / "weights.safetensors").open("ab") as stream:
        stream.write(b"foreign checkpoint bytes")
    with pytest.raises(Exception, match="checksum|size mismatch"):
        prepare_trusted_base_checkpoint(
            root,
            lineage_bytes=wire(chain),
            trusted_genesis_bytes=genesis_bytes,
            expected_genesis_sha256=pin,
        )


def test_large_cyclic_lineage_fails_boundedly():
    m = lineage()
    m["checkpoints"] *= 2049
    with pytest.raises(ValueError, match="complete checkpoint graph"):
        validate_base_lineage(m, GENESIS)
