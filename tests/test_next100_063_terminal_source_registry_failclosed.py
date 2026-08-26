from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/next100_063_terminal_source_registry_v1.json"
VALIDATOR_PATH = ROOT / "tools/validate_next100_063_terminal_source_registry.py"

_spec = importlib.util.spec_from_file_location("next100_063_validator", VALIDATOR_PATH)
assert _spec is not None and _spec.loader is not None
validator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(validator)


def _load() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _reseal(data: dict) -> dict:
    sealed = copy.deepcopy(data)
    sealed.pop("registry_identity_sha256", None)
    canonical = json.dumps(
        sealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    sealed["registry_identity_sha256"] = hashlib.sha256(canonical).hexdigest()
    return sealed


def _run_resealed(data: dict) -> None:
    sealed = _reseal(data)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "registry.json"
        path.write_text(json.dumps(sealed, ensure_ascii=False), encoding="utf-8")
        old_path = validator.PATH
        old_identity = validator.EXPECTED_REGISTRY_IDENTITY
        validator.PATH = path
        validator.EXPECTED_REGISTRY_IDENTITY = sealed["registry_identity_sha256"]
        try:
            validator.main()
        finally:
            validator.PATH = old_path
            validator.EXPECTED_REGISTRY_IDENTITY = old_identity


class TerminalSourceRegistryFailClosedTests(unittest.TestCase):
    def test_current_registry_passes(self) -> None:
        validator.main()

    def test_selection_authorization_cannot_hide_behind_new_string(self) -> None:
        data = _load()
        data["terminal_additions"][0]["evaluation"] = "AUTHORIZED_FOR_SELECTION"
        with self.assertRaisesRegex(SystemExit, "evaluation permission must remain explicitly denied"):
            _run_resealed(data)

    def test_local_free_boundary_cannot_be_weakened(self) -> None:
        data = _load()
        data["local_free_only"] = False
        with self.assertRaisesRegex(SystemExit, "must remain LOCAL_FREE"):
            _run_resealed(data)

    def test_global_dedup_requirement_cannot_be_disabled(self) -> None:
        data = _load()
        data["composition_policy"]["global_cross_source_dedup_required_before_corpus_identity"] = False
        with self.assertRaisesRegex(SystemExit, "composition policy weakened"):
            _run_resealed(data)

    def test_counted_authority_cannot_also_be_held_out(self) -> None:
        data = _load()
        data["held_out_or_noncomposable"].append(
            {"pr": data["terminal_additions"][0]["pr"], "reason": "adversarial overlap"}
        )
        with self.assertRaisesRegex(SystemExit, "also marked held-out"):
            _run_resealed(data)

    def test_bool_is_not_valid_capacity(self) -> None:
        data = _load()
        row = data["terminal_additions"][0]
        pr = row["pr"]
        expected = list(validator.EXPECTED_TERMINAL_AUTHORITIES[pr])
        row["normalized_bytes"] = True
        expected[3] = True
        old = validator.EXPECTED_TERMINAL_AUTHORITIES[pr]
        validator.EXPECTED_TERMINAL_AUTHORITIES[pr] = tuple(expected)
        try:
            with self.assertRaisesRegex(SystemExit, "non-positive or non-integer capacity"):
                _run_resealed(data)
        finally:
            validator.EXPECTED_TERMINAL_AUTHORITIES[pr] = old


if __name__ == "__main__":
    unittest.main()
