from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.validate_next100022_ua_wikisource import (
    CONFIG,
    QualificationError,
    _canonical_json_bytes,
    validate,
)


def _write_with_identity(cfg: dict) -> Path:
    cfg = copy.deepcopy(cfg)
    cfg.pop("authority_identity_sha256", None)
    cfg["authority_identity_sha256"] = hashlib.sha256(_canonical_json_bytes(cfg)).hexdigest()
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False)
    json.dump(cfg, handle, ensure_ascii=False, sort_keys=True)
    handle.write("\n")
    handle.close()
    return Path(handle.name)


class Next100022WikisourceQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.base = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_terminal_authority_validates(self) -> None:
        result = validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["rights_training_permission"], "ALLOWED")
        self.assertEqual(result["evaluation"], "NOT_SEPARATELY_ADMITTED")
        self.assertTrue(result["corpus_training_selection"].startswith("BLOCKED_UNTIL_"))

    def test_evaluation_permission_cannot_be_inferred(self) -> None:
        cfg = copy.deepcopy(self.base)
        cfg["rights"]["uses"]["evaluation"] = "ALLOWED"
        path = _write_with_identity(cfg)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(QualificationError, "evaluation may not be inferred"):
            validate(path)

    def test_wikimedia_domain_cannot_define_family(self) -> None:
        cfg = copy.deepcopy(self.base)
        cfg["family_lineage"]["wikimedia_platform_is_not_family_identity"] = False
        path = _write_with_identity(cfg)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(QualificationError, "hosting platform"):
            validate(path)

    def test_near_match_decontamination_remains_fail_closed(self) -> None:
        cfg = copy.deepcopy(self.base)
        cfg["evaluation_exclusion"]["corpus_training_selection"] = "ALLOWED"
        path = _write_with_identity(cfg)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(QualificationError, "corpus training must fail closed"):
            validate(path)

    def test_generic_cc_only_scope_cannot_be_silently_admitted(self) -> None:
        cfg = copy.deepcopy(self.base)
        cfg["rights"]["scope_boundary"] = "Only exact page."
        path = _write_with_identity(cfg)
        self.addCleanup(path.unlink, missing_ok=True)
        with self.assertRaisesRegex(QualificationError, "generic licensed-content"):
            validate(path)


if __name__ == "__main__":
    unittest.main()
