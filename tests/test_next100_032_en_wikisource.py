from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next100_032_validator",
    ROOT / "tools/validate_next100_032_en_wikisource.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_next100_032_terminal_source_authority():
    assert MODULE.validate(ROOT) == []
