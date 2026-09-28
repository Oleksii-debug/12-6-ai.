from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_cli() -> ModuleType:
    script = Path(__file__).parents[1] / "tools" / "assess_learned20m_training_lease.py"
    spec = importlib.util.spec_from_file_location("assess_learned20m_training_lease_cli", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_malformed_enum_returns_machine_readable_denial(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps({"resource": {"resource_class": []}}),
        encoding="utf-8",
    )

    result = _load_cli().main(["assess_learned20m_training_lease.py", str(manifest_path)])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert result == 1
    assert payload["contract_valid"] is False
    assert payload["local_duplicate_guard_open"] is False
    assert "Traceback" not in captured.err
