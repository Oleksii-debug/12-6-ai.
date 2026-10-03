"""Portable-run builder must not parse nonzero costs into a free-budget zero."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "tools/build_r01_portable_run_packet.py"
READINESS = ROOT / "configs/research/r01_learned20m_launch_readiness_v1.json"
TEMPLATE = ROOT / "configs/research/r01_portable_local_free_run_packet_v1.json"
OVERLAY = ROOT / "configs/research/r01_portable_session_overlay_v1.json"


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("portable_builder_underflow", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "token",
    [
        "1e-9999",
        "-1e-9999",
        "0.000000000000000001e-9999",
        "-0.000000000000000001e-9999",
    ],
)
def test_builder_input_rejects_nonzero_number_underflow(
    tmp_path: Path, token: str,
) -> None:
    source = tmp_path / "input.json"
    source.write_text('{"resource":{"maximum_cost_usd":' + token + '}}', encoding="utf-8")

    with pytest.raises(ValueError, match="nonzero JSON number underflowed to zero"):
        _builder()._load_object(source)


@pytest.mark.parametrize("token", ["0e-9999", "-0.000e-9999", "0.0"])
def test_builder_keeps_real_json_zero(tmp_path: Path, token: str) -> None:
    source = tmp_path / "input.json"
    source.write_text('{"resource":{"maximum_cost_usd":' + token + '}}', encoding="utf-8")

    data = _builder()._load_object(source)

    assert data == {"resource": {"maximum_cost_usd": 0.0}}


def test_builder_cli_rejects_underflow_without_packet_publication(
    tmp_path: Path,
) -> None:
    template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
    text = json.dumps(template)
    marker = '"maximum_cost_usd": 0'
    assert text.count(marker) == 1
    template_path = tmp_path / "шаблон із пробілами.json"
    template_path.write_text(
        text.replace(marker, '"maximum_cost_usd": 1e-9999'),
        encoding="utf-8",
    )
    output = tmp_path / "published.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--readiness",
            str(READINESS),
            "--template",
            str(template_path),
            "--overlay",
            str(OVERLAY),
            "--output",
            str(output),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert proc.returncode == 2
    assert proc.stderr == ""
    result = json.loads(proc.stdout)
    assert result["binding_ready"] is False
    assert "nonzero JSON number underflowed to zero" in result["error"]
    assert not output.exists()
