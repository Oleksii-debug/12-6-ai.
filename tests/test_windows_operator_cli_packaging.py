from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import venv
import zipfile
from pathlib import Path

from twelve_six.windows_operator_cli import build_delegate_argv, resolve_default_paths

ROOT = Path(__file__).resolve().parents[1]
PROFILE_RELATIVE = Path("configs/research/r01_windows_local_free_operator_v1.json")
PACKET_RELATIVE = Path("configs/research/r01_portable_local_free_run_packet_v1.json")
INSTALL_DATA_SUFFIX = Path("share/twelve-six-ai")
ENTRY_POINT = "twelve-six-windows = twelve_six.windows_operator_cli:main"


def _fake_checkout(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    module = root / "src/twelve_six/windows_operator_cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("# fixture\n", encoding="utf-8")
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    for relative in (PROFILE_RELATIVE, PACKET_RELATIVE):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    return module


def test_source_checkout_defaults_preserve_existing_operator_layout(tmp_path: Path) -> None:
    module = _fake_checkout(tmp_path)
    checkout = module.parents[2]
    profile, packet, state = resolve_default_paths(
        module_path=module,
        prefix=tmp_path / "ignored-prefix",
        home=tmp_path / "ignored-home",
    )
    assert profile == checkout / PROFILE_RELATIVE
    assert packet == checkout / PACKET_RELATIVE
    assert state == checkout / ".twelve-six-local"


def test_installed_defaults_use_wheel_share_and_owner_home(tmp_path: Path) -> None:
    module = tmp_path / "venv/Lib/site-packages/twelve_six/windows_operator_cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("# installed fixture\n", encoding="utf-8")
    prefix = tmp_path / "venv"
    home = tmp_path / "owner"
    home.mkdir()
    profile, packet, state = resolve_default_paths(
        module_path=module,
        prefix=prefix,
        home=home,
    )
    assert profile == prefix / INSTALL_DATA_SUFFIX / PROFILE_RELATIVE
    assert packet == prefix / INSTALL_DATA_SUFFIX / PACKET_RELATIVE
    assert state == home / ".twelve-six-local"


def test_delegate_argv_injects_only_missing_defaults(tmp_path: Path) -> None:
    module = tmp_path / "venv/Lib/site-packages/twelve_six/windows_operator_cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("# installed fixture\n", encoding="utf-8")
    prefix = tmp_path / "venv"
    home = tmp_path / "owner"
    supplied = [
        "--profile",
        "custom-profile.json",
        "--packet=custom-packet.json",
        "--state-dir",
        "custom-state",
        "--json",
        "verify",
        "--target",
        "20m",
    ]
    assert build_delegate_argv(
        supplied,
        module_path=module,
        prefix=prefix,
        home=home,
    ) == supplied

    injected = build_delegate_argv(
        ["--json", "verify", "--target", "20m"],
        module_path=module,
        prefix=prefix,
        home=home,
    )
    assert injected[:6] == [
        "--profile",
        str(prefix / INSTALL_DATA_SUFFIX / PROFILE_RELATIVE),
        "--packet",
        str(prefix / INSTALL_DATA_SUFFIX / PACKET_RELATIVE),
        "--state-dir",
        str(home / ".twelve-six-local"),
    ]
    assert injected[6:] == ["--json", "verify", "--target", "20m"]


def test_pyproject_packages_exact_canonical_assets_and_one_cli() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["scripts"]["twelve-six-windows"] == (
        "twelve_six.windows_operator_cli:main"
    )
    packaged = config["tool"]["setuptools"]["data-files"][
        "share/twelve-six-ai/configs/research"
    ]
    assert packaged == [PROFILE_RELATIVE.as_posix(), PACKET_RELATIVE.as_posix()]
    for relative in (PROFILE_RELATIVE, PACKET_RELATIVE):
        assert (ROOT / relative).is_file()


def _venv_paths(venv_dir: Path) -> tuple[Path, Path]:
    if os.name == "nt":
        return (
            venv_dir / "Scripts/python.exe",
            venv_dir / "Scripts/twelve-six-windows.exe",
        )
    return venv_dir / "bin/python", venv_dir / "bin/twelve-six-windows"


def test_built_wheel_contains_exact_assets_and_noneditable_cli_uses_them(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(ROOT),
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    wheels = list(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        for relative in (PROFILE_RELATIVE, PACKET_RELATIVE):
            suffix = (INSTALL_DATA_SUFFIX / relative).as_posix()
            matches = [name for name in names if name.endswith(suffix)]
            assert len(matches) == 1
            assert archive.read(matches[0]) == (ROOT / relative).read_bytes()
        entry_points = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        assert len(entry_points) == 1
        assert ENTRY_POINT in archive.read(entry_points[0]).decode("utf-8")

    venv_dir = tmp_path / "installed"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    venv_python, console = _venv_paths(venv_dir)
    install = subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            str(wheel),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    assert console.is_file()

    env = dict(os.environ)
    owner_home = tmp_path / "owner-home"
    owner_home.mkdir()
    env["HOME"] = str(owner_home)
    env["USERPROFILE"] = str(owner_home)
    smoke = subprocess.run(
        [str(console), "--json", "verify", "--target", "20m"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert smoke.returncode in {0, 2}, smoke.stdout + smoke.stderr
    result = json.loads(smoke.stdout)
    assert result["contract_errors"] == []
    assert result["launch_authorized"] is False
    assert result["training_authorized"] is False
    assert result["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert result["truth_boundary"]["training_executed"] is False
