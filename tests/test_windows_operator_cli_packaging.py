from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
import venv
import zipfile
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest

from twelve_six import windows_operator_cli
from twelve_six.windows_operator_cli import build_delegate_argv, resolve_default_paths

ROOT = Path(__file__).resolve().parents[1]
PROFILE_RELATIVE = Path("configs/research/r01_windows_local_free_operator_v1.json")
PACKET_RELATIVE = Path("configs/research/r01_portable_local_free_run_packet_v1.json")
INSTALL_DATA_SUFFIX = Path("share/twelve-six-ai")
ENTRY_POINT = "twelve-six-windows = twelve_six.windows_operator_cli:main"
PROFILE_RECORD = "../../../share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name
PACKET_RECORD = "../../../share/twelve-six-ai/configs/research/" + PACKET_RELATIVE.name


class _FakeDistribution:
    def __init__(
        self,
        locations: dict[str, Path],
        *,
        files_available: bool = True,
    ) -> None:
        self._locations = locations
        self.files = (
            [metadata.PackagePath(entry) for entry in locations] if files_available else None
        )

    def locate_file(self, entry: object) -> Path:
        return self._locations[str(entry)]


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


def _fake_installed_module(tmp_path: Path) -> Path:
    module = tmp_path / "system-prefix/lib/site-packages/twelve_six/windows_operator_cli.py"
    module.parent.mkdir(parents=True)
    module.write_text("# installed fixture\n", encoding="utf-8")
    return module


def _fake_user_scheme_distribution(
    tmp_path: Path,
) -> tuple[_FakeDistribution, Path, Path]:
    data_root = tmp_path / "user-base/share/twelve-six-ai/configs/research"
    data_root.mkdir(parents=True)
    profile = data_root / PROFILE_RELATIVE.name
    packet = data_root / PACKET_RELATIVE.name
    profile.write_text("{}\n", encoding="utf-8")
    packet.write_text("{}\n", encoding="utf-8")
    return (
        _FakeDistribution({PROFILE_RECORD: profile, PACKET_RECORD: packet}),
        profile,
        packet,
    )


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


def test_installed_defaults_follow_distribution_record_not_sys_prefix(tmp_path: Path) -> None:
    module = _fake_installed_module(tmp_path)
    distribution, expected_profile, expected_packet = _fake_user_scheme_distribution(tmp_path)
    unrelated_prefix = tmp_path / "system-prefix"
    owner_home = tmp_path / "owner"
    owner_home.mkdir()

    profile, packet, state = resolve_default_paths(
        module_path=module,
        prefix=unrelated_prefix,
        home=owner_home,
        distribution=distribution,
    )

    assert profile == expected_profile
    assert packet == expected_packet
    assert profile != unrelated_prefix / INSTALL_DATA_SUFFIX / PROFILE_RELATIVE
    assert packet != unrelated_prefix / INSTALL_DATA_SUFFIX / PACKET_RELATIVE
    assert state == owner_home / ".twelve-six-local"


@pytest.mark.parametrize("upward_prefix", ["", "../", "../../", "../../../../../../"])
def test_installed_record_resolution_accepts_only_upward_scheme_prefixes(
    tmp_path: Path,
    upward_prefix: str,
) -> None:
    module = _fake_installed_module(tmp_path)
    data_root = tmp_path / "accepted/share/twelve-six-ai/configs/research"
    data_root.mkdir(parents=True)
    profile = data_root / PROFILE_RELATIVE.name
    packet = data_root / PACKET_RELATIVE.name
    profile.write_text("{}\n", encoding="utf-8")
    packet.write_text("{}\n", encoding="utf-8")
    profile_record = (
        upward_prefix + "share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name
    )
    packet_record = (
        upward_prefix + "share/twelve-six-ai/configs/research/" + PACKET_RELATIVE.name
    )
    distribution = _FakeDistribution({profile_record: profile, packet_record: packet})

    resolved_profile, resolved_packet, _ = resolve_default_paths(
        module_path=module,
        distribution=distribution,
    )

    assert resolved_profile == profile
    assert resolved_packet == packet


@pytest.mark.parametrize(
    "profile_record",
    [
        "evil/share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name,
        "/tmp/evil/share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name,
        "../../../foo/../share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name,
        "..\\..\\..\\share\\twelve-six-ai\\configs\\research\\" + PROFILE_RELATIVE.name,
    ],
)
def test_installed_record_resolution_rejects_noncanonical_prefix_aliases(
    tmp_path: Path,
    profile_record: str,
) -> None:
    module = _fake_installed_module(tmp_path)
    profile = tmp_path / "alias-profile.json"
    packet = tmp_path / "packet.json"
    profile.write_text("{}\n", encoding="utf-8")
    packet.write_text("{}\n", encoding="utf-8")
    distribution = _FakeDistribution({profile_record: profile, PACKET_RECORD: packet})

    with pytest.raises(RuntimeError, match="found 0"):
        resolve_default_paths(module_path=module, distribution=distribution)


def test_installed_record_resolution_fails_closed_on_missing_duplicate_or_nonfile(
    tmp_path: Path,
) -> None:
    module = _fake_installed_module(tmp_path)

    with pytest.raises(RuntimeError, match="found 0"):
        resolve_default_paths(module_path=module, distribution=_FakeDistribution({}))

    first = tmp_path / "first-profile.json"
    second = tmp_path / "second-profile.json"
    first.write_text("{}\n", encoding="utf-8")
    second.write_text("{}\n", encoding="utf-8")
    duplicate = _FakeDistribution(
        {
            PROFILE_RECORD: first,
            "../../../../share/twelve-six-ai/configs/research/" + PROFILE_RELATIVE.name: second,
        }
    )
    with pytest.raises(RuntimeError, match="found 2"):
        resolve_default_paths(module_path=module, distribution=duplicate)

    directory = tmp_path / "not-a-file"
    directory.mkdir()
    nonfile = _FakeDistribution({PROFILE_RECORD: directory})
    with pytest.raises(RuntimeError, match="not a regular file"):
        resolve_default_paths(module_path=module, distribution=nonfile)


def test_installed_record_resolution_rejects_symlinked_profile(tmp_path: Path) -> None:
    module = _fake_installed_module(tmp_path)
    substituted = tmp_path / "substituted-profile.json"
    substituted.write_text("{}\n", encoding="utf-8")
    symlink = tmp_path / "linked-profile.json"
    try:
        symlink.symlink_to(substituted)
    except (NotImplementedError, OSError):
        pytest.skip("creating symlinks is unsupported on this machine")
    packet = tmp_path / "packet.json"
    packet.write_text("{}\n", encoding="utf-8")
    distribution = _FakeDistribution({PROFILE_RECORD: symlink, PACKET_RECORD: packet})
    with pytest.raises(RuntimeError, match="not a regular file"):
        resolve_default_paths(module_path=module, distribution=distribution)


def test_installed_record_resolution_fails_closed_without_record_listing(tmp_path: Path) -> None:
    module = _fake_installed_module(tmp_path)
    unavailable = _FakeDistribution({}, files_available=False)
    with pytest.raises(RuntimeError, match="RECORD file list is unavailable"):
        resolve_default_paths(module_path=module, distribution=unavailable)


def test_delegate_argv_preserves_complete_explicit_overrides_without_metadata(
    tmp_path: Path,
) -> None:
    module = _fake_installed_module(tmp_path)
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
    assert build_delegate_argv(supplied, module_path=module) == supplied


def test_delegate_argv_resolves_only_missing_installed_asset(tmp_path: Path) -> None:
    module = _fake_installed_module(tmp_path)
    data_root = tmp_path / "user-base/share/twelve-six-ai/configs/research"
    data_root.mkdir(parents=True)
    packet = data_root / PACKET_RELATIVE.name
    packet.write_text("{}\n", encoding="utf-8")
    distribution = _FakeDistribution({PACKET_RECORD: packet})
    owner_home = tmp_path / "owner"
    owner_home.mkdir()

    supplied = ["--profile", "custom-profile.json", "--json", "verify", "--target", "20m"]
    delegated = build_delegate_argv(
        supplied,
        module_path=module,
        prefix=tmp_path / "unrelated-prefix",
        home=owner_home,
        distribution=distribution,
    )
    assert delegated[:4] == [
        "--packet",
        str(packet),
        "--state-dir",
        str(owner_home / ".twelve-six-local"),
    ]
    assert delegated[4:] == supplied


def test_main_fails_closed_on_bootstrap_runtime_error_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_bootstrap(_argv: object) -> list[str]:
        raise RuntimeError("broken\nmetadata")

    monkeypatch.setattr(windows_operator_cli, "build_delegate_argv", fail_bootstrap)

    result = windows_operator_cli.main(["verify"])
    captured = capsys.readouterr()

    assert result == windows_operator_cli.windows_operator_preflight.EXIT_ERROR
    assert captured.err == ""
    assert "Traceback" not in captured.out
    assert captured.out.splitlines() == [
        "OPERATOR_STATUS: ERROR",
        r"ERROR: installed_operator_bootstrap_failed:broken\x0ametadata",
        "LAUNCH_AUTHORIZED: false",
        "TRAINING_AUTHORIZED: false",
    ]


def test_main_bootstrap_os_error_is_one_line_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_bootstrap(_argv: object) -> list[str]:
        raise OSError("record\nread")

    monkeypatch.setattr(windows_operator_cli, "build_delegate_argv", fail_bootstrap)

    result = windows_operator_cli.main(["--json", "verify"])
    captured = capsys.readouterr()

    assert result == windows_operator_cli.windows_operator_preflight.EXIT_ERROR
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert "\\n" in captured.out
    payload = json.loads(captured.out)
    assert payload["error"] == "installed_operator_bootstrap_failed:record\nread"
    assert payload["launch_authorized"] is False
    assert payload["status"] == "ERROR"
    assert payload["training_authorized"] is False
    assert payload["truth_boundary"] == windows_operator_cli.windows_operator_preflight._TRUTH_BOUNDARY


def test_main_preserves_successful_delegate_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated = ["--state-dir", "state", "verify"]
    observed: list[list[str]] = []

    monkeypatch.setattr(
        windows_operator_cli,
        "build_delegate_argv",
        lambda argv: delegated,
    )

    def fake_preflight_main(argv: list[str]) -> int:
        observed.append(argv)
        return 41

    monkeypatch.setattr(
        windows_operator_cli.windows_operator_preflight,
        "main",
        fake_preflight_main,
    )

    assert windows_operator_cli.main(["verify"]) == 41
    assert observed == [delegated]


@pytest.mark.parametrize(
    "args",
    [["--help"], ["-h"], ["verify", "--help"], ["--json", "status", "--help"]],
)
def test_help_never_requires_installed_record_or_assets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
) -> None:
    def fail_if_bootstrapped(_argv: object) -> list[str]:
        raise AssertionError("help must not inspect installed metadata")

    monkeypatch.setattr(windows_operator_cli, "build_delegate_argv", fail_if_bootstrapped)
    with pytest.raises(SystemExit) as exit_info:
        windows_operator_cli.main(args)
    captured = capsys.readouterr()
    assert exit_info.value.code == 0
    assert "usage:" in captured.out
    assert captured.err == ""


def test_malformed_installed_record_value_error_is_safe_text_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def malformed_record(_argv: object) -> list[str]:
        raise ValueError("invalid RECORD entry")

    monkeypatch.setattr(windows_operator_cli, "build_delegate_argv", malformed_record)
    result = windows_operator_cli.main(["verify"])
    captured = capsys.readouterr()
    assert result == windows_operator_cli.windows_operator_preflight.EXIT_ERROR
    assert captured.err == ""
    assert captured.out.splitlines() == [
        "OPERATOR_STATUS: ERROR",
        "ERROR: installed_operator_bootstrap_failed:invalid RECORD entry",
        "LAUNCH_AUTHORIZED: false",
        "TRAINING_AUTHORIZED: false",
    ]

def test_pyproject_packages_exact_canonical_assets_and_one_cli() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert config["project"]["dependencies"] == [
        "numpy>=1.26",
        "safetensors>=0.5",
        "torch>=2.5",
    ]
    assert config["project"]["scripts"]["twelve-six-windows"] == (
        "twelve_six.windows_operator_cli:main"
    )
    packaged = config["tool"]["setuptools"]["data-files"][
        "share/twelve-six-ai/configs/research"
    ]
    assert packaged == [PROFILE_RELATIVE.as_posix(), PACKET_RELATIVE.as_posix()]
    assert config["tool"]["ruff"]["src"] == ["src"]
    assert config["tool"]["ruff"]["lint"]["isort"]["known-first-party"] == [
        "twelve_six",
        "tools",
    ]
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
    clean_env = dict(os.environ)
    clean_env.pop("PYTHONPATH", None)
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
        env=clean_env,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    assert console.is_file()

    metadata_probe = subprocess.run(
        [
            str(venv_python),
            "-c",
            (
                "from twelve_six.windows_operator_cli import resolve_default_paths; "
                "p, q, _ = resolve_default_paths(); "
                "print(p); print(q); "
                "raise SystemExit(0 if p.is_file() and q.is_file() else 3)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=clean_env,
    )
    assert metadata_probe.returncode == 0, metadata_probe.stdout + metadata_probe.stderr

    env = dict(clean_env)
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

    installed_paths = metadata_probe.stdout.splitlines()
    assert len(installed_paths) == 2
    installed_profile = Path(installed_paths[0])
    assert installed_profile.is_file()
    installed_profile.unlink()

    missing_asset = subprocess.run(
        [str(console), "--json", "verify", "--target", "20m"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert missing_asset.returncode == 3
    assert missing_asset.stderr == ""
    assert "Traceback" not in missing_asset.stdout
    assert len(missing_asset.stdout.splitlines()) == 1
    missing_result = json.loads(missing_asset.stdout)
    assert missing_result["status"] == "ERROR"
    assert missing_result["error"].startswith("installed_operator_bootstrap_failed:")
    assert "installed canonical asset is missing or not a regular file:" in missing_result["error"]
    assert missing_result["launch_authorized"] is False
    assert missing_result["training_authorized"] is False
    assert missing_result["truth_boundary"]["authorized_optimized_target_exposure"] == 0
    assert missing_result["truth_boundary"]["training_executed"] is False


@pytest.mark.parametrize(
    "linked_component", ["share", "twelve-six-ai", "configs", "research"],
)
def test_installed_record_rejects_symlinked_canonical_asset_directory(
    tmp_path: Path,
    linked_component: str,
) -> None:
    """A regular leaf behind a linked parent is still a substituted asset."""
    module = _fake_installed_module(tmp_path)
    location = tmp_path / "installed-prefix"
    location.mkdir()
    for component in ("share", "twelve-six-ai", "configs", "research"):
        child = location / component
        if component == linked_component:
            destination = tmp_path / "replacement-tree"
            destination.mkdir()
            try:
                child.symlink_to(destination, target_is_directory=True)
            except (NotImplementedError, OSError):
                pytest.skip("creating directory symlinks is unsupported on this machine")
        else:
            child.mkdir(parents=True)
        location = child
    profile = location / PROFILE_RELATIVE.name
    profile.write_text("{}\n", encoding="utf-8")
    assert profile.is_file() and not profile.is_symlink()
    assert any(parent.is_symlink() for parent in profile.parents[:4])
    packet = tmp_path / "regular-packet.json"
    packet.write_text("{}\n", encoding="utf-8")
    distribution = _FakeDistribution({PROFILE_RECORD: profile, PACKET_RECORD: packet})
    with pytest.raises(RuntimeError, match="symlinked directory"):
        resolve_default_paths(module_path=module, distribution=distribution)


def test_windows_reparse_directory_detection_on_python311(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Python 3.11 has lstat reparse attributes but no Path.is_junction."""
    monkeypatch.setattr(windows_operator_cli, "_ON_WINDOWS", True)
    flag = windows_operator_cli._WINDOWS_REPARSE_POINT
    redirected = SimpleNamespace(
        is_symlink=lambda: False,
        lstat=lambda: SimpleNamespace(st_file_attributes=flag),
    )
    ordinary = SimpleNamespace(
        is_symlink=lambda: False,
        lstat=lambda: SimpleNamespace(st_file_attributes=0),
    )
    assert windows_operator_cli._is_redirected_installed_directory(redirected)
    assert not windows_operator_cli._is_redirected_installed_directory(ordinary)

    def forbidden_stat() -> None:
        raise PermissionError("uninspectable directory")

    unreadable = SimpleNamespace(is_symlink=lambda: False, lstat=forbidden_stat)
    with pytest.raises(RuntimeError, match="cannot inspect installed canonical"):
        windows_operator_cli._is_redirected_installed_directory(unreadable)
