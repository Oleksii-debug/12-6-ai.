"""Installed-wheel entry point for the canonical Windows LOCAL_FREE operator.

This module owns packaging/path discovery only. All operator policy, validation,
status, and safe-stop semantics remain in ``windows_operator_preflight``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

from twelve_six import windows_operator_preflight

_PROFILE_RELATIVE = Path("configs/research/r01_windows_local_free_operator_v1.json")
_PACKET_RELATIVE = Path("configs/research/r01_portable_local_free_run_packet_v1.json")
_INSTALL_SHARE = Path("share/twelve-six-ai")
_STATE_DIR_NAME = ".twelve-six-local"


def _source_checkout_root(module_path: Path) -> Path | None:
    """Return a repository root only when the canonical source assets are present."""
    resolved = module_path.resolve()
    try:
        candidate = resolved.parents[2]
    except IndexError:
        return None
    if not (candidate / "pyproject.toml").is_file():
        return None
    if not (candidate / _PROFILE_RELATIVE).is_file():
        return None
    if not (candidate / _PACKET_RELATIVE).is_file():
        return None
    return candidate


def resolve_default_paths(
    *,
    module_path: Path | None = None,
    prefix: Path | None = None,
    home: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve profile, packet, and state defaults for source or wheel installs."""
    module = module_path or Path(__file__)
    checkout = _source_checkout_root(module)
    if checkout is not None:
        return (
            checkout / _PROFILE_RELATIVE,
            checkout / _PACKET_RELATIVE,
            checkout / _STATE_DIR_NAME,
        )

    install_root = (prefix or Path(sys.prefix)) / _INSTALL_SHARE
    owner_home = home or Path.home()
    return (
        install_root / _PROFILE_RELATIVE,
        install_root / _PACKET_RELATIVE,
        owner_home / _STATE_DIR_NAME,
    )


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def build_delegate_argv(
    argv: Sequence[str],
    *,
    module_path: Path | None = None,
    prefix: Path | None = None,
    home: Path | None = None,
) -> list[str]:
    """Inject only missing install-aware defaults before delegating to the canonical CLI."""
    profile, packet, state_dir = resolve_default_paths(
        module_path=module_path,
        prefix=prefix,
        home=home,
    )
    injected: list[str] = []
    for option, path in (
        ("--profile", profile),
        ("--packet", packet),
        ("--state-dir", state_dir),
    ):
        if not _has_option(argv, option):
            injected.extend((option, str(path)))
    return [*injected, *argv]


def main(argv: list[str] | None = None) -> int:
    """Delegate to the canonical operator after install-aware path injection."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    return windows_operator_preflight.main(build_delegate_argv(supplied))


if __name__ == "__main__":
    raise SystemExit(main())
