"""Installed-wheel entry point for the canonical Windows LOCAL_FREE operator.

This module owns packaging/path discovery only. All operator policy, validation,
status, and safe-stop semantics remain in ``windows_operator_preflight``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from importlib import metadata
from pathlib import Path, PurePosixPath

from twelve_six import windows_operator_preflight

_PROFILE_RELATIVE = Path("configs/research/r01_windows_local_free_operator_v1.json")
_PACKET_RELATIVE = Path("configs/research/r01_portable_local_free_run_packet_v1.json")
_INSTALL_DATA_RELATIVE = PurePosixPath("share/twelve-six-ai/configs/research")
_DISTRIBUTION_NAME = "twelve-six-ai"
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


def _distribution_or_fail(
    supplied: metadata.Distribution | None,
) -> metadata.Distribution:
    if supplied is not None:
        return supplied
    try:
        return metadata.distribution(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            f"installed distribution metadata not found: {_DISTRIBUTION_NAME}"
        ) from exc


def _matches_installed_asset(entry: object, expected: PurePosixPath) -> bool:
    raw = str(entry)
    if "\\" in raw:
        return False
    candidate = PurePosixPath(raw)
    if candidate.is_absolute():
        return False
    parts = candidate.parts
    expected_parts = expected.parts
    if len(parts) < len(expected_parts):
        return False
    if parts[-len(expected_parts) :] != expected_parts:
        return False
    prefix = parts[: -len(expected_parts)]
    return all(part == ".." for part in prefix)


def _locate_installed_asset(
    distribution: metadata.Distribution,
    relative: Path,
) -> Path:
    expected = _INSTALL_DATA_RELATIVE / relative.name
    files = distribution.files
    if files is None:
        raise RuntimeError("installed distribution RECORD file list is unavailable")
    matches = [entry for entry in files if _matches_installed_asset(entry, expected)]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one installed canonical asset {expected.as_posix()}, "
            f"found {len(matches)}"
        )
    located = Path(distribution.locate_file(matches[0]))
    if not located.is_file():
        raise RuntimeError(
            f"installed canonical asset is missing or not a regular file: {located}"
        )
    return located


def _installed_asset_defaults(
    distribution: metadata.Distribution | None,
) -> tuple[Path, Path]:
    dist = _distribution_or_fail(distribution)
    return (
        _locate_installed_asset(dist, _PROFILE_RELATIVE),
        _locate_installed_asset(dist, _PACKET_RELATIVE),
    )


def resolve_default_paths(
    *,
    module_path: Path | None = None,
    prefix: Path | None = None,
    home: Path | None = None,
    distribution: metadata.Distribution | None = None,
) -> tuple[Path, Path, Path]:
    """Resolve profile, packet, and state defaults for source or wheel installs.

    ``prefix`` is retained for caller compatibility but intentionally does not
    authorize installed data-file location; the installed distribution's RECORD
    metadata is authoritative for those paths.
    """
    module = module_path or Path(__file__)
    checkout = _source_checkout_root(module)
    if checkout is not None:
        return (
            checkout / _PROFILE_RELATIVE,
            checkout / _PACKET_RELATIVE,
            checkout / _STATE_DIR_NAME,
        )

    profile, packet = _installed_asset_defaults(distribution)
    owner_home = home or Path.home()
    return profile, packet, owner_home / _STATE_DIR_NAME


def _has_option(argv: Sequence[str], option: str) -> bool:
    return any(item == option or item.startswith(f"{option}=") for item in argv)


def build_delegate_argv(
    argv: Sequence[str],
    *,
    module_path: Path | None = None,
    prefix: Path | None = None,
    home: Path | None = None,
    distribution: metadata.Distribution | None = None,
) -> list[str]:
    """Inject only missing install-aware defaults before delegating to the canonical CLI."""
    module = module_path or Path(__file__)
    checkout = _source_checkout_root(module)
    missing_profile = not _has_option(argv, "--profile")
    missing_packet = not _has_option(argv, "--packet")
    missing_state = not _has_option(argv, "--state-dir")

    profile: Path | None = None
    packet: Path | None = None
    if missing_profile or missing_packet:
        if checkout is not None:
            profile = checkout / _PROFILE_RELATIVE
            packet = checkout / _PACKET_RELATIVE
        else:
            dist = _distribution_or_fail(distribution)
            if missing_profile:
                profile = _locate_installed_asset(dist, _PROFILE_RELATIVE)
            if missing_packet:
                packet = _locate_installed_asset(dist, _PACKET_RELATIVE)

    state_dir = (
        checkout / _STATE_DIR_NAME
        if checkout is not None
        else (home or Path.home()) / _STATE_DIR_NAME
    )
    injected: list[str] = []
    for option, path, missing in (
        ("--profile", profile, missing_profile),
        ("--packet", packet, missing_packet),
        ("--state-dir", state_dir, missing_state),
    ):
        if missing:
            assert path is not None
            injected.extend((option, str(path)))
    return [*injected, *argv]


def _bootstrap_error_result(exc: OSError | RuntimeError | ValueError) -> dict[str, object]:
    return {
        "status": "ERROR",
        "error": f"installed_operator_bootstrap_failed:{exc}",
        "launch_authorized": False,
        "training_authorized": False,
        "truth_boundary": dict(windows_operator_preflight._TRUTH_BOUNDARY),
    }


def main(argv: list[str] | None = None) -> int:
    """Delegate to the canonical operator after install-aware path injection."""
    supplied = list(sys.argv[1:] if argv is None else argv)
    # Help must remain available even when a wheel's RECORD or assets are broken.
    # argparse exits before reading either operator input or creating state.
    if "--help" in supplied or "-h" in supplied:
        return windows_operator_preflight.main(supplied)
    try:
        delegated = build_delegate_argv(supplied)
    except (OSError, RuntimeError, ValueError) as exc:
        windows_operator_preflight._print_result(
            _bootstrap_error_result(exc),
            as_json=_has_option(supplied, "--json"),
        )
        return windows_operator_preflight.EXIT_ERROR
    return windows_operator_preflight.main(delegated)


if __name__ == "__main__":
    raise SystemExit(main())
