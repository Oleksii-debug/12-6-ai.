"""Conservative Hugging Face directory export for verified 12-6 checkpoints."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .core import MANIFEST_NAME, WEIGHTS_NAME, canonical_json_bytes, verify_checkpoint


def export_hf_directory(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    hf_config: Mapping[str, Any],
    overwrite: bool = False,
) -> Path:
    """Create the standard single-file SafeTensors/config layout.

    This function guarantees directory/file layout and provenance preservation.
    It does not claim Transformers architecture compatibility; D01/D07 must
    provide and test the architecture-specific config/model registration.
    """

    source = Path(checkpoint_dir)
    verify_checkpoint(source)
    destination = Path(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"export destination already exists: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    shutil.copy2(source / WEIGHTS_NAME, destination / "model.safetensors")
    (destination / "config.json").write_bytes(canonical_json_bytes(dict(hf_config)) + b"\n")
    shutil.copy2(source / MANIFEST_NAME, destination / "12-6-checkpoint-manifest.json")
    return destination
