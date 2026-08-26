"""Verified loader for the compressed RECOVER-174 evaluation implementation."""

from __future__ import annotations

import gzip
import hashlib
from pathlib import Path

_PAYLOAD = Path(__file__).with_name("_recover174_real_core_impl.py.gz")
_COMPRESSED_SHA256 = "d4d1d2dd55bf59c0579a327801bde7acf444c3cea66b24ed2abade3b643c7093"
_SOURCE_SHA256 = "6ebb48592f435762f9698d9292b48f026dedb08b6a7bc42d4c114783bb63e1f0"
_blob = _PAYLOAD.read_bytes()
if hashlib.sha256(_blob).hexdigest() != _COMPRESSED_SHA256:
    raise RuntimeError("RECOVER-174 compressed implementation hash mismatch")
_source = gzip.decompress(_blob)
if hashlib.sha256(_source).hexdigest() != _SOURCE_SHA256:
    raise RuntimeError("RECOVER-174 implementation source hash mismatch")
exec(compile(_source, f"{_PAYLOAD}::source", "exec"), globals(), globals())
