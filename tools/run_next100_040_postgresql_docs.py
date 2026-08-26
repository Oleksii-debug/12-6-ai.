#!/usr/bin/env python3
"""Exact-source runner for NEXT100-040 PostgreSQL docs qualification."""
from __future__ import annotations

import qualify_next100_040_postgresql_docs as qualifier

# Bind semantic chapter labels used by the qualifier to the canonical PostgreSQL
# source filenames that actually exist at REL_18_6 / 724edf9b... . These are
# source-form substitutions only; rendered HTML/PDF/manpage copies remain excluded.
_PATH_FIXES = {
    "doc/src/sgml/functions.sgml": "doc/src/sgml/func.sgml",
    "doc/src/sgml/indexes.sgml": "doc/src/sgml/indices.sgml",
    "doc/src/sgml/full-text.sgml": "doc/src/sgml/textsearch.sgml",
    "doc/src/sgml/performance-tips.sgml": "doc/src/sgml/perform.sgml",
}

# The first exact-head materialization proved that func.sgml and textsearch.sgml
# leave unresolved residual SGML under POSTGRES_SGML_TEXT_V1. Fail closed by
# excluding those two source objects from this bounded authority instead of
# weakening the quality gate or silently changing normalization semantics.
_EXCLUDED_AFTER_EXACT_QUALITY_PROBE = {
    "doc/src/sgml/func.sgml",
    "doc/src/sgml/textsearch.sgml",
}

canonical_paths = tuple(_PATH_FIXES.get(path, path) for path in qualifier.DOC_PATHS)
qualifier.DOC_PATHS = tuple(
    path for path in canonical_paths if path not in _EXCLUDED_AFTER_EXACT_QUALITY_PROBE
)

raise SystemExit(qualifier.main())
