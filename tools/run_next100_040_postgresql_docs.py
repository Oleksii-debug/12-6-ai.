#!/usr/bin/env python3
"""Exact-source runner for NEXT100-040 PostgreSQL docs qualification."""
from __future__ import annotations

import qualify_next100_040_postgresql_docs as qualifier

# PostgreSQL's source filename for the Functions and Operators chapter is
# `func.sgml`; keep the semantic bounded set unchanged while binding the
# provenance path that actually exists at REL_18_6 / 724edf9b... .
qualifier.DOC_PATHS = tuple(
    "doc/src/sgml/func.sgml" if path == "doc/src/sgml/functions.sgml" else path
    for path in qualifier.DOC_PATHS
)

raise SystemExit(qualifier.main())
