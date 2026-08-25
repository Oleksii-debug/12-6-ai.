"""Shared test bootstrap for branch-local LEARN-04 DATA-10 composition."""

from twelve_six import learn04_1m_experiment as _learn04

_learn04.TRAIN_RECORDS = tuple(
    (stratum, record_id, text.rstrip("\n") if stratum == "code" else text)
    for stratum, record_id, text in _learn04.TRAIN_RECORDS
)
