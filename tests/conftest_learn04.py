"""LEARN-04 test helper; mirrors DATA-10 code-layout normalization."""

from twelve_six import learn04_1m_experiment as experiment

experiment.TRAIN_RECORDS = tuple(
    (stratum, record_id, text.rstrip("\n") if stratum == "code" else text)
    for stratum, record_id, text in experiment.TRAIN_RECORDS
)
