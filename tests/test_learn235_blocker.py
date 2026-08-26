from pathlib import Path

from tools.validate_learn235_prerequisites import validate


def test_learn235_blocker_is_self_consistent_and_fail_closed() -> None:
    report = validate(Path("evidence/learn235/blocker.json"))
    assert report["status"] == "BLOCKED_NO_TERMINAL_LEARN234_IDENTITY"
    assert report["training_executed"] is False
    assert report["optimizer_updates"] == 0
    assert report["required_missing_authority"]["worker_id"] == (
        "LEARN-234-EXTERNAL-REAL-500K"
    )
