from pathlib import Path


def test_required_control_docs_exist():
    root = Path(__file__).resolve().parents[1]
    for rel in [
        "docs/AUTOPULSE_CONTROL.md",
        "docs/ROLE_REGISTRY.md",
        "docs/TRAINING_AUTHORIZATION.md",
        "docs/S0_EXECUTION_PLAN.md",
    ]:
        assert (root / rel).is_file()
