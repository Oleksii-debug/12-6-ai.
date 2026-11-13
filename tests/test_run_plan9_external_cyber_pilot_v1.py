from types import SimpleNamespace

from tools import run_plan9_external_cyber_pilot_v1 as operator


def test_pilot_operator_reopens_in_new_process_and_verifies_final_checkpoint(monkeypatch, tmp_path):
    initial = {"phase": "PREPARED", "committed_step": 0, "tokens_seen": 0, "receipt_sha256": "initial"}
    paused = {"phase": "PAUSED", "committed_step": 2, "tokens_seen": 1303, "receipt_sha256": "paused"}
    complete = {"phase": "COMPLETED", "committed_step": 4, "tokens_seen": 2696, "receipt_sha256": "complete"}
    proof = {"checkpoint_verified": True}
    plan = {"packet_sha256": "packet", "run_manifest_sha256": "run"}
    calls = []
    responses = iter((plan, initial, paused, complete, proof))

    def invoke(action, *arguments):
        calls.append((action, arguments))
        return next(responses)

    install = tmp_path / "install-report.json"
    install.write_bytes(b"{}")
    monkeypatch.setattr(operator.sys, "platform", "linux")
    monkeypatch.setattr(operator, "_invoke", invoke)
    args = SimpleNamespace(
        execute=True, output=tmp_path, scratch=tmp_path, bundle=tmp_path, lock_profile="linux-x86_64",
        packet_sha256="packet", install_report=install, benchmark_receipt=tmp_path / "benchmark.json",
        benchmark_receipt_sha256="measurement",
    )
    assert operator.run(args) == proof
    assert [action for action, _ in calls] == ["plan", "prepare", "run", "run", "verify-checkpoint"]
    assert [arguments[-1] for _, arguments in calls[2:]] == ["initial", "paused", "complete"]
    assert (tmp_path / "checkpoint-proof.json").read_bytes() == operator.canonical(proof)
