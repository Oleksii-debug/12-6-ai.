import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from twelve_six.chaos_harness import (
    FaultCampaign,
    FaultCase,
    _bytes,
    _digest,
    recover_fault,
    run_fault,
)


class ChaosHarnessTests(unittest.TestCase):
    def case(self, fault, witness):
        case = FaultCase(fault, fault, _digest(_bytes(witness)))
        return FaultCampaign("fixture-recovery", "a" * 40, (case,)), case

    def test_process_kill_and_no_duplicate_effect(self):
        with tempfile.TemporaryDirectory() as directory:
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            campaign, case = self.case("process_kill", {"terminated": True})
            def inject():
                process.kill()
                process.wait(timeout=5)
            receipt = run_fault(Path(directory), campaign, case, inject=inject,
                                recover=lambda: {"terminated": process.poll() is not None})
            self.assertEqual(receipt["evidence_level"], "Q0")
            with self.assertRaisesRegex(ValueError, "never replay"):
                run_fault(Path(directory), campaign, case, inject=lambda: self.fail("replayed"),
                          recover=lambda: {"terminated": True})
            self.assertEqual(receipt, recover_fault(
                Path(directory), campaign, case, recover=lambda: {"terminated": True}))

    def test_network_loss_disk_corruption_and_stale_checkpoint_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, case = self.case("network_loss", {"unreachable": True})
            def refused():
                raise ConnectionError("fixture transport unavailable")
            run_fault(root, campaign, case, inject=refused,
                      recover=lambda: {"unreachable": True})

            artifact = root / "checkpoint.bin"
            artifact.write_bytes(b"correct")
            campaign, case = self.case("disk_corruption", {"corrupt": True})
            run_fault(root, campaign, case, inject=lambda: artifact.write_bytes(b"broken"),
                      recover=lambda: {"corrupt": artifact.read_bytes() != b"correct"})
            with self.assertRaisesRegex(ValueError, "witness mismatch"):
                recover_fault(root, campaign, case, recover=lambda: {"corrupt": False})
            failure = root / campaign.identity_sha256() / (case.case_id + ".failure")
            self.assertEqual(json.loads(failure.read_text())["case_id"], case.case_id)
            receipt = root / campaign.identity_sha256() / (case.case_id + ".receipt")
            receipt.write_text(json.dumps({"forged": True}))
            with self.assertRaisesRegex(ValueError, "conflicting durable"):
                recover_fault(root, campaign, case, recover=lambda: {"corrupt": True})

    def test_interrupted_checkpoint_and_provider_failure_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state"
            campaign, case = self.case("checkpoint_interruption", {"rollback": True})
            def interrupted():
                checkpoint.write_bytes(b"partial")
                raise RuntimeError("power loss")
            run_fault(root, campaign, case, inject=interrupted,
                      recover=lambda: {"rollback": checkpoint.read_bytes() == b"partial"})
            checkpoint.unlink()
            campaign, case = self.case("provider_failure", {"denied": True})
            def unavailable():
                raise ConnectionError("provider unavailable")
            run_fault(root, campaign, case, inject=unavailable,
                      recover=lambda: {"denied": True})
            with self.assertRaisesRegex(ValueError, "never replay"):
                run_fault(root, campaign, case, inject=unavailable,
                          recover=lambda: {"denied": True})


if __name__ == "__main__":
    unittest.main()
