import json
import socket
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
    campaign_readback,
    parse_campaign_bytes,
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
            stale = root / "generation"
            stale.write_text("old")
            campaign, case = self.case("stale_state", {"stale_denied": True})
            run_fault(root, campaign, case, inject=lambda: None,
                      recover=lambda: {"stale_denied": stale.read_text() != "current"})

    def test_interrupted_checkpoint_and_provider_failure_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state"
            campaign, case = self.case("checkpoint_interruption", {"rollback": True})
            def interrupted():
                checkpoint.write_bytes(b"partial")
                raise RuntimeError("power loss")
            def rollback_checkpoint():
                checkpoint.unlink(missing_ok=True)
                return {"rollback": not checkpoint.exists()}
            run_fault(root, campaign, case, inject=interrupted,
                      recover=rollback_checkpoint)
            campaign, case = self.case("provider_failure", {"denied": True})
            def unavailable():
                raise ConnectionError("provider unavailable")
            run_fault(root, campaign, case, inject=unavailable,
                      recover=lambda: {"denied": True})
            with self.assertRaisesRegex(ValueError, "never replay"):
                run_fault(root, campaign, case, inject=unavailable,
                          recover=lambda: {"denied": True})
            campaign, case = self.case("node_failure", {"replica_required": True})
            run_fault(root, campaign, case, inject=unavailable,
                      recover=lambda: {"replica_required": True})
            publication = root / "published"
            campaign, case = self.case("interrupted_publication", {"rollback": True})
            def rollback_publication():
                publication.unlink(missing_ok=True)
                return {"rollback": not publication.exists()}
            run_fault(root, campaign, case,
                      inject=lambda: publication.write_bytes(b"partial"),
                      recover=rollback_publication)

    def test_versioned_fault_matrix_readback_and_corruption(self):
        path = Path(__file__).resolve().parents[1] / (
            "configs/plan8/chaos/component-recovery-fixtures-v1.json"
        )
        raw = path.read_bytes()
        campaign = parse_campaign_bytes(raw)
        self.assertEqual(len(campaign.cases), 8)
        self.assertEqual({case.fault for case in campaign.cases}, {
            "process_kill", "network_loss", "disk_corruption", "stale_state",
            "checkpoint_interruption", "provider_failure", "node_failure",
            "interrupted_publication",
        })
        with self.assertRaisesRegex(ValueError, "duplicate campaign JSON field"):
            parse_campaign_bytes(raw.replace(b'"campaign_id":',
                                             b'"campaign_id":"duplicate", "campaign_id":'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = campaign_readback(root, campaign)
            self.assertFalse(report["complete"])
            self.assertEqual(len(report["missing"]), 8)
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            self.addCleanup(lambda: process.kill() if process.poll() is None else None)
            case = next(item for item in campaign.cases if item.fault == "process_kill")
            def inject():
                process.kill()
                process.wait(timeout=5)
            run_fault(root, campaign, case, inject=inject,
                      recover=lambda: {"terminated": process.poll() is not None})
            report = campaign_readback(root, campaign)
            self.assertEqual(report["passed"], ["process_kill"])
            self.assertEqual(len(report["missing"]), 7)
            receipt = root / campaign.identity_sha256() / "process_kill.receipt"
            receipt.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(ValueError, "receipt was corrupted"):
                campaign_readback(root, campaign)

    def test_network_loss_closes_real_local_transport_without_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            receiver, sender = socket.socketpair()
            self.addCleanup(receiver.close)
            self.addCleanup(sender.close)
            campaign, case = self.case("network_loss", {"transport_closed": True})
            receipt = run_fault(
                Path(directory), campaign, case, inject=sender.close,
                recover=lambda: {"transport_closed": receiver.recv(1) == b""},
            )
            self.assertEqual(receipt["effect_replayed"], False)
            with self.assertRaisesRegex(ValueError, "never replay"):
                run_fault(Path(directory), campaign, case,
                          inject=lambda: self.fail("network effect replayed"),
                          recover=lambda: {"transport_closed": True})

    def test_node_failure_local_replica_proxy_keeps_survivor(self):
        with tempfile.TemporaryDirectory() as directory:
            primary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            replica = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            self.addCleanup(lambda: primary.kill() if primary.poll() is None else None)
            self.addCleanup(lambda: replica.kill() if replica.poll() is None else None)
            campaign, case = self.case("node_failure", {"primary_down": True,
                                                       "replica_alive": True})
            def inject():
                primary.kill()
                primary.wait(timeout=5)
            receipt = run_fault(
                Path(directory), campaign, case, inject=inject,
                recover=lambda: {"primary_down": primary.poll() is not None,
                                 "replica_alive": replica.poll() is None},
            )
            self.assertEqual(receipt["effect_replayed"], False)

    def test_complete_versioned_q0_campaign_with_real_local_effects(self):
        path = Path(__file__).resolve().parents[1] / (
            "configs/plan8/chaos/component-recovery-fixtures-v1.json"
        )
        campaign = parse_campaign_bytes(path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint, corrupted, publication = (
                root / "checkpoint", root / "artifact", root / "publication"
            )
            corrupted.write_bytes(b"intact")
            stale = root / "generation"
            stale.write_text("old", encoding="utf-8")
            primary = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            replica = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            receiver, sender = socket.socketpair()
            for child in (primary, replica, process):
                self.addCleanup(lambda item=child: item.kill() if item.poll() is None else None)
            self.addCleanup(receiver.close)
            self.addCleanup(sender.close)

            def interrupt_checkpoint():
                checkpoint.write_bytes(b"partial")
                raise RuntimeError("checkpoint interrupted")
            def rollback_checkpoint():
                checkpoint.unlink(missing_ok=True)
                return {"rollback": not checkpoint.exists()}
            def rollback_publication():
                publication.unlink(missing_ok=True)
                return {"rollback": not publication.exists()}
            def kill(child):
                child.kill()
                child.wait(timeout=5)
            def unavailable():
                raise ConnectionError("fixture provider unavailable")

            actions = {
                "checkpoint_interruption": (interrupt_checkpoint, rollback_checkpoint),
                "disk_corruption": (lambda: corrupted.write_bytes(b"broken"),
                                    lambda: {"corrupt": corrupted.read_bytes() != b"intact"}),
                "interrupted_publication": (lambda: publication.write_bytes(b"partial"),
                                            rollback_publication),
                "network_loss": (sender.close,
                                 lambda: {"unreachable": receiver.recv(1) == b""}),
                "node_failure": (lambda: kill(primary),
                                 lambda: {"replica_required": primary.poll() is not None
                                          and replica.poll() is None}),
                "process_kill": (lambda: kill(process),
                                 lambda: {"terminated": process.poll() is not None}),
                "provider_failure": (unavailable, lambda: {"denied": True}),
                "stale_state": (lambda: None,
                                lambda: {"stale_denied": stale.read_text() != "current"}),
            }
            for case in campaign.cases:
                inject, recover = actions[case.fault]
                run_fault(root, campaign, case, inject=inject, recover=recover)
            report = campaign_readback(root, campaign)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["passed"]), 8)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["failed"], [])


if __name__ == "__main__":
    unittest.main()
