import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch

from twelve_six import chaos_harness
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


def _rollback_partial(staged: Path, witness: Path, accepted: Path, expected: bytes) -> dict[str, bool]:
    if staged.is_file() and not witness.exists() and staged.read_bytes() == b"partial":
        staged.replace(witness)
    return {
        "rollback": (
            witness.is_file()
            and witness.read_bytes() == b"partial"
            and not staged.exists()
            and accepted.read_bytes() == expected
        ),
    }


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

    def test_injector_death_before_publication_then_after_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, case = self.case("process_kill", {"applied_once": True})
            child_code = r"""
import os
import sys
from pathlib import Path
from twelve_six.chaos_harness import FaultCampaign, FaultCase, _bytes, _digest, run_fault

root = Path(sys.argv[1])
case = FaultCase("process_kill", "process_kill", _digest(_bytes({"applied_once": True})))
campaign = FaultCampaign("fixture-recovery", "a" * 40, (case,))
if sys.argv[2] == "before":
    os.link = lambda *_: os._exit(72)

def inject():
    with (root / "effect.log").open("ab") as stream:
        stream.write(b"applied\n")
        stream.flush()
        os.fsync(stream.fileno())
    os._exit(73)

run_fault(root, campaign, case, inject=inject, recover=lambda: {"applied_once": False})
"""
            env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
            before = subprocess.run(
                [sys.executable, "-c", child_code, str(root), "before"], env=env,
                check=False, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(before.returncode, 72, before.stderr)
            self.assertFalse((root / "effect.log").exists())
            self.assertEqual(campaign_readback(root, campaign)["claimed_pending"], [])

            after = subprocess.run(
                [sys.executable, "-c", child_code, str(root), "after"], env=env,
                check=False, capture_output=True, text=True, timeout=15,
            )
            self.assertEqual(after.returncode, 73, after.stderr)
            pending = campaign_readback(root, campaign)
            self.assertEqual(pending["claimed_pending"], [case.case_id])
            self.assertFalse(pending["complete"])
            effect = root / "effect.log"
            def observe():
                return {"applied_once": effect.read_bytes() == b"applied\n"}

            recovered = recover_fault(root, campaign, case, recover=observe)
            self.assertFalse(recovered["effect_replayed"])
            self.assertTrue(campaign_readback(root, campaign)["complete"])
            with self.assertRaisesRegex(ValueError, "never replay"):
                run_fault(root, campaign, case, inject=lambda: self.fail("replayed"), recover=observe)
            self.assertEqual(effect.read_bytes(), b"applied\n")

    def test_recovery_waits_for_in_flight_injection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, case = self.case("network_loss", {"applied_once": True})
            injecting, release, recovery_started, observed = Event(), Event(), Event(), Event()
            effects: list[str] = []

            def inject():
                injecting.set()
                if not release.wait(8):
                    raise AssertionError("test did not release the injector")
                effects.append("injected")

            def observe():
                return {"applied_once": effects == ["injected"]}

            def recover_in_parallel():
                recovery_started.set()

                def competing_observer():
                    observed.set()
                    return observe()

                return recover_fault(root, campaign, case, recover=competing_observer)

            with ThreadPoolExecutor(max_workers=2) as executor:
                running = executor.submit(run_fault, root, campaign, case, inject=inject, recover=observe)
                try:
                    self.assertTrue(injecting.wait(5))
                    competing = executor.submit(recover_in_parallel)
                    self.assertTrue(recovery_started.wait(5))
                    self.assertFalse(observed.wait(0.25))
                finally:
                    release.set()
                self.assertEqual(running.result(timeout=12), competing.result(timeout=12))
            self.assertEqual(effects, ["injected"])

    def test_durable_failure_cannot_be_followed_by_pass_or_forged_readback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign, case = self.case("disk_corruption", {"recovered": True})
            with self.assertRaisesRegex(ValueError, "recovery witness mismatch"):
                run_fault(root, campaign, case, inject=lambda: None,
                          recover=lambda: {"recovered": False})
            self.assertEqual(campaign_readback(root, campaign)["failed"], [case.case_id])
            with self.assertRaisesRegex(ValueError, "durable failure"):
                recover_fault(root, campaign, case, recover=lambda: self.fail("reobserved"))
            journal = root / campaign.identity_sha256()
            self.assertFalse((journal / (case.case_id + ".receipt")).exists())
            failure = journal / (case.case_id + ".failure")
            failure.write_bytes(b'{"forged":true}\n')
            with self.assertRaisesRegex(ValueError, "failure receipt was corrupted"):
                campaign_readback(root, campaign)

    def test_rollback_requires_independent_evidence_that_interruption_occurred(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged, witness, accepted = root / "staged", root / "witness", root / "accepted"
            accepted.write_bytes(b"known-good")
            campaign, case = self.case("interrupted_publication", {"rollback": True})
            with self.assertRaisesRegex(ValueError, "recovery witness mismatch"):
                run_fault(
                    root, campaign, case, inject=lambda: None,
                    recover=lambda: _rollback_partial(staged, witness, accepted, b"known-good"),
                )
            self.assertEqual(campaign_readback(root, campaign)["failed"], [case.case_id])
            self.assertFalse(witness.exists())

    def test_readback_uses_one_stable_campaign_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            witness = {"observed": True}
            expected = _digest(_bytes(witness))
            first = FaultCase("aa", "network_loss", expected)
            second = FaultCase("bb", "node_failure", expected)
            campaign = FaultCampaign("stable-snapshot", "a" * 40, (first, second))
            run_fault(root, campaign, first, inject=lambda: None, recover=lambda: witness)
            reading_second, release, writing, written = Event(), Event(), Event(), Event()
            original_present = chaos_harness._present

            def pause_before_second_claim(path: Path):
                if path.name == "bb.claim":
                    reading_second.set()
                    if not release.wait(8):
                        raise AssertionError("test did not release the reader")
                return original_present(path)

            def change_both_outcomes():
                writing.set()
                with self.assertRaisesRegex(ValueError, "recovery witness mismatch"):
                    recover_fault(root, campaign, first, recover=lambda: {"observed": False})
                run_fault(root, campaign, second, inject=lambda: None, recover=lambda: witness)
                written.set()

            with (
                patch.object(chaos_harness, "_present", pause_before_second_claim),
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                reading = executor.submit(campaign_readback, root, campaign)
                try:
                    self.assertTrue(reading_second.wait(5))
                    changing = executor.submit(change_both_outcomes)
                    self.assertTrue(writing.wait(5))
                    self.assertFalse(written.wait(0.25))
                finally:
                    release.set()
                earlier = reading.result(timeout=12)
                changing.result(timeout=12)
            self.assertFalse(earlier["complete"])
            self.assertEqual(earlier["passed"], ["aa"])
            self.assertEqual(earlier["missing"], ["bb"])
            later = campaign_readback(root, campaign)
            self.assertFalse(later["complete"])
            self.assertEqual(later["failed"], ["aa"])
            self.assertEqual(later["passed"], ["bb"])

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
            stable_checkpoint, stable_publication = root / "active-checkpoint", root / "active-publication"
            observed_checkpoint, observed_publication = root / "observed-checkpoint", root / "observed-publication"
            stable_checkpoint.write_bytes(b"known-good-checkpoint")
            stable_publication.write_bytes(b"known-good-publication")
            corrupted.write_bytes(b"intact")
            stale = root / "generation"
            stale.write_text("current", encoding="utf-8")
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
                return _rollback_partial(
                    checkpoint, observed_checkpoint, stable_checkpoint, b"known-good-checkpoint",
                )
            def rollback_publication():
                return _rollback_partial(
                    publication, observed_publication, stable_publication, b"known-good-publication",
                )
            def kill(child):
                child.kill()
                child.wait(timeout=5)
            provider_attempts: list[str] = []

            def unavailable():
                provider_attempts.append("unavailable")
                raise ConnectionError("fixture provider unavailable")

            actions = {
                "checkpoint_interruption": (interrupt_checkpoint, rollback_checkpoint),
                "disk_corruption": (lambda: corrupted.write_bytes(b"broken"),
                                    lambda: {"corrupt": corrupted.read_bytes() == b"broken"}),
                "interrupted_publication": (lambda: publication.write_bytes(b"partial"),
                                            rollback_publication),
                "network_loss": (sender.close,
                                 lambda: {"unreachable": receiver.recv(1) == b""}),
                "node_failure": (lambda: kill(primary),
                                 lambda: {"replica_required": primary.poll() is not None
                                          and replica.poll() is None}),
                "process_kill": (lambda: kill(process),
                                 lambda: {"terminated": process.poll() is not None}),
                "provider_failure": (unavailable,
                                     lambda: {"denied": provider_attempts == ["unavailable"]}),
                "stale_state": (lambda: stale.write_text("old", encoding="utf-8"),
                                lambda: {"stale_denied": stale.read_text() != "current"}),
            }
            for case in campaign.cases:
                inject, recover = actions[case.fault]
                receipt = run_fault(root, campaign, case, inject=inject, recover=recover)
                if case.fault in {"checkpoint_interruption", "interrupted_publication"}:
                    self.assertEqual(receipt, recover_fault(root, campaign, case, recover=recover))
            report = campaign_readback(root, campaign)
            self.assertTrue(report["complete"])
            self.assertEqual(len(report["passed"]), 8)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["failed"], [])
            reader = subprocess.run(
                [sys.executable, "-c",
                 ("import json,sys; from pathlib import Path; "
                 "from twelve_six.chaos_harness import parse_campaign_bytes,campaign_readback; "
                 "c=parse_campaign_bytes(Path(sys.argv[2]).read_bytes()); "
                 "print(json.dumps(campaign_readback(Path(sys.argv[1]),c),sort_keys=True))"),
                 str(root), str(path)],
                env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
                check=True, capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(json.loads(reader.stdout), report)


if __name__ == "__main__":
    unittest.main()
