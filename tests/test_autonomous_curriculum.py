"""Plan 6 Section 11 LOCAL_FREE capability/holdout/curriculum qualification."""
from dataclasses import asdict, replace

import pytest

from twelve_six.autonomous_curriculum import (
    CapabilityArea,
    CurriculumProposal,
    CurriculumRecipe,
    _mac,
    measure_capability_gaps,
    plan_curriculum,
    verify_curriculum_restart,
)
from twelve_six.continual_learning import CapabilityProbe, FrozenCapabilitySuite
from twelve_six.post_base_instruction import canonical_digest as digest
from twelve_six.post_base_reasoning import TinyPolicy

H = "a" * 64
KEY = b"v" * 32


def setup():
    policy = TinyPolicy(("known", "unknown", "other"), ("A", "B"),
                        ((3., 0.), (3., 0.), (3., 0.)))
    known = FrozenCapabilitySuite("frozen_benchmark", "holdout", H, (
        CapabilityProbe("bench_probe", "known", digest({"action": "A"})),
    ))
    gen = FrozenCapabilitySuite("frozen_generalization", "holdout", H, (
        CapabilityProbe("gen_probe", "unknown", digest({"action": "B"})),
    ))
    area = CapabilityArea("planning", known, gen)
    trusted_suites = frozenset({known.identity(), gen.identity()})
    snapshot = measure_capability_gaps(policy, (area,), trusted_suite_roots=trusted_suites)
    proposal = CurriculumProposal("candidate-A", "planning", "data", "generator",
                                  "trusted-independent", H, H, H,
                                  ("train_probe",), 80, 80, 10,
                                  ("new_training_context",))
    proposal = replace(proposal, signature=_mac(KEY, proposal.payload()))
    other = replace(proposal, proposal_id="candidate-B", kind="experiment", cost_units=50)
    other = replace(other, signature=_mac(KEY, other.payload()))
    recipe = CurriculumRecipe(snapshot.manifest_sha256, max_targets=2, budget_units=30)
    kwargs = {
        "trusted_verifier_id": "trusted-independent",
        "trusted_verifier_version_sha256": H,
        "trusted_proposal_roots": frozenset({H}),
        "verifier_key": KEY,
        "champion": policy,
        "trusted_suite_roots": trusted_suites,
    }
    return policy, (area,), trusted_suites, snapshot, (proposal, other), recipe, kwargs


def run(args=None):
    a = args or setup()
    return plan_curriculum(a[3], a[1], a[4], a[5], **a[6])


def test_independent_holdout_exposes_hidden_gap_and_replay():
    a = setup()
    gap = a[3].gaps[0]
    assert gap.benchmark_accuracy == 1 and gap.generalization_accuracy == 0
    assert gap.uncertainty > 0 and gap.conservative_gap > 0.8
    result = run(a)
    assert result.target_ids == ("candidate-A",)
    assert result.total_cost_units == 10
    assert not result.authorized_paid_compute and not result.authorized_training
    assert not result.promotion_authorized
    assert result == run(a)
    assert verify_curriculum_restart(a[3], a[1], a[4], a[5], result, **a[6])


def test_holdout_probe_and_untrusted_suite_rejected():
    a = setup()
    with pytest.raises(ValueError):
        measure_capability_gaps(a[0], a[1], trusted_suite_roots=frozenset({H}))
    duplicated = replace(a[1][0], generalization=replace(
        a[1][0].generalization,
        probes=(CapabilityProbe("bench_probe", "unknown", digest({"action": "B"})),)))
    with pytest.raises(ValueError):
        measure_capability_gaps(a[0], (duplicated,), trusted_suite_roots=a[2])
    same_context = replace(a[1][0], generalization=replace(
        a[1][0].generalization,
        probes=(CapabilityProbe("new_probe_id", "known", digest({"action": "B"})),)))
    roots = frozenset({same_context.benchmark.identity(),
                       same_context.generalization.identity()})
    with pytest.raises(ValueError, match="must be disjoint"):
        measure_capability_gaps(a[0], (same_context,), trusted_suite_roots=roots)


def test_no_eval_train_task_leakage():
    a = setup()
    leaked = replace(a[4][0], train_task_ids=("gen_probe",))
    leaked = replace(leaked, signature=_mac(KEY, leaked.payload()))
    with pytest.raises(ValueError):
        plan_curriculum(a[3], a[1], (leaked,), a[5], **a[6])


def test_forged_evidence_and_self_assessment_rejected():
    a = setup()
    for field in ({"signature": H}, {"expected_gain": 99},
                  {"evidence_sha256": "b" * 64}):
        bad = replace(a[4][0], **field)
        with pytest.raises(ValueError):
            plan_curriculum(a[3], a[1], (bad,), a[5], **a[6])
    self_issued = replace(a[4][0], verifier_id=a[4][0].producer_id)
    self_issued = replace(self_issued, signature=_mac(KEY, self_issued.payload()))
    with pytest.raises(ValueError):
        plan_curriculum(a[3], a[1], (self_issued,), a[5], **a[6])


def test_stale_snapshot_and_frozen_area_version_rejected():
    a = setup()
    with pytest.raises(ValueError):
        plan_curriculum(a[3], a[1], a[4], replace(a[5], snapshot_sha256=H), **a[6])
    changed = replace(a[1][0], target_accuracy=0.7)
    with pytest.raises(ValueError):
        plan_curriculum(a[3], (changed,), a[4], a[5], **a[6])


def test_bounded_cost_gain_quality_proposals():
    a = setup()
    for field in ({"cost_units": 0}, {"cost_units": True},
                  {"expected_gain": 101}, {"evidence_quality": 0},
                  {"split": "eval"}, {"train_task_ids": ("x", "x")}):
        invalid = replace(a[4][0], **field)
        with pytest.raises(ValueError):
            plan_curriculum(a[3], a[1], (invalid,), a[5], **a[6])
    for field in ({"max_targets": 9}, {"budget_units": -1},
                  {"max_targets": True}):
        with pytest.raises(ValueError):
            replace(a[5], **field).validate()


def test_budget_and_stop_do_not_dispatch_training():
    a = setup()
    empty = plan_curriculum(a[3], a[1], a[4], replace(a[5], budget_units=1), **a[6])
    assert not empty.target_ids and empty.state == "STOP_NO_EVIDENCED_VALUE"
    assert empty.total_cost_units == 0 and not empty.authorized_training


def test_replay_tamper_denied():
    a = setup()
    actual = run(a)
    for bad in (replace(actual, manifest_sha256=H),
                replace(actual, authorized_training=True),
                replace(actual, promotion_authorized=True)):
        with pytest.raises(ValueError):
            verify_curriculum_restart(a[3], a[1], a[4], a[5], bad, **a[6])


def test_forged_self_resealed_snapshot_cannot_change_capability_gap():
    a = setup()
    from_dataclass = a[3]
    forged_gap = replace(from_dataclass.gaps[0], conservative_gap=0.0,
                         generalization_accuracy=1.0)
    forged_receipt = digest({
        "champion": from_dataclass.champion_sha256,
        "suites": from_dataclass.suite_roots_sha256,
        "areas": from_dataclass.areas_sha256,
        "gaps": [asdict(forged_gap)],
    })
    forged = replace(from_dataclass, gaps=(forged_gap,),
                     manifest_sha256=forged_receipt)
    changed_recipe = replace(a[5], snapshot_sha256=forged_receipt)
    with pytest.raises(ValueError, match="forged/stale"):
        plan_curriculum(forged, a[1], a[4], changed_recipe, **a[6])


def test_training_alias_of_holdout_context_denied_even_if_task_id_changes():
    a = setup()
    forged = replace(a[4][0], train_task_ids=("fresh_train_id",),
                     train_context_ids=("unknown",))
    forged = replace(forged, signature=_mac(KEY, forged.payload()))
    with pytest.raises(ValueError, match="holdout-leaking"):
        plan_curriculum(a[3], a[1], (forged,), a[5], **a[6])


def test_unbound_training_contexts_fail_closed():
    a = setup()
    for ctxs in ((), ("x", "x"), ["new_training_context"]):
        invalid = replace(a[4][0], train_context_ids=ctxs)
        with pytest.raises(ValueError):
            plan_curriculum(a[3], a[1], (invalid,), a[5], **a[6])
