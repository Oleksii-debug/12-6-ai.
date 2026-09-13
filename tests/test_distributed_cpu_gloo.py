from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.cpu_probe import run_cpu_gloo_probe


def test_local_free_four_rank_gloo_probe() -> None:
    result = run_cpu_gloo_probe(ParallelPlan(data_parallel=2, tensor_parallel=2))
    assert result.world_size == 4
    assert result.all_reduce_sum == 6
    assert result.ranks_seen == (0, 1, 2, 3)
