import pytest
import torch
import torch.nn.functional as F

from twelve_six.distributed.contracts import ParallelPlan
from twelve_six.distributed.rank_layout import RankLayout
from twelve_six.distributed.tensor_parallel import (
    TensorParallelPlan,
    build_megatron_core_tp_adapter,
)
from twelve_six.distributed.tp_probe import run_cpu_tensor_parallel_probe
from twelve_six.model import ModelSpec, TwelveSixDecoder, apply_rope


def _planning_spec_400m() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32768,
        max_seq_len=4096,
        d_model=1024,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        head_dim=64,
        d_ff=5120,
        rope_rotary_dim=64,
    )


def _planning_spec_1b() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32768,
        max_seq_len=4096,
        d_model=2048,
        n_layers=18,
        n_heads=32,
        n_kv_heads=8,
        head_dim=64,
        d_ff=6720,
        rope_rotary_dim=64,
    )


def _tiny_gqa_spec() -> ModelSpec:
    return ModelSpec(
        schema_version=1,
        vocab_size=32,
        max_seq_len=8,
        d_model=16,
        n_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=4,
        d_ff=32,
        rope_rotary_dim=4,
    )


def test_non_frozen_400m_and_1b_gqa_candidates_have_head_aligned_tp_degrees() -> None:
    s5 = _planning_spec_400m()
    assert s5.identity_sha256() == (
        "9abfb6d1ac2e9c28fac20aff4ae804ad54b4102ce6f1bdeeadddf5a56027f28c"
    )
    s5_tp4 = TensorParallelPlan.from_model_spec(s5, 4)
    assert s5_tp4.rank_geometry(3).local_query_heads == 4
    assert s5_tp4.rank_geometry(3).local_kv_heads == 1
    assert s5_tp4.rank_geometry(3).local_d_ff == 1280
    with pytest.raises(ValueError, match="n_kv_heads"):
        TensorParallelPlan.from_model_spec(s5, 8)

    s6 = _planning_spec_1b()
    assert s6.identity_sha256() == (
        "cc64cbe94a461c364f063652098e55bdcf640d4be756ee1e743a23dda3de7261"
    )
    s6_tp8 = TensorParallelPlan.from_model_spec(s6, 8)
    assert s6_tp8.rank_geometry(7).local_query_heads == 4
    assert s6_tp8.rank_geometry(7).local_kv_heads == 1
    assert s6_tp8.rank_geometry(7).local_d_ff == 840
    with pytest.raises(ValueError, match="n_kv_heads"):
        TensorParallelPlan.from_model_spec(s6, 16)


def test_rank_mapping_reuses_existing_d12_layout() -> None:
    spec = _planning_spec_1b()
    plan = TensorParallelPlan.from_model_spec(spec, 4)
    layout = RankLayout(
        ParallelPlan(
            data_parallel=2,
            tensor_parallel=4,
            pipeline_parallel=2,
            context_parallel=1,
        )
    )
    for global_rank in range(layout.plan.world_size):
        coordinate = layout.coordinate(global_rank)
        geometry = plan.rank_geometry_for_global_rank(layout, global_rank)
        assert geometry.tp_rank == coordinate.tp
        assert layout.axis_group(global_rank, "tp")[coordinate.tp] == global_rank


def test_partition_shapes_and_checkpoint_identity_are_topology_specific() -> None:
    spec = _planning_spec_1b()
    tp4 = TensorParallelPlan.from_model_spec(spec, 4)
    tp8 = TensorParallelPlan.from_model_spec(spec, 8)
    rules = {rule.parameter: rule for rule in tp8.parameter_partitions()}
    assert rules["attn.q_proj.weight"].global_shape == (2048, 2048)
    assert rules["attn.q_proj.weight"].local_shape == (256, 2048)
    assert rules["attn.k_proj.weight"].local_shape == (64, 2048)
    assert rules["attn.out_proj.weight"].local_shape == (2048, 256)
    assert rules["mlp.gate_proj.weight"].local_shape == (840, 2048)
    assert rules["mlp.down_proj.weight"].local_shape == (2048, 840)

    assert tp4.model_spec_sha256 == tp8.model_spec_sha256 == spec.identity_sha256()
    assert tp4.identity_sha256 != tp8.identity_sha256
    assert tp4.checkpoint_layout_sha256 != tp8.checkpoint_layout_sha256
    payload = tp8.checkpoint_layout_payload
    assert payload["d05_semantic_model_identity_unchanged"] is True
    assert payload["canonical_fqns_unchanged"] is True
    assert payload["physical_layout_identity_is_topology_specific"] is True


def test_megatron_adapter_boundary_preserves_gqa_geometry() -> None:
    plan = TensorParallelPlan.from_model_spec(_planning_spec_1b(), 8)
    adapter = build_megatron_core_tp_adapter(plan)
    assert adapter.tensor_model_parallel_size == 8
    assert adapter.num_attention_heads == 32
    assert adapter.num_query_groups == 8
    assert adapter.kv_channels == 64
    assert adapter.ffn_hidden_size == 6720
    assert adapter.qkv_fusion_required is True
    assert adapter.swiglu_fc1_fusion_required is True
    assert adapter.payload["status"] == "adapter-boundary-only-not-runtime-tested"


def test_fake_head_aligned_attention_and_mlp_partition_match_full_forward() -> None:
    torch.manual_seed(17)
    spec = _tiny_gqa_spec()
    model = TwelveSixDecoder(spec).eval()
    block = model.blocks[0]
    x = torch.randn(2, 5, spec.d_model)
    plan = TensorParallelPlan.from_model_spec(spec, 2)

    attention_partials = []
    mlp_partials = []
    for tp_rank in range(plan.tp_degree):
        geometry = plan.rank_geometry(tp_rank)
        q_start = geometry.query_head_start * spec.head_dim
        q_stop = geometry.query_head_stop * spec.head_dim
        kv_start = geometry.kv_head_start * spec.head_dim
        kv_stop = geometry.kv_head_stop * spec.head_dim

        q = F.linear(x, block.attn.q_proj.weight[q_start:q_stop])
        k = F.linear(x, block.attn.k_proj.weight[kv_start:kv_stop])
        v = F.linear(x, block.attn.v_proj.weight[kv_start:kv_stop])
        batch, seq_len, _ = q.shape
        q = q.view(batch, seq_len, geometry.local_query_heads, spec.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, geometry.local_kv_heads, spec.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, geometry.local_kv_heads, spec.head_dim).transpose(1, 2)
        cos, sin = block.attn.rope.cos_sin(seq_len, device=x.device, dtype=q.dtype)
        q = apply_rope(q, cos, sin, spec.rope_rotary_dim)
        k = apply_rope(k, cos, sin, spec.rope_rotary_dim)
        repeats = geometry.local_query_heads // geometry.local_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attended = attended.transpose(1, 2).contiguous().view(batch, seq_len, geometry.local_q_dim)
        attention_partials.append(
            F.linear(attended, block.attn.out_proj.weight[:, q_start:q_stop])
        )

        gate = F.linear(
            x,
            block.mlp.gate_proj.weight[geometry.ffn_start : geometry.ffn_stop],
        )
        up = F.linear(
            x,
            block.mlp.up_proj.weight[geometry.ffn_start : geometry.ffn_stop],
        )
        hidden = F.silu(gate) * up
        mlp_partials.append(
            F.linear(
                hidden,
                block.mlp.down_proj.weight[:, geometry.ffn_start : geometry.ffn_stop],
            )
        )

    torch.testing.assert_close(sum(attention_partials), block.attn(x), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(sum(mlp_partials), block.mlp(x), atol=1e-6, rtol=1e-6)


def test_real_local_free_cpu_gloo_dtensor_tensor_parallel_probe() -> None:
    result = run_cpu_tensor_parallel_probe(2)
    assert result.world_size == 2
    assert result.ranks_seen == (0, 1)
    assert result.max_abs_forward_error <= 2e-6
    assert result.parameter_partitioning_passed is True
    assert result.state_dict_schema_stable is True
