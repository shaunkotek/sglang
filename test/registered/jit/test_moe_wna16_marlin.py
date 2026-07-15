import itertools
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from sgl_kernel.scalar_type import scalar_types

from sglang.jit_kernel.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization import modelopt_quant
from sglang.srt.layers.quantization.marlin_utils_fp4 import (
    prepare_moe_nvfp4_layer_for_marlin,
)
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.utils.common import is_sm80_supported, is_sm90_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_marlin_utils import (
    awq_marlin_quantize,
    make_nvfp4_weight_and_ref,
    marlin_quantize,
)

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


def _has_aot_moe_wna16_marlin_gemm() -> bool:
    return hasattr(torch.ops.sgl_kernel, "moe_wna16_marlin_gemm") and hasattr(
        torch.ops.sgl_kernel.moe_wna16_marlin_gemm, "default"
    )


AOT_AVAILABLE = _has_aot_moe_wna16_marlin_gemm()


class _ModelOptNvFp4ConfigStub:
    def __init__(self, quant_method):
        self.quant_method = quant_method

    def get_name(self):
        return "modelopt_fp4"

    def get_quant_method(self, _layer, _prefix):
        return self.quant_method


def _make_modelopt_fused_moe(*, configured_backend, routed_scaling_factor):
    """Build enough of FusedMoE to exercise backend resolution and scale policy."""
    quant_method = object.__new__(ModelOptNvFp4FusedMoEMethod)
    quant_method.create_weights = Mock()
    quant_config = _ModelOptNvFp4ConfigStub(quant_method)
    parallel = SimpleNamespace(
        moe_ep_size=1,
        moe_ep_rank=0,
        moe_tp_size=1,
        moe_tp_rank=0,
    )
    a2a_backend = SimpleNamespace(is_ascend_fuseep=lambda: False)

    with (
        patch.object(
            fused_moe_layer,
            "get_moe_runner_backend",
            return_value=configured_backend,
        ),
        patch.object(
            modelopt_quant,
            "get_moe_runner_backend",
            return_value=configured_backend,
        ),
        # The backend resolver itself remains real. Avoid constructing its
        # heavyweight runner because this test invokes Marlin directly below.
        patch.object(modelopt_quant, "MoeRunner", return_value=Mock()),
        patch.object(fused_moe_layer, "get_parallel", return_value=parallel),
        patch.object(
            fused_moe_layer, "create_kt_config_from_server_args", return_value=None
        ),
        patch.object(
            fused_moe_layer,
            "get_server_args",
            return_value=SimpleNamespace(moe_runner_backend=configured_backend.value),
        ),
        patch.object(fused_moe_layer, "create_moe_dispatcher", return_value=Mock()),
        patch.object(fused_moe_layer, "get_moe_a2a_backend", return_value=a2a_backend),
        patch.object(fused_moe_layer, "print_info_once"),
    ):
        return fused_moe_layer.FusedMoE(
            num_experts=4,
            hidden_size=256,
            intermediate_size=192,
            layer_id=0,
            top_k=2,
            quant_config=quant_config,
            routed_scaling_factor=routed_scaling_factor,
            is_gated=False,
        )


def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def _get_scalar_type(num_bits: int, has_zp: bool):
    if has_zp:
        assert num_bits == 4
        return scalar_types.uint4
    else:
        return scalar_types.uint4b8 if num_bits == 4 else scalar_types.uint8b128


def _setup_moe_weights(e, n, k, quant_type, group_size, act_order, dtype):
    """Set up quantized MoE weights for a single gate (e experts, output n, input k)."""
    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 20

    w_ref_l = []
    qweight_l = []
    scales_l = []
    zeros_l = []
    g_idx_l = []
    sort_indices_l = []

    for i in range(e):
        if has_zp:
            w_ref, qweight, scales, zeros = awq_marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            zeros_l.append(zeros)
        else:
            test_perm = torch.randperm(k)
            w_ref, qweight, scales, g_idx, sort_indices, _ = marlin_quantize(
                w[i].transpose(1, 0), quant_type, group_size, act_order, test_perm
            )
            w_ref_l.append(w_ref.T)
            qweight_l.append(qweight)
            scales_l.append(scales)
            g_idx_l.append(g_idx)
            sort_indices_l.append(sort_indices)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweight_l).contiguous()
    scales = stack_and_dev(scales_l)
    g_idx = stack_and_dev(g_idx_l) if g_idx_l else None
    sort_indices = stack_and_dev(sort_indices_l) if sort_indices_l else None
    zeros = stack_and_dev(zeros_l) if zeros_l else None

    return w_ref, qweight, scales, zeros, g_idx, sort_indices


def _run_single_gemm(
    fn,
    a,
    c,
    qweight,
    scales,
    zeros,
    g_idx,
    sort_indices,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    quant_type,
    block_size_m,
    topk,
    size_m,
    size_n,
    size_k,
    mul_topk_weights,
    is_k_full,
    use_atomic_add,
):
    return fn(
        a,
        c,
        qweight,
        None,  # b_bias
        scales,
        None,  # global_scale
        zeros,
        g_idx,
        sort_indices,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=mul_topk_weights,
        is_ep=False,
        b_q_type=quant_type,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def _run_single_gemm_aot(
    a,
    c,
    qweight,
    scales,
    zeros,
    g_idx,
    sort_indices,
    workspace,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    topk_weights,
    quant_type,
    block_size_m,
    topk,
    size_m,
    size_n,
    size_k,
    mul_topk_weights,
    is_k_full,
    use_atomic_add,
):
    return torch.ops.sgl_kernel.moe_wna16_marlin_gemm.default(
        a,
        c,
        qweight,
        None,  # b_bias
        scales,
        None,  # global_scale
        zeros,
        g_idx,
        sort_indices,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        moe_block_size=block_size_m,
        top_k=topk,
        mul_topk_weights=mul_topk_weights,
        is_ep=False,
        b_q_type_id=quant_type.id,
        size_m=size_m,
        size_n=size_n,
        size_k=size_k,
        is_k_full=is_k_full,
        use_atomic_add=use_atomic_add,
        use_fp32_reduce=True,
        is_zp_float=False,
    )


def generate_test_cases():
    m_list = [1, 123]
    n_list = [128, 1024]
    k_list = [256]
    e_list = [4]
    topk_list = [2]
    dtype_list = [torch.float16, torch.bfloat16]
    group_size_list = [128]
    act_order_list = [False, True]
    quant_type_list = [scalar_types.uint4, scalar_types.uint4b8]

    all_combinations = itertools.product(
        m_list,
        n_list,
        k_list,
        e_list,
        topk_list,
        dtype_list,
        group_size_list,
        act_order_list,
        quant_type_list,
    )

    def is_valid(m, n, k, e, topk, dtype, group_size, act_order, quant_type):
        has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]
        if act_order:
            if group_size == -1 or group_size == k:
                return False
            if has_zp:
                return False
        if group_size > 0 and k % group_size != 0:
            return False
        return True

    return [case for case in all_combinations if is_valid(*case)]


TEST_CASES = generate_test_cases()


@pytest.mark.parametrize(
    "m,n,k,e,topk,dtype,group_size,act_order,quant_type",
    TEST_CASES,
    ids=[
        f"m{c[0]}_n{c[1]}_k{c[2]}_e{c[3]}_t{c[4]}_{c[5].__name__ if hasattr(c[5], '__name__') else str(c[5]).split('.')[-1]}_g{c[6]}_act{c[7]}_{c[8]}"
        for c in TEST_CASES
    ],
)
def test_moe_wna16_marlin_gemm(
    m, n, k, e, topk, dtype, group_size, act_order, quant_type
):
    if not AOT_AVAILABLE:
        pytest.skip("sgl_kernel moe_wna16_marlin_gemm AOT op not available")

    torch.manual_seed(0)

    has_zp = quant_type in [scalar_types.uint4, scalar_types.uint8]

    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

    # Set up quantized weights for first gemm (gate_up: output 2*n, input k)
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, 2 * n, k, quant_type, group_size, act_order, dtype
    )

    # Compute block_size_m
    for block_size_m in [8, 16, 32, 48, 64]:
        if m * topk / e / block_size_m < 0.9:
            break

    # Align tokens
    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_size_m, e
    )

    # Workspace
    sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    max_workspace_size = (max(2 * n, k) // 64) * (
        sorted_token_ids.size(0) // block_size_m
    )
    max_workspace_size = min(max_workspace_size, sms * 4)
    workspace = torch.zeros(
        max_workspace_size, dtype=torch.int, device="cuda", requires_grad=False
    )

    use_atomic_add = (
        dtype == torch.half or torch.cuda.get_device_capability("cuda")[0] >= 9
    )

    scalar_type = _get_scalar_type(4, has_zp)

    # --- Run JIT kernel ---
    c_jit = torch.empty((m * topk, 2 * n), dtype=dtype, device="cuda")
    c_jit = _run_single_gemm(
        moe_wna16_marlin_gemm,
        a,
        c_jit,
        qweight1,
        scales1,
        zeros1,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        scalar_type,
        block_size_m,
        topk,
        m,
        2 * n,
        k,
        False,
        True,
        use_atomic_add,
    )

    torch.cuda.synchronize()

    # --- Check bitwise equality with AOT kernel ---
    c_aot = torch.empty((m * topk, 2 * n), dtype=dtype, device="cuda")
    c_aot = _run_single_gemm_aot(
        a,
        c_aot,
        qweight1,
        scales1,
        zeros1,
        g_idx1,
        sort_indices1,
        workspace,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        topk_weights,
        scalar_type,
        block_size_m,
        topk,
        m,
        2 * n,
        k,
        False,
        True,
        use_atomic_add,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(c_jit, c_aot, rtol=0, atol=0)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="Non-gated NVFP4 Marlin fallback test requires CUDA SM8X/SM9X",
)
def test_fused_marlin_moe_non_gated_relu2():
    torch.manual_seed(0)

    m = 17
    n = 128
    k = 256
    e = 4
    topk = 2
    dtype = torch.float16
    group_size = 128
    quant_type = scalar_types.uint4b8
    routed_scaling_factor = 2.0

    hidden_states = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w_ref1, qweight1, scales1, zeros1, g_idx1, sort_indices1 = _setup_moe_weights(
        e, n, k, quant_type, group_size, False, dtype
    )
    w_ref2, qweight2, scales2, zeros2, g_idx2, sort_indices2 = _setup_moe_weights(
        e, k, n, quant_type, group_size, False, dtype
    )

    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=qweight1,
        w2=qweight2,
        w1_scale=scales1,
        w2_scale=scales2,
        gating_output=router_logits,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        g_idx1=g_idx1,
        g_idx2=g_idx2,
        sort_indices1=sort_indices1,
        sort_indices2=sort_indices2,
        w1_zeros=zeros1,
        w2_zeros=zeros2,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=routed_scaling_factor,
        activation="relu2",
        is_gated=False,
    )

    output_ref = torch.zeros_like(hidden_states)
    for token_idx in range(m):
        for route_idx in range(topk):
            expert_id = topk_ids[token_idx, route_idx]
            intermediate = hidden_states[token_idx] @ w_ref1[expert_id].T
            intermediate = torch.square(torch.relu(intermediate))
            routed = intermediate @ w_ref2[expert_id].T
            output_ref[token_idx] += routed * topk_weights[token_idx, route_idx]
    output_ref *= routed_scaling_factor

    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref, rtol=0.04, atol=0.04)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="NVFP4 Marlin MoE padding test requires CUDA SM8X/SM9X",
)
def test_fused_marlin_moe_nvfp4_non_gated_padded_intermediate_launches():
    torch.manual_seed(0)

    m = 17
    intermediate_size = 192
    hidden_size = 256
    e = 4
    topk = 2
    dtype = torch.bfloat16
    nvfp4_group_size = 16

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=nvfp4_group_size)
    layer.moe_runner_config = SimpleNamespace(is_gated=False)
    layer.params_dtype = dtype
    layer.intermediate_size_per_partition = intermediate_size
    layer.w13_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (e, intermediate_size, hidden_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.randint(
            0,
            256,
            (e, hidden_size, intermediate_size // 2),
            device="cuda",
            dtype=torch.uint8,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.rand(
            (e, intermediate_size, hidden_size // nvfp4_group_size),
            device="cuda",
            dtype=dtype,
        ),
        requires_grad=False,
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.rand(
            (e, hidden_size, intermediate_size // nvfp4_group_size),
            device="cuda",
            dtype=dtype,
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.ones((e,), device="cuda", dtype=dtype), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.ones((e,), device="cuda", dtype=dtype), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)

    assert layer.w13_weight.shape[1] * 16 == 256
    assert layer.w2_weight.shape[1] * 16 == 256

    hidden_states = torch.randn((m, hidden_size), device="cuda", dtype=dtype) / 10

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    score_softmax = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score_softmax, topk)

    out = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=score,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w1_global_scale=layer.w13_weight_scale_2,
        w2_global_scale=layer.w2_weight_scale_2,
        workspace=layer.workspace,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=1.0,
        activation="relu2",
        is_gated=False,
    )

    torch.cuda.synchronize()
    assert out.shape == (m, hidden_size)


@pytest.mark.skipif(
    not (is_sm80_supported() or is_sm90_supported()),
    reason="ModelOpt NVFP4 Marlin scale test requires CUDA SM80, SM86, or SM90",
)
@pytest.mark.parametrize(
    "configured_backend",
    [MoeRunnerBackend.MARLIN, MoeRunnerBackend.AUTO],
    ids=["explicit-marlin", "auto-resolves-to-marlin"],
)
def test_modelopt_nvfp4_marlin_routed_scale_applied_once(configured_backend):
    torch.manual_seed(0)

    m = 17
    intermediate_size = 192
    hidden_size = 256
    e = 4
    topk = 2
    dtype = torch.bfloat16
    group_size = 16
    routed_scaling_factor = 2.0

    w13_packed_l, w13_scales_l, w13_gscale_l, w13_ref_l = [], [], [], []
    w2_packed_l, w2_scales_l, w2_gscale_l, w2_ref_l = [], [], [], []
    for _ in range(e):
        packed, scales, gscale, ref = make_nvfp4_weight_and_ref(
            intermediate_size, hidden_size, dtype, group_size=group_size
        )
        w13_packed_l.append(packed)
        w13_scales_l.append(scales)
        w13_gscale_l.append(gscale)
        w13_ref_l.append(ref)

        packed, scales, gscale, ref = make_nvfp4_weight_and_ref(
            hidden_size, intermediate_size, dtype, group_size=group_size
        )
        w2_packed_l.append(packed)
        w2_scales_l.append(scales)
        w2_gscale_l.append(gscale)
        w2_ref_l.append(ref)

    layer = torch.nn.Module()
    layer.quant_config = SimpleNamespace(group_size=group_size)
    layer.moe_runner_config = SimpleNamespace(is_gated=False)
    layer.params_dtype = dtype
    layer.intermediate_size_per_partition = intermediate_size
    layer.w13_weight = torch.nn.Parameter(
        torch.stack(w13_packed_l), requires_grad=False
    )
    layer.w2_weight = torch.nn.Parameter(torch.stack(w2_packed_l), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(
        torch.stack(w13_scales_l), requires_grad=False
    )
    layer.w2_weight_scale = torch.nn.Parameter(
        torch.stack(w2_scales_l), requires_grad=False
    )
    layer.w13_weight_scale_2 = torch.nn.Parameter(
        torch.stack(w13_gscale_l), requires_grad=False
    )
    layer.w2_weight_scale_2 = torch.nn.Parameter(
        torch.stack(w2_gscale_l), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)

    policy_layer = _make_modelopt_fused_moe(
        configured_backend=configured_backend,
        routed_scaling_factor=routed_scaling_factor,
    )
    assert policy_layer.quant_method._moe_runner_backend.is_marlin()

    # Scale activations down so relu² doesn't blow up intermediate magnitudes;
    # this keeps output values small so tighter element-wise tolerance is realistic.
    hidden_states = torch.randn((m, hidden_size), device="cuda", dtype=dtype) / 20
    router_logits = torch.randn((m, e), device="cuda", dtype=dtype)
    correction_bias = torch.randn((e,), device="cuda", dtype=torch.float32) / 10

    # This is the real grouped sigmoid routing path used by Nemotron-H. Before
    # the fix ModelOpt always asks it to multiply the weights by the routed
    # scale, even though Marlin also multiplies the reduced output by it.
    topk_output = select_experts(
        hidden_states,
        router_logits,
        TopKConfig(
            top_k=topk,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=2,
            renormalize=True,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=(
                policy_layer.should_fuse_routed_scaling_factor_in_topk
            ),
            scoring_func="sigmoid",
        ),
    )
    reference_topk_output = select_experts(
        hidden_states,
        router_logits,
        TopKConfig(
            top_k=topk,
            use_grouped_topk=True,
            topk_group=1,
            num_expert_group=2,
            renormalize=True,
            correction_bias=correction_bias,
            routed_scaling_factor=routed_scaling_factor,
            apply_routed_scaling_factor_on_output=False,
            scoring_func="sigmoid",
        ),
    )
    torch.testing.assert_close(
        topk_output.topk_ids, reference_topk_output.topk_ids, rtol=0, atol=0
    )

    output = fused_marlin_moe(
        hidden_states=hidden_states,
        w1=layer.w13_weight,
        w2=layer.w2_weight,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gating_output=router_logits,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        w1_global_scale=layer.w13_weight_scale_2,
        w2_global_scale=layer.w2_weight_scale_2,
        workspace=layer.workspace,
        num_bits=4,
        is_k_full=True,
        routed_scaling_factor=routed_scaling_factor,
        activation="relu2",
        is_gated=False,
    )

    w13_ref = torch.stack(w13_ref_l)
    w2_ref = torch.stack(w2_ref_l)
    output_ref = torch.zeros_like(hidden_states)
    for token_idx in range(m):
        for route_idx in range(topk):
            expert_id = reference_topk_output.topk_ids[token_idx, route_idx]
            intermediate = hidden_states[token_idx] @ w13_ref[expert_id].T
            intermediate = torch.square(torch.relu(intermediate))
            routed = intermediate @ w2_ref[expert_id].T
            output_ref[token_idx] += (
                routed * reference_topk_output.topk_weights[token_idx, route_idx]
            )
    output_ref *= routed_scaling_factor

    torch.cuda.synchronize()
    # NVFP4 dequantization plus two BF16 GEMMs accumulates larger absolute
    # error after the 2x routed scale. The exact assertion below separately
    # guarantees that the router did not pre-apply that scale.
    torch.testing.assert_close(output, output_ref, rtol=0.05, atol=0.75)
    torch.testing.assert_close(
        topk_output.topk_weights,
        reference_topk_output.topk_weights,
        rtol=0,
        atol=0,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
