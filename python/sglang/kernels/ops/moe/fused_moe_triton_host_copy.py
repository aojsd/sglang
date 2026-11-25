"""Paged experts, fused host copy for the Triton MoE kernel.

``fused_moe_kernel`` calls the two kernel functions below from blocks that exist only when it is
given ``b_host_ptr``, the copy of every expert in pinned host memory, in B's layout. B then
holds only some experts ("slots"), and ``HOST_IDS_START`` is their number. An expert id below
``HOST_IDS_START`` is a slot, as without the host copy. An id of ``HOST_IDS_START + e`` is
expert e being paged in: the kernel reads it from the host copy and writes it into slot
``expert_slot_ptr[e]`` of B, which no other block reads in this launch.
``missing_expert_count_ptr[0]`` is the number of such experts in the launch.

A launch with weight scales or biases is also given their copies for every expert in pinned
host memory, ``b_scale_host_ptr`` and ``bias_host_ptr``, in the layouts of the scale and bias
slot tables. The kernel pages them in with the weights.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


def validate_paged_expert_info(
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    paged_expert_info: dict,
) -> None:
    """Each host table in ``paged_expert_info`` must have the layout of the slot table it
    mirrors: ``B``, and ``B_scale`` and ``bias`` in a launch that has them."""
    for slot_table, key, name in (
        (B, "host_table", "the weights"),
        (B_scale, "host_scales_table", "the weight scales"),
        (bias, "host_bias_table", "the biases"),
    ):
        if slot_table is None:
            continue
        host_table = paged_expert_info[key]
        assert host_table is not None, (
            "Paged experts, fused host copy: the launch has "
            f"{name} but no host table of them"
        )
        assert host_table.shape[1:] == slot_table.shape[1:], (
            "Paged experts, fused host copy: the shape of one expert in the host "
            f"table of {name}, {tuple(host_table.shape[1:])}, does not match the "
            f"slot table's, {tuple(slot_table.shape[1:])}"
        )
        assert host_table.dtype == slot_table.dtype, (
            "Paged experts, fused host copy: the dtype of the host table of "
            f"{name}, {host_table.dtype}, does not match the slot table's, "
            f"{slot_table.dtype}"
        )


def paged_expert_kernel_kwargs(
    B: torch.Tensor,
    B_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    paged_expert_info: dict,
) -> dict:
    """``fused_moe_kernel``'s arguments for paging experts into the slot table ``B``, from
    ``paged_expert_info``: ``host_table`` (every expert in pinned host memory, in B's
    layout), ``host_scales_table`` and ``host_bias_table`` (the same for ``B_scale`` and
    ``bias``, in a launch that has them), ``expert_slot`` (the layer's expert -> slot table)
    and ``missing_expert_count`` (the number of experts its current step pages in)."""
    validate_paged_expert_info(
        B=B, B_scale=B_scale, bias=bias, paged_expert_info=paged_expert_info
    )
    host_scales = None if B_scale is None else paged_expert_info["host_scales_table"]
    host_bias = None if bias is None else paged_expert_info["host_bias_table"]
    return {
        "HOST_IDS_START": B.shape[0],
        "b_host_ptr": paged_expert_info["host_table"],
        "b_scale_host_ptr": host_scales,
        "bias_host_ptr": host_bias,
        "expert_slot_ptr": paged_expert_info["expert_slot"],
        "missing_expert_count_ptr": paged_expert_info["missing_expert_count"],
    }


@triton.jit
def spread_missing_expert_blocks(
    pid_m, num_tokens_post_padded, missing_expert_count_ptr, BLOCK_SIZE_M: tl.constexpr
):
    """The block of tokens this thread block takes. The experts read from host memory have the
    highest ids, so their blocks of tokens come last in the sorted order. We spread them evenly
    through the launch, so that the host reads overlap with the work on resident experts."""
    groups = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    hosts = tl.minimum(tl.load(missing_expert_count_ptr), groups)
    if pid_m < groups:
        before = (pid_m * hosts + groups - 1) // groups
        if ((pid_m + 1) * hosts + groups - 1) // groups > before:
            pid_m = groups - hosts + before
        else:
            pid_m = pid_m - before
    return pid_m


@triton.jit
def matmul_with_host_expert(
    accumulator,
    a_ptrs,
    token_mask,
    K,
    stride_ak,
    b_ptr,
    b_host_ptr,
    expert_slot_ptr,
    off_experts,
    offs_k,
    offs_bn,
    stride_be,
    stride_bk,
    stride_bn,
    a_scale_ptrs,
    b_scale_ptrs,
    b_scale_slot_ptr,
    pid_n,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    compute_type: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    swap_ab: tl.constexpr,
    HOST_IDS_START: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """The kernel's K loop for a block that reads its expert from host memory: it writes each
    tile into the expert's slot as it arrives, and returns 0 for K, so that the kernel's own
    loop has nothing left to do. It is a separate loop so that blocks for resident experts do not
    include any store logic for expert weights. The multiply is the kernel's own, for every
    format. Per-tile weight scales are read through ``b_scale_ptrs``, which the kernel has
    pointed at the host table for this block, and written into the slot as they are read."""
    tile_scales: tl.constexpr = (
        (use_fp8_w8a8 or use_int8_w8a8) and group_k > 0 and group_n > 0
    )
    if off_experts >= HOST_IDS_START:
        expert = off_experts - HOST_IDS_START
        slot = tl.load(expert_slot_ptr + expert).to(tl.int64)
        b_tile = offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        b_ptrs = b_host_ptr + expert * stride_be + b_tile
        b_copy_ptrs = b_ptr + slot * stride_be + b_tile
        if tile_scales:  # where the kernel reads the block's scales, in the slot
            if BLOCK_SIZE_N > group_n:
                offs_bsn = offs_bn // group_n
            else:
                offs_bsn = pid_n * BLOCK_SIZE_N // group_n
            b_scale_copy_ptrs = (
                b_scale_slot_ptr + slot * stride_bse + offs_bsn * stride_bsn
            )
        for k_start in range(0, K, BLOCK_SIZE_K):
            if even_Ks:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
                tl.store(b_copy_ptrs, b)
            else:
                k_mask = offs_k < K - k_start
                a = tl.load(
                    a_ptrs, mask=token_mask[:, None] & k_mask[None, :], other=0.0
                )
                b = tl.load(b_ptrs, mask=k_mask[:, None], other=0.0)
                tl.store(b_copy_ptrs, b, mask=k_mask[:, None])
            # The same multiply as the kernel's own loop.
            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                    tl.store(b_scale_copy_ptrs + offs_ks * stride_bsk, b_scale)
                    if swap_ab:
                        a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                        a_scale, b_scale = b_scale, a_scale
                    if BLOCK_SIZE_N > group_n:
                        accumulator += (
                            tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
                        )
                    elif swap_ab:
                        accumulator += tl.dot(a, b) * (a_scale * b_scale[None, :])
                    else:
                        accumulator += tl.dot(a, b) * (a_scale[:, None] * b_scale)
                else:
                    if use_fp8_w8a8:
                        if swap_ab:
                            a, b = tl.trans(b, (1, 0)), tl.trans(a, (1, 0))
                        accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        accumulator += tl.dot(a, b)
            else:
                accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
            b_copy_ptrs += BLOCK_SIZE_K * stride_bk
        K = K * 0
    return accumulator, K
