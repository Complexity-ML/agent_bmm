# Copyright 2026 INL Dynamics / Complexity-ML. All rights reserved.
# Licensed under CC-BY-NC-4.0
"""
Triton Kernel for BMM Routing — Fused expert dispatch.

Replaces the torch.bmm dispatch with a custom Triton kernel that fuses:
  1. Expert weight gathering (indexing)
  2. Up projection (matmul + SiLU)
  3. Down projection (matmul)

Into a single GPU kernel launch — zero intermediate allocations.

Requires: pip install triton>=2.1
Fallback: uses torch.bmm if triton is not available.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    pass


if _HAS_TRITON:

    @triton.jit
    def _bmm_expert_dispatch_kernel(
        # Pointers
        x_ptr,
        up_ptr,
        down_ptr,
        ids_ptr,
        out_ptr,
        # Dimensions
        N,
        H: tl.constexpr,
        E: tl.constexpr,
        # Strides
        stride_x_n,
        stride_up_k,
        stride_up_h,
        stride_down_k,
        stride_down_e,
        stride_out_n,
        # Block sizes
        BLOCK_H: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        """Fused expert dispatch: gather + up_proj + SiLU + down_proj."""
        pid = tl.program_id(0)
        if pid >= N:
            return

        # Load expert id for this query
        expert_id = tl.load(ids_ptr + pid)

        # Load input x[pid, :H]
        h_offsets = tl.arange(0, BLOCK_H)
        h_mask = h_offsets < H
        tl.load(x_ptr + pid * stride_x_n + h_offsets, mask=h_mask, other=0.0)

        # Up projection: x @ up_proj[expert_id] → (E,)
        e_offsets = tl.arange(0, BLOCK_E)
        e_mask = e_offsets < E

        up_result = tl.zeros([BLOCK_E], dtype=tl.float32)
        for h_idx in range(H):
            x_val = tl.load(x_ptr + pid * stride_x_n + h_idx)
            up_col = tl.load(
                up_ptr + expert_id * stride_up_k + h_idx * stride_up_h + e_offsets,
                mask=e_mask,
                other=0.0,
            )
            up_result += x_val * up_col

        # SiLU activation
        activated = up_result * tl.sigmoid(up_result)

        # Down projection: activated @ down_proj[expert_id] → (H,)
        out_row = tl.zeros([BLOCK_H], dtype=tl.float32)
        for e_idx in range(E):
            tl.load(
                up_ptr + expert_id * stride_up_k + 0 + e_idx  # placeholder, reuse activated
            )
            # Actually use the activated value
            down_col = tl.load(
                down_ptr + expert_id * stride_down_k + e_idx * stride_down_e + h_offsets,
                mask=h_mask,
                other=0.0,
            )
            out_row += activated[e_idx] * down_col

        # Store output
        tl.store(out_ptr + pid * stride_out_n + h_offsets, out_row, mask=h_mask)


def triton_bmm_dispatch(
    x: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """
    Fused BMM expert dispatch using Triton.

    Args:
        x: (N, H) input queries
        up_proj: (K, H, E) expert up projection weights
        down_proj: (K, E, H) expert down projection weights
        expert_ids: (N,) selected expert per query

    Returns:
        (N, H) expert outputs
    """
    if not _HAS_TRITON:
        return _torch_fallback(x, up_proj, down_proj, expert_ids)

    N, H = x.shape
    E = up_proj.shape[2]

    # Ensure contiguous
    x = x.contiguous()
    up_proj = up_proj.contiguous()
    down_proj = down_proj.contiguous()
    expert_ids = expert_ids.contiguous().to(torch.int32)

    output = torch.empty_like(x)

    BLOCK_H = triton.next_power_of_2(H)
    BLOCK_E = triton.next_power_of_2(E)

    grid = (N,)
    _bmm_expert_dispatch_kernel[grid](
        x,
        up_proj,
        down_proj,
        expert_ids,
        output,
        N,
        H,
        E,
        x.stride(0),
        up_proj.stride(0),
        up_proj.stride(1),
        down_proj.stride(0),
        down_proj.stride(1),
        output.stride(0),
        BLOCK_H=BLOCK_H,
        BLOCK_E=BLOCK_E,
    )
    return output


def _torch_fallback(
    x: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    """Standard torch.bmm fallback when Triton is not available."""
    import torch.nn.functional as F

    sel_up = up_proj[expert_ids]
    sel_down = down_proj[expert_ids]
    up = torch.bmm(x.unsqueeze(1), sel_up).squeeze(1)
    activated = F.silu(up)
    return torch.bmm(activated.unsqueeze(1), sel_down).squeeze(1)


def is_triton_available() -> bool:
    """Check if Triton is available."""
    return _HAS_TRITON
