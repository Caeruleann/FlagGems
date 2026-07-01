import logging

import torch
import triton

from flag_gems.ops.pairwise_distance import (
    pairwise_distance_general_kernel,
    pairwise_distance_general_kernel_1,
    pairwise_distance_general_kernel_2,
    pairwise_distance_max_kernel,
    pairwise_distance_max_kernel_1,
    pairwise_distance_max_kernel_2,
    pairwise_distance_min_kernel,
    pairwise_distance_min_kernel_1,
    pairwise_distance_min_kernel_2,
    pairwise_distance_p0_kernel,
    pairwise_distance_p0_kernel_1,
    pairwise_distance_p0_kernel_2,
    pairwise_distance_p1_kernel,
    pairwise_distance_p1_kernel_1,
    pairwise_distance_p1_kernel_2,
    pairwise_distance_p2_kernel,
    pairwise_distance_p2_kernel_1,
    pairwise_distance_p2_kernel_2,
)

logger = logging.getLogger(
    f"flag_gems.runtime._ascend.ops.{__name__.split('.')[-1]}"
)

# Split-K only for very large D (see module docstring). Below this the single
# kernel (one program per BLOCK_M rows) is faster -- split-K's 2nd launch + mid
# buffer GM round-trip isn't worth it.
PAIRWISE_SPLIT_D_THRESHOLD = 524288

# Split-K chunk size (BLOCK_SIZE of kernel_1). Picked by a do_bench sweep over
# {1024,2048,4096,8192} on D in {1M, 4M, 10M}; 4096 won (1.5-2.4x over 1024).
# min(4096, next_pow2(D)) gives the largest-fit chunk without masked waste.
PAIRWISE_SPLIT_MAX_BLOCK = 4096


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS_ASCEND PAIRWISE_DISTANCE")
    # Only broadcast/contiguous when needed (same shape + contiguous is the common
    # case; unconditional broadcast_tensors + contiguous materializes a copy).
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]

    # Empty feature dim: torch returns 0 for finite p; inf/-inf have no identity
    # element over an empty reduction. Short-circuit (also avoids ZeroDivisionError
    # in the split-K plumbing when BLOCK_SIZE == 0).
    if D == 0:
        if p == float("inf") or p == float("-inf"):
            raise RuntimeError(
                "pairwise_distance cannot compute the inf/-inf norm on an empty "
                "reduction dimension (no identity element)"
            )
        out = torch.zeros(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
        if keepdim:
            out = out.unsqueeze(-1)
        return out

    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)

    use_split = D >= PAIRWISE_SPLIT_D_THRESHOLD
    if use_split:
        BLOCK_SIZE = min(PAIRWISE_SPLIT_MAX_BLOCK, triton.next_power_of_2(D))
        MID_SIZE = triton.cdiv(D, BLOCK_SIZE)
        BLOCK_MID = triton.next_power_of_2(MID_SIZE)
        mid = torch.empty((N, MID_SIZE), device=x1.device, dtype=torch.float32)
        split_grid = (N, MID_SIZE)
        final_grid = (N,)
    else:
        # single kernel: libtuner picks BLOCK_M/BLOCK_D per D from tune_configs.yaml
        single_grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]),)

    if p == 2.0:
        if not use_split:
            pairwise_distance_p2_kernel[single_grid](x1, x2, out, N, D, eps)
        else:
            pairwise_distance_p2_kernel_1[split_grid](
                x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_p2_kernel_2[final_grid](mid, out, MID_SIZE, BLOCK_MID)
    elif p == 1.0:
        if not use_split:
            pairwise_distance_p1_kernel[single_grid](x1, x2, out, N, D, eps)
        else:
            pairwise_distance_p1_kernel_1[split_grid](
                x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_p1_kernel_2[final_grid](mid, out, MID_SIZE, BLOCK_MID)
    elif p == 0.0:
        if not use_split:
            pairwise_distance_p0_kernel[single_grid](x1, x2, out, N, D, eps)
        else:
            pairwise_distance_p0_kernel_1[split_grid](
                x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_p0_kernel_2[final_grid](mid, out, MID_SIZE, BLOCK_MID)
    elif p == float("inf"):
        if not use_split:
            pairwise_distance_max_kernel[single_grid](x1, x2, out, N, D, eps)
        else:
            pairwise_distance_max_kernel_1[split_grid](
                x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_max_kernel_2[final_grid](mid, out, MID_SIZE, BLOCK_MID)
    elif p == float("-inf"):
        if not use_split:
            pairwise_distance_min_kernel[single_grid](x1, x2, out, N, D, eps)
        else:
            pairwise_distance_min_kernel_1[split_grid](
                x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_min_kernel_2[final_grid](mid, out, MID_SIZE, BLOCK_MID)
    else:
        if not use_split:
            pairwise_distance_general_kernel[single_grid](x1, x2, out, N, D, eps, p)
        else:
            pairwise_distance_general_kernel_1[split_grid](
                x1, x2, mid, D, eps, p, MID_SIZE, BLOCK_SIZE
            )
            pairwise_distance_general_kernel_2[final_grid](
                mid, out, p, MID_SIZE, BLOCK_MID
            )

    return out
