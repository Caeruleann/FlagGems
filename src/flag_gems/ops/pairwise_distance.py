import logging

import torch
import triton
import triton.language as tl

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim, libtuner
from flag_gems.utils import triton_lang_extension as ext

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

@libentry()
@libtuner(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    ],
    key=["D"]
)
@triton.jit
def pairwise_distance_p2_kernel(
    x1_ptr, x2_ptr, out_ptr, D, eps, p, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    x1_ptr = x1_ptr + pid * D
    x2_ptr = x2_ptr + pid * D
    acc = tl.zeros([], dtype=tl.float32)
    for start in range(0, D, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        a = tl.load(x1_ptr + cols, mask=mask, other=0)
        b = tl.load(x2_ptr + cols, mask=mask, other=0)
        diff = tl.abs(a - b + eps)
        diff = diff.to(tl.float32)
        if p == 0.0:
            acc += tl.sum(tl.where(mask, (diff != 0.0).to(tl.float32), 0.0))
        else:
            acc += tl.sum(tl.where(mask, pow(diff, p), 0.0))
    if p == 0.0:
        dist = acc
    else:
        dist = pow(acc, 1.0 / p)
    tl.store(out_ptr + pid, dist)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS PAIRWISE_DISTANCE")
    x1, x2 = torch.broadcast_tensors(x1, x2)
    x1, x2 = x1.contiguous(), x2.contiguous()
    D = x1.shape[-1]
    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)
    grid = (N,)
    pairwise_distance_p2_kernel[grid](x1, x2, out, D, eps, p)

    return out
