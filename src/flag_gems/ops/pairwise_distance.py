import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.shape_utils import volume

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def pairwise_distance_p2_kernel(
    x1_ptr, x2_ptr, out_ptr, D, eps, BLOCK_SIZE: tl.constexpr
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
        diff = a - b + eps
        diff = diff.to(tl.float32)
        acc += tl.sum(diff * diff)
    dist = tl.sqrt(acc)
    tl.store(out_ptr + pid, dist)


def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS PAIRWISE_DISTANCE")
    N, D = x1.shape if x1.ndim == 2 else (1, x1.shape[-1])
    out = torch.empty((N,), device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)
    if x1.ndim == 1:
        out = out.squeeze(0)
    grid = (N,)

    if p == 2.0:
        pairwise_distance_p2_kernel[grid](x1, x2, out, D, eps, BLOCK_SIZE=256)

    return out
