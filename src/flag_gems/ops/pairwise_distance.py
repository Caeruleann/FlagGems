import logging

import torch
import triton
import triton.language as tl
import math

from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import dim_compress, libentry, tl_extra_shim, libtuner
from flag_gems.utils import triton_lang_extension as ext

pow = tl_extra_shim.pow
logger = logging.getLogger(__name__)

PAIRWISE_DISTANCE_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
    triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
]

@libentry()
@libtuner(configs=PAIRWISE_DISTANCE_CONFIGS, key=["D"])
@triton.jit
def pairwise_distance_kernel(
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
        acc += tl.sum(tl.where(mask, pow(diff, p), 0.0))
    dist = pow(acc, 1.0 / p)
    tl.store(out_ptr + pid, dist)

@libentry()
@libtuner(configs=PAIRWISE_DISTANCE_CONFIGS, key=["D"])
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
        diff = tl.abs(a - b + eps)
        diff = diff.to(tl.float32)
        acc += tl.sum(tl.where(mask, (diff * diff), 0.0))

    dist = tl.sqrt(acc)
    tl.store(out_ptr + pid, dist)

@libentry()
@libtuner(configs=PAIRWISE_DISTANCE_CONFIGS, key=["D"])
@triton.jit
def pairwise_distance_p1_kernel(
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
        diff = tl.abs(a - b + eps)
        diff = diff.to(tl.float32)
        acc += tl.sum(tl.where(mask, diff, 0.0))

    tl.store(out_ptr + pid, acc)

@libentry()
@libtuner(configs=PAIRWISE_DISTANCE_CONFIGS, key=["D"])
@triton.jit
def pairwise_distance_p0_kernel(
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
        diff = tl.abs(a - b + eps)
        acc += tl.sum(tl.where(mask, (diff != 0).to(tl.float32), 0.0))

    tl.store(out_ptr + pid, acc)

@libentry()
@triton.jit
def pairwise_distance_p2_kernel_1(
    x1_ptr, x2_ptr, mid_ptr, D, eps, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    offset = pid_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    base = pid_n * D
    mask = offset < D
    a = tl.load(x1_ptr + base + offset, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x2_ptr + base + offset, mask=mask, other=0.0).to(tl.float32)
    diff = tl.abs(a - b + eps)
    mid = tl.sum(tl.where(mask, diff * diff, 0.0))
    tl.store(mid_ptr + pid_n * MID_SIZE + pid_d, mid)


@libentry()
@triton.jit
def pairwise_distance_p2_kernel_2(
    mid_ptr, out_ptr, MID_SIZE, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset = tl.arange(0, BLOCK_SIZE)
    mask = offset < MID_SIZE
    mid = tl.load(mid_ptr + pid * MID_SIZE + offset, mask=mask, other=0.0)
    sum = tl.sqrt(tl.sum(mid))

    tl.store(out_ptr + pid, sum)

def pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    logger.debug("GEMS PAIRWISE_DISTANCE")
    # 按需广播/连续化:同形+连续是常见情况,无条件 broadcast_tensors + contiguous 会
    # 偷偷 materialize/copy,在小张量上吃掉 ~15us。只在真正需要时才做(参照 sum 的写法)。
    if x1.shape != x2.shape:
        x1, x2 = torch.broadcast_tensors(x1, x2)
    if not x1.is_contiguous():
        x1 = x1.contiguous()
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    D = x1.shape[-1]
    N = x1.numel() // D
    out = torch.empty(x1.shape[:-1], device=x1.device, dtype=x1.dtype)
    if keepdim:
        out = out.unsqueeze(-1)
    grid = (N,)
    if p == 2.0:
        # 分派:大 N(每行 1 个 program 已贴满带宽)或 小 D(split-K 没意义、MID 太小)
        #       -> 单 kernel,省 split-K 的 mid 分配 + 二次 launch;
        #   小 N + 大 D(单 kernel 只有 N 个 program 撑不满卡) -> split-K 拆 D 补并行。
        if N >= 1024 or D < 8192:
            pairwise_distance_p2_kernel[grid](x1, x2, out, D, eps)
        else:
            BLOCK_SIZE = 1024
            MID_SIZE = triton.cdiv(D, BLOCK_SIZE)
            BLOCK_MID = triton.next_power_of_2(MID_SIZE)
            mid = torch.empty((N, MID_SIZE), device=x1.device, dtype=torch.float32)
            pairwise_distance_p2_kernel_1[(N, MID_SIZE)](x1, x2, mid, D, eps, MID_SIZE, BLOCK_SIZE)
            pairwise_distance_p2_kernel_2[(N,)](mid, out, MID_SIZE, BLOCK_MID)
    elif p == 1.0:
        pairwise_distance_p1_kernel[grid](x1, x2, out, D, eps)
    elif p == 0.0:
        pairwise_distance_p0_kernel[grid](x1, x2, out, D, eps)
    else:
        pairwise_distance_kernel[grid](x1, x2, out, D, eps, p)

    return out
