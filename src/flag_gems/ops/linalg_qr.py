# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-Triton implementation of ``torch.linalg.qr`` (``aten::linalg_qr``).

Blocked Householder QR with the compact WY (Gram-solve) block representation,
following the algorithm family described in

    Michael Lutz, "QR Decomp at the Speed of Light", https://ml-mike.com/writing/qr_v2

Routing summary (see :func:`linalg_qr` for the exact conditions):

* **Fused path** (small square + short tall-skinny): a single ``_qr_fused_kernel``
  does unblocked QR + R extraction + Q assembly in one launch.
* **TSQR path** (tall-skinny, m > 512): ``_tsqr_local_kernel`` factors all row
  blocks concurrently; a flat reduction combines the R factors; Q = A R^{-1}
  via ``_trsm_kernel``.
* **Blocked path** (large square): ``_geqrt_sram_kernel`` factors IB-wide panels
  that fit in shared memory (one CTA, no global re-reads); taller panels fall
  back to the multi-CTA ``_geqrt_mcta_kernel`` (row-split across NC CTAs).
  ``_larft_kernel`` builds the WY factor T (via the Gram-solve trick and a
  small in-kernel triangular inverse); ``_larfb_kernel`` applies the block
  reflector for the trailing update and Q assembly with plain GEMMs.

Every numerical step lives in a Triton kernel -- the python wrapper only
allocates buffers and launches kernels.

Supports the three modes ``"reduced"`` / ``"complete"`` / ``"r"``.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
_PANEL_IB = 32
_PANEL_RM = 64  # row tile inside the panel kernel
_MCTA_NC_MAX = 64  # max CTAs cooperating on one panel (multi-CTA path)
_MCTA_MIN_NC = 4  # multi-CTA from NC>=4 (M>=256); below it the barrier overhead per reflector dominates
# SRAM-resident panel factorisation is used while the panel tile fits shared
# memory: BM=next_pow2(M) and BM*_PANEL_IB*itemsize must stay within the SRAM
# budget.  _PANEL_IB=32, fp32 -> BM<=1024 (128 KB) is the safe cap.
_GEQRT_SRAM_MAX_M = 512
_LARFB_RM = 64  # row tile for the block-reflector apply kernel
_LARFB_TN = 32  # column tile for the block-reflector apply kernel (tuned: TN=32
#                gives ~1.9x on large trailing updates vs TN=16; TN=64 spills)
_TSQR_ASPECT = 4  # m >= _TSQR_ASPECT * n  =>  tall-skinny candidate for TSQR
# TSQR's flat reduction only beats the blocked path (geqrt_sram, zero-sync panels)
# once m is large enough; below this the per-block local-QR sync overhead dominates
# and the blocked path is faster (empirical crossover ~640-700 on H20).
_TSQR_MIN_M = 700
_TSQR_BLOCK = 1024  # row block used by the flat TSQR reduction
# Max elements per TSQR row block for the register-resident single-CTA local
# QR (no atomics / cross-CTA barriers / global re-reads).  Above this the
# multi-CTA _tsqr_local_kernel is kept.
_TSQR_SRAM_ELEM = 16384
# Single-launch fused SRAM kernel: used when the matrix tile (and Q tile) fit in
# shared memory.  Covers small square matrices AND tall-skinny ones (large m,
# small n).  Caps keep the BM/BN/BQ tiles inside SRAM.
_FUSED_DIM = 128   # max columns n (BN tile)
_FUSED_M = 4096    # max rows m (BM tile) -- relevant for tall-skinny
_FUSED_TALL_M = 512  # for tall-skinny, fused only up to this m (single CTA serializes beyond)
_FUSED_ELEM = 8192  # max elements in the A tile (m*n) and Q tile (qcols*m)


# ===========================================================================
# Kernel 1: blocked-Householder panel factorization (unblocked QR of IB cols)
# ===========================================================================
@libentry()
@triton.jit
def _geqrt_kernel(
    W,
    V,
    TAU,
    M,
    kk,
    ib,
    n,
    k,
    nr,
    sWb,
    sWm,
    sWn,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
    RM: tl.constexpr,
    IBN: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB

    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rows_local = tl.arange(0, RM)
    col_idx = tl.arange(0, IBN)
    num_tiles = (M + RM - 1) // RM

    for j in range(nr):
        pivot = kk + j

        # ---- pass A: pivot alpha + tail norm --------------------------------
        alpha = zero
        xnorm_sq = zero
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            col = tl.load(Wb + rows_g * sWm + (kk + j) * sWn, mask=rmask, other=zero)
            alpha += tl.sum(tl.where(rmask & (local == j), col, zero))
            xnorm_sq += tl.sum(tl.where(rmask & (local > j), col * col, zero))

        # ---- dlarfg ---------------------------------------------------------
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta

        tl.store(TAUb + (kk + j) * sTauN, tau)
        tl.store(Wb + pivot * sWm + (kk + j) * sWn, beta_eff)

        # ---- pass B: store Householder vector (explicit unit lower) ---------
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            col = tl.load(Wb + rows_g * sWm + (kk + j) * sWn, mask=rmask, other=zero)
            v_tail = col / denom
            v = tl.where(local > j, v_tail, tl.where(local == j, one, zero))
            v = tl.where(reflect, v, tl.where(local == j, one, zero))
            tl.store(Vb + rows_g * sVm + (kk + j) * sVn, v, mask=rmask)
        tl.debug_barrier()

        # ---- pass C: w = tau * (v^H W[:, trailing]) -------------------------
        w = tl.zeros([IBN], dtype=dt)
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            v = tl.load(Vb + rows_g * sVm + (kk + j) * sVn, mask=rmask, other=zero)
            wblock_off = rows_g[:, None] * sWm + (kk + col_idx)[None, :] * sWn
            cmask = rmask[:, None] & (col_idx[None, :] < ib)
            Wt = tl.load(Wb + wblock_off, mask=cmask, other=zero)
            w += tl.sum(v[:, None] * Wt, axis=0)
        w = tau * w
        w = tl.where((col_idx > j) & (col_idx < ib), w, zero)

        # ---- pass D: apply W[:, trailing] -= v * w --------------------------
        for t in range(num_tiles):
            local = t * RM + rows_local
            rows_g = kk + local
            rmask = local < M
            v = tl.load(Vb + rows_g * sVm + (kk + j) * sVn, mask=rmask, other=zero)
            wblock_off = rows_g[:, None] * sWm + (kk + col_idx)[None, :] * sWn
            cmask = rmask[:, None] & (col_idx[None, :] < ib)
            Wt = tl.load(Wb + wblock_off, mask=cmask, other=zero)
            upd = v[:, None] * w[None, :]
            upd = tl.where((col_idx[None, :] > j) & (col_idx[None, :] < ib), upd, zero)
            Wt = Wt - upd
            smask = rmask[:, None] & (col_idx[None, :] > j) & (col_idx[None, :] < ib)
            tl.store(Wb + wblock_off, Wt, mask=smask)
        tl.debug_barrier()


# ===========================================================================
# Kernel 1a: multi-CTA panel factorisation (row-split across NC CTAs).
# For tall panels the single-CTA geqrt leaves most SMs idle because the
# reflector chain is serial.  Here the panel rows are split across NC CTAs;
# each CTA owns a contiguous row range, so the only cross-CTA coordination is
# the two per-column reductions (alpha/norm after pass A, w after pass C),
# done with atomic_add + a spinning counter barrier.  Everything else (passes
# B and D, reading V/W) touches only a CTA's own rows.
# ===========================================================================
@triton.jit
def _barrier(ctr, off, NC):
    tl.atomic_add(ctr + off, 1)
    while tl.load(ctr + off, volatile=True) < NC:
        pass


@libentry()
@triton.jit
def _geqrt_mcta_kernel(
    W, V, TAU, alpha_buf, xnorm_buf, w_sum, ctr,
    M, kk, ib, n, k, nr, p,
    sWb, sWm, sWn, sVb, sVm, sVn, sTauB, sTauN,
    sAB, sAM, sXB, sXM, sWB, sWM, sWN, sCtrB, sCtrM,
    CHUNK: tl.constexpr, RM: tl.constexpr, IBN: tl.constexpr,
    NC: tl.constexpr, NSYNC: tl.constexpr, NUM_TILES: tl.constexpr,
):
    pid_b = tle.program_id(0)
    c = tle.program_id(1)
    Wb = W + pid_b * sWb
    Vb = V + pid_b * sVb
    TAUb = TAU + pid_b * sTauB
    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows_local = tl.arange(0, RM)
    col_idx = tl.arange(0, IBN)
    row_lo = c * CHUNK

    if NUM_TILES == 1:
        # Fast path: each CTA owns exactly one row-tile (CHUNK==RM).  Load the
        # panel chunk ONCE into registers and sustain it across the whole
        # reflector loop -- col_j is extracted from it, the trailing update is
        # applied in place, and only V/tau are written per reflector.  This is
        # the multi-CTA analogue of _geqrt_sram_kernel: row parallelism (NC CTAs)
        # with zero per-reflector global reload of the panel.  Registers survive
        # the cross-CTA barriers, so the sustained chunk stays live.
        lr = rows_local
        plr = row_lo + lr
        gr = kk + plr
        rmask = (plr < M) & (plr < row_lo + CHUNK)
        cmask = col_idx < ib
        Wblk = tl.load(Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                       mask=rmask[:, None] & cmask[None, :], other=zero)
        for j in range(nr):
            # column j from the sustained panel chunk
            col_j = tl.sum(tl.where(col_idx[None, :] == j, Wblk, zero), axis=1)
            alpha_c = tl.sum(tl.where(rmask & (plr == j), col_j, zero))
            xnorm_c = tl.sum(tl.where(rmask & (plr > j), col_j * col_j, zero))
            tl.atomic_add(alpha_buf + pid_b * sAB + p * sAM + j, alpha_c)
            tl.atomic_add(xnorm_buf + pid_b * sXB + p * sXM + j, xnorm_c)
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j * NSYNC + 0, NC)
            alpha = tl.load(alpha_buf + pid_b * sAB + p * sAM + j)
            xnorm_sq = tl.load(xnorm_buf + pid_b * sXB + p * sXM + j)

            norm = tl.sqrt(alpha * alpha + xnorm_sq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xnorm_sq > zero
            beta_eff = tl.where(reflect, beta, alpha)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            tl.store(TAUb + (kk + j) * sTauN, tau)
            # V from col_j -> global V buffer
            v_tail = col_j / denom
            v = tl.where(plr > j, v_tail, tl.where(plr == j, one, zero))
            v = tl.where(reflect, v, tl.where(plr == j, one, zero))
            tl.store(Vb + gr * sVm + (kk + j) * sVn, v, mask=rmask)
            # write R diagonal into the sustained chunk (owning CTA only)
            diag = (lr[:, None] == (j - row_lo)) & (col_idx[None, :] == j)
            Wblk = tl.where(diag, beta_eff, Wblk)
            # w = tau * v^H (trailing panel cols) from the sustained chunk
            wmask = (col_idx[None, :] > j) & cmask[None, :]
            w_c = tau * tl.sum(tl.where(wmask, v[:, None] * Wblk, zero), axis=0)
            tl.atomic_add(w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx, w_c, mask=col_idx < ib)
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j * NSYNC + 1, NC)
            w = tl.load(w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                        mask=col_idx < ib, other=zero)
            # in-place trailing update of the sustained chunk
            upd = v[:, None] * w[None, :]
            upd = tl.where(wmask, upd, zero)
            Wblk = Wblk - upd
        # flush the sustained panel chunk once (upper triangle now holds R)
        tl.store(Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn, Wblk,
                 mask=rmask[:, None] & cmask[None, :])
    else:
        # General path (CHUNK > RM, very tall panels): tile loop per pass.
        for j in range(nr):
            alpha_c = zero
            xnorm_c = zero
            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                col = tl.load(Wb + gr * sWm + (kk + j) * sWn, mask=rmask, other=zero)
                alpha_c += tl.sum(tl.where(rmask & (plr == j), col, zero))
                xnorm_c += tl.sum(tl.where(rmask & (plr > j), col * col, zero))
            tl.atomic_add(alpha_buf + pid_b * sAB + p * sAM + j, alpha_c)
            tl.atomic_add(xnorm_buf + pid_b * sXB + p * sXM + j, xnorm_c)
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j * NSYNC + 0, NC)
            alpha = tl.load(alpha_buf + pid_b * sAB + p * sAM + j)
            xnorm_sq = tl.load(xnorm_buf + pid_b * sXB + p * sXM + j)

            norm = tl.sqrt(alpha * alpha + xnorm_sq)
            beta = tl.where(alpha >= zero, -norm, norm)
            reflect = xnorm_sq > zero
            beta_eff = tl.where(reflect, beta, alpha)
            tau = tl.where(reflect, (beta - alpha) / beta, zero)
            denom = alpha - beta
            tl.store(TAUb + (kk + j) * sTauN, tau)
            tl.store(Wb + (kk + j) * sWm + (kk + j) * sWn, beta_eff)

            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                col = tl.load(Wb + gr * sWm + (kk + j) * sWn, mask=rmask, other=zero)
                v_tail = col / denom
                v = tl.where(plr > j, v_tail, tl.where(plr == j, one, zero))
                v = tl.where(reflect, v, tl.where(plr == j, one, zero))
                tl.store(Vb + gr * sVm + (kk + j) * sVn, v, mask=rmask)
            tl.debug_barrier()

            w_c = tl.zeros([IBN], dtype=dt)
            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                v = tl.load(Vb + gr * sVm + (kk + j) * sVn, mask=rmask, other=zero)
                wblk = tl.load(Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                               mask=rmask[:, None] & (col_idx[None, :] < ib), other=zero)
                w_c += tl.sum(v[:, None] * wblk, axis=0)
            w_c = tau * w_c
            w_c = tl.where((col_idx > j) & (col_idx < ib), w_c, zero)
            tl.atomic_add(w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx, w_c, mask=col_idx < ib)
            _barrier(ctr, pid_b * sCtrB + p * sCtrM + j * NSYNC + 1, NC)
            w = tl.load(w_sum + pid_b * sWB + p * sWM + j * sWN + col_idx,
                        mask=col_idx < ib, other=zero)

            for t in range(NUM_TILES):
                lr = t * RM + rows_local
                plr = row_lo + lr
                gr = kk + plr
                rmask = (plr < M) & (plr < row_lo + CHUNK)
                v = tl.load(Vb + gr * sVm + (kk + j) * sVn, mask=rmask, other=zero)
                wblk = tl.load(Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn,
                               mask=rmask[:, None] & (col_idx[None, :] < ib), other=zero)
                upd = v[:, None] * w[None, :]
                upd = tl.where((col_idx[None, :] > j) & (col_idx[None, :] < ib), upd, zero)
                wblk = wblk - upd
                smask = rmask[:, None] & (col_idx[None, :] > j) & (col_idx[None, :] < ib)
                tl.store(Wb + gr[:, None] * sWm + (kk + col_idx)[None, :] * sWn, wblk, mask=smask)
            tl.debug_barrier()


# ===========================================================================
# ===========================================================================
# Kernel 1c: fused TSQR local QR — all blocks in one launch.
# Grid = (B, num_blocks * NC).  Each (block, cta) pair factors one row-block's
# n columns concurrently.  Replaces the serial python loop of per-block
# _blocked_qr calls with a single kernel launch so all blocks' CTAs run at once.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_local_kernel(
    W, R_out, V_local, TAU_local,
    alpha_buf, xnorm_buf, w_sum, ctr,
    m, n, br, k_max, num_blocks,
    sWb, sWm, sWn,
    sRb, sRm, sRn,
    sVb, sVm, sVn,
    sTauB,
    sAB, sAM, sXB, sXM, sWB, sWM, sWN, sCtrB, sCtrM,
    CHUNK: tl.constexpr, RM: tl.constexpr, IBN: tl.constexpr,
    NC: tl.constexpr, NSYNC: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_raw = tle.program_id(1)
    block_id = pid_raw // NC
    c = pid_raw - block_id * NC

    Wb = W + pid_b * sWb
    Vb = V_local + pid_b * sVb
    TAUb = TAU_local + pid_b * sTauB + block_id * n
    ROb = R_out + pid_b * sRb + block_id * sRm

    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    rows_local = tl.arange(0, RM)
    col_idx = tl.arange(0, IBN)

    blk_start = block_id * br
    M = m - blk_start
    M = tl.minimum(M, br)
    nr = tl.minimum(k_max, M)
    row_lo = blk_start + c * CHUNK
    num_tiles = (CHUNK + RM - 1) // RM

    for j in range(nr):
        alpha_c = zero
        xnorm_c = zero
        for t in range(num_tiles):
            lr = t * RM + rows_local
            plr = c * CHUNK + lr
            gr = blk_start + plr
            rmask = (plr < M) & (plr < (c + 1) * CHUNK)
            col = tl.load(Wb + gr * sWm + j * sWn, mask=rmask, other=zero)
            alpha_c += tl.sum(tl.where(rmask & (plr == j), col, zero))
            xnorm_c += tl.sum(tl.where(rmask & (plr > j), col * col, zero))
        tl.atomic_add(alpha_buf + pid_b * sAB + block_id * sAM + j, alpha_c)
        tl.atomic_add(xnorm_buf + pid_b * sXB + block_id * sXM + j, xnorm_c)
        _barrier(ctr, pid_b * sCtrB + block_id * sCtrM + j * NSYNC + 0, NC)
        alpha = tl.load(alpha_buf + pid_b * sAB + block_id * sAM + j)
        xnorm_sq = tl.load(xnorm_buf + pid_b * sXB + block_id * sXM + j)

        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        tl.store(TAUb + j, tau)
        tl.store(Wb + (blk_start + j) * sWm + j * sWn, beta_eff)

        for t in range(num_tiles):
            lr = t * RM + rows_local
            plr = c * CHUNK + lr
            gr = blk_start + plr
            rmask = (plr < M) & (plr < (c + 1) * CHUNK)
            col = tl.load(Wb + gr * sWm + j * sWn, mask=rmask, other=zero)
            v_tail = col / denom
            v = tl.where(plr > j, v_tail, tl.where(plr == j, one, zero))
            v = tl.where(reflect, v, tl.where(plr == j, one, zero))
            tl.store(Vb + gr * sVm + j * sVn, v, mask=rmask)
        tl.debug_barrier()

        w_c = tl.zeros([IBN], dtype=dt)
        for t in range(num_tiles):
            lr = t * RM + rows_local
            plr = c * CHUNK + lr
            gr = blk_start + plr
            rmask = (plr < M) & (plr < (c + 1) * CHUNK)
            v = tl.load(Vb + gr * sVm + j * sVn, mask=rmask, other=zero)
            wblk = tl.load(Wb + gr[:, None] * sWm + col_idx[None, :] * sWn,
                           mask=rmask[:, None] & (col_idx[None, :] < n), other=zero)
            w_c += tl.sum(v[:, None] * wblk, axis=0)
        w_c = tau * w_c
        w_c = tl.where((col_idx > j) & (col_idx < n), w_c, zero)
        tl.atomic_add(w_sum + pid_b * sWB + block_id * sWM + j * sWN + col_idx, w_c, mask=col_idx < n)
        _barrier(ctr, pid_b * sCtrB + block_id * sCtrM + j * NSYNC + 1, NC)
        w = tl.load(w_sum + pid_b * sWB + block_id * sWM + j * sWN + col_idx,
                    mask=col_idx < n, other=zero)

        for t in range(num_tiles):
            lr = t * RM + rows_local
            plr = c * CHUNK + lr
            gr = blk_start + plr
            rmask = (plr < M) & (plr < (c + 1) * CHUNK)
            v = tl.load(Vb + gr * sVm + j * sVn, mask=rmask, other=zero)
            wblk = tl.load(Wb + gr[:, None] * sWm + col_idx[None, :] * sWn,
                           mask=rmask[:, None] & (col_idx[None, :] < n), other=zero)
            upd = v[:, None] * w[None, :]
            upd = tl.where((col_idx[None, :] > j) & (col_idx[None, :] < n), upd, zero)
            wblk = wblk - upd
            smask = rmask[:, None] & (col_idx[None, :] > j) & (col_idx[None, :] < n)
            tl.store(Wb + gr[:, None] * sWm + col_idx[None, :] * sWn, wblk, mask=smask)
        tl.debug_barrier()

    # extract R = triu(W[block rows, :n]) and write to R_out[block, :, :]
    ri_idx = tl.arange(0, IBN)
    ci_idx = tl.arange(0, IBN)
    r_tile = tl.load(Wb + (blk_start + ri_idx)[:, None] * sWm + ci_idx[None, :] * sWn,
                     mask=(ri_idx[:, None] < n) & (ci_idx[None, :] < n), other=zero)
    r_tile = tl.where(ri_idx[:, None] <= ci_idx[None, :], r_tile, zero)
    tl.store(ROb + ri_idx[:, None] * sRn + ci_idx[None, :], r_tile,
             mask=(ri_idx[:, None] < n) & (ci_idx[None, :] < n))


# ===========================================================================
# Kernel 1e: register-resident TSQR local QR for narrow blocks.  One CTA per
# row block; the whole block tile (BM x IBN) lives in registers across the
# reflector chain -- no atomics, no cross-CTA barriers, no global re-reads.
# Used when a block fits a single-CTA register tile (br * n small); otherwise
# the multi-CTA _tsqr_local_kernel above is kept.
# Output contract matches _tsqr_local_kernel: R_blocks[block] = triu(local R),
# V_local holds the unit-diagonal reflectors, TAU_local the taus.
# ===========================================================================
@libentry()
@triton.jit
def _tsqr_local_sram_kernel(
    W, R_out, V_local, TAU_local,
    m, n, br, num_blocks, k_max,
    sWb, sWm, sWn, sRb, sRm, sRn, sVb, sVm, sVn, sTauB,
    BM: tl.constexpr, IBN: tl.constexpr,
):
    pid_b = tle.program_id(0)
    block_id = tle.program_id(1)
    Wb = W + pid_b * sWb
    Vb = V_local + pid_b * sVb
    TAUb = TAU_local + pid_b * sTauB + block_id * n
    ROb = R_out + pid_b * sRb + block_id * sRm

    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)   # block-local row index
    cn = tl.arange(0, IBN)  # col index 0..n-1
    blk_start = block_id * br
    M = tl.minimum(br, m - blk_start)
    nr = tl.minimum(k_max, M)
    rmask = rm < M
    cmask = cn < n
    rows_g = blk_start + rm

    # load the whole row block into one register tile
    A = tl.load(Wb + rows_g[:, None] * sWm + cn[None, :] * sWn,
                mask=rmask[:, None] & cmask[None, :], other=zero)

    for j in range(nr):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = col_j / denom
        # R diagonal + reflector tail into the in-register block
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)
        tl.store(TAUb + j, tau)
        # Householder vector vj (0/<j, 1/==j, tail/>j) to the V buffer
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))
        tl.store(Vb + rows_g * sVm + j * sVn, vj, mask=rmask)
        # trailing update within the block (cols j+1..n), in registers
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * A, zero), axis=0)
        A = tl.where(pmask, A - vj[:, None] * w[None, :], A)

    # extract R = triu(A) into R_blocks[block, :n, :n]
    r_tile = tl.where(rm[:, None] <= cn[None, :], A, zero)
    tl.store(ROb + rm[:, None] * sRn + cn[None, :], r_tile,
             mask=(rm[:, None] < n) & (cn[None, :] < n))


# ===========================================================================
# Kernel 1b: fused QR for matrices that fit in shared memory.
# One CTA per matrix does the *entire* job -- unblocked Householder
# factorisation, R extraction and (optionally) Q assembly -- operating on
# in-SRAM tiles with no global round-trips and no per-panel launches.  This is
# what makes single small/medium matrices competitive (one launch instead of
# ~20).
# ===========================================================================
# Kernel 1b: fused QR for matrices that fit in shared memory.
# One CTA per matrix does the *entire* job -- unblocked Householder
# factorisation, R extraction and (optionally) Q assembly -- operating on
# in-SRAM tiles with no global round-trips and no per-panel launches.  This is
# what makes single small/medium matrices competitive (one launch instead of
# ~20).
# ===========================================================================
@libentry()
@triton.jit
def _qr_fused_kernel(
    W, Qout, Rout, TAU,
    m, n, k, qcols, rrows, put_Q,
    sWb, sWm, sWn, sQb, sQm, sQn, sRb, sRm, sRn, sTauB, sTauN,
    BM: tl.constexpr, BN: tl.constexpr, BQ: tl.constexpr, BK: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    dt = Wb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)
    cn = tl.arange(0, BN)
    cq = tl.arange(0, BQ)
    ik = tl.arange(0, BK)
    rmask = rm < m
    cmask = cn < n

    # load the whole matrix into a register/SRAM tile
    A = tl.load(Wb + rm[:, None] * sWm + cn[None, :] * sWn,
                mask=rmask[:, None] & cmask[None, :], other=zero)

    tau_arr = tl.zeros([BK], dtype=dt)

    # ---- unblocked Householder QR, in place on the tile ----
    for j in range(k):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)  # A[:, j]
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = col_j / denom  # reflector tail (rows > j)
        tau_arr = tl.where(ik == j, tau, tau_arr)

        # write R diagonal A[j,j] and reflector tail A[r>j, j]
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)

        # reflector vector vj: 0 (r<j), 1 (r==j), v_tail (r>j)
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))

        # trailing update A[:, c>j] -= vj * (tau * vj^T A[:, c>j])
        w = tau * tl.sum(vj[:, None] * A, axis=0)  # (BN,)
        w = tl.where((cn > j) & cmask, w, zero)
        A = A - vj[:, None] * w[None, :]

    # ---- R = triu(A), written to the first `rrows` rows ----
    R_tile = tl.where(rm[:, None] <= cn[None, :], A, zero)
    rrmask = rm < rrows
    tl.store(Rout + pid * sRb + rm[:, None] * sRm + cn[None, :] * sRn,
             R_tile, mask=rrmask[:, None] & cmask[None, :])
    # tau output
    tl.store(TAU + pid * sTauB + ik * sTauN, tau_arr, mask=ik < k)

    if put_Q:
        # Q (m x qcols) = identity, then apply reflectors in reverse
        Q = tl.where(rm[:, None] == cq[None, :], one, zero)
        Q = tl.where(rmask[:, None] & (cq[None, :] < qcols), Q, zero)
        for jj in range(k):
            j = k - 1 - jj
            tauj = tl.sum(tl.where(ik == j, tau_arr, zero))
            col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
            vj = tl.where(rm > j, col_j, tl.where(rm == j, one, zero))
            w = tauj * tl.sum(vj[:, None] * Q, axis=0)  # (BQ,)
            Q = Q - vj[:, None] * w[None, :]
        tl.store(Qout + pid * sQb + rm[:, None] * sQm + cq[None, :] * sQn,
                 Q, mask=rmask[:, None] & (cq[None, :] < qcols))


# ===========================================================================
# Kernel 1d: SRAM-resident panel factorisation (blocked-path geqrt replacement).
# The multi-CTA _geqrt_mcta_kernel re-reads the panel from global memory on each
# of its 4 passes per reflector and, for moderately-tall panels (e.g. 256x32),
# only spawns NC = M/RM ~= 4 CTAs -- severe SM under-utilisation plus redundant
# global traffic.  This kernel keeps the whole panel in one SRAM tile (one CTA
# per batch) and runs the unblocked reflector chain with no global re-reads, the
# same trick that makes _qr_fused_kernel fast.  Measured ~3x faster than mcta on
# a 256x32 panel.  Used only while the panel tile fits shared memory
# (BM*IB*itemsize <= ~SRAM budget); taller panels fall back to multi-CTA.
# Output contract matches _geqrt_kernel: V (unit diag + tail) to the V buffer,
# tau to the tau buffer, R left in W's panel upper triangle.
# ===========================================================================
@libentry()
@triton.jit
def _geqrt_sram_kernel(
    W, V, TAU, M, ib, kk, n, k,
    sWb, sWm, sWn, sVb, sVm, sVn, sTauB, sTauN,
    BM: tl.constexpr, IBN: tl.constexpr,
):
    pid = tle.program_id(0)
    Wb = W + pid * sWb
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB
    dt = Wb.dtype.element_ty
    zero = tl.zeros((), dtype=dt)
    one = tl.full((), 1.0, dtype=dt)

    rm = tl.arange(0, BM)          # panel-local row  (0..BM-1 -> rows kk..kk+M-1)
    cn = tl.arange(0, IBN)         # panel-local col   (0..IBN-1 -> cols kk..kk+IBN-1)
    rmask = rm < M
    cmask = cn < ib

    # load the panel W[kk:kk+M, kk:kk+ib] into one SRAM tile
    rows_g = kk + rm
    cols_g = kk + cn
    A = tl.load(Wb + rows_g[:, None] * sWm + cols_g[None, :] * sWn,
                mask=rmask[:, None] & cmask[None, :], other=zero)

    for j in range(ib):
        col_j = tl.sum(tl.where(cn[None, :] == j, A, zero), axis=1)
        alpha = tl.sum(tl.where(rm == j, col_j, zero))
        xnorm_sq = tl.sum(tl.where(rm > j, col_j * col_j, zero))
        norm = tl.sqrt(alpha * alpha + xnorm_sq)
        beta = tl.where(alpha >= zero, -norm, norm)
        reflect = xnorm_sq > zero
        beta_eff = tl.where(reflect, beta, alpha)
        tau = tl.where(reflect, (beta - alpha) / beta, zero)
        denom = alpha - beta
        v_tail = col_j / denom
        # R diagonal + reflector tail into the in-SRAM panel
        A = tl.where((rm[:, None] == j) & (cn[None, :] == j), beta_eff, A)
        A = tl.where((rm[:, None] > j) & (cn[None, :] == j), v_tail[:, None], A)
        # write tau
        tl.store(TAUb + (kk + j) * sTauN, tau)
        # write Householder vector vj (0/<j, 1/==j, tail/>j) to the V buffer
        vj = tl.where(rm > j, v_tail, tl.where(rm == j, one, zero))
        vj = tl.where(reflect, vj, tl.where(rm == j, one, zero))
        tl.store(Vb + rows_g * sVm + (kk + j) * sVn, vj, mask=rmask)
        # trailing update within the panel (cols j+1..ib), in SRAM
        pmask = cn[None, :] > j
        w = tau * tl.sum(tl.where(pmask, vj[:, None] * A, zero), axis=0)
        A = tl.where(pmask, A - vj[:, None] * w[None, :], A)

    # write the panel back to W (upper triangle holds R; strict-lower holds the
    # reflector tails, harmless -- _triu_copy only reads the upper triangle)
    tl.store(Wb + rows_g[:, None] * sWm + cols_g[None, :] * sWn,
             A, mask=rmask[:, None] & cmask[None, :])


# ===========================================================================
# Kernel 2: build the WY factor T (DLARFT, Gram-solve form).  One CTA per
# batch element.
#
# The Gram-solve trick (ml-mike.com/writing/qr_v2) builds M = T^{-1} directly
# from the reflector Gram matrix, then inverts the small upper-triangular M
# in-kernel (32 serial steps on a 32x32 tile) to obtain T.  Larfb then applies
# Y = T (T^H) @ W1 with plain GEMMs -- no per-tile serial solve.
# ===========================================================================
@libentry()
@triton.jit
def _larft_kernel(
    V,
    TAU,
    MOUT,
    M,
    ib,
    sVb,
    sVm,
    sVn,
    sTauB,
    sTauN,
    sMb,
    sMm,
    sMn,
    RM: tl.constexpr,
    IBN: tl.constexpr,
    INVERT: tl.constexpr,
):
    """Build the WY factor from the reflector Gram matrix.

    INVERT=True (fp32): store T = (triu(V^H V, 1) + diag(1/tau))^{-1}, so larfb
    applies Y = T (T^H) @ W1 with plain GEMMs (blog w1431).
    INVERT=False (fp64): store M = T^{-1} directly; larfb falls back to its
    in-kernel triangular solve (fp64 GEMMs + serial inversion are slower).
    """
    pid = tle.program_id(0)
    Vb = V + pid * sVb
    TAUb = TAU + pid * sTauB
    Tb = MOUT + pid * sMb

    dt = Vb.dtype.element_ty
    idx = tl.arange(0, IBN)  # row/col index 0..IBN-1
    num_tiles = (M + RM - 1) // RM

    # ---- Gram G = V^H V  (IBN x IBN), one parallel matmul over the M rows ----
    G = tl.zeros((IBN, IBN), dtype=dt)
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + idx[None, :] * sVn
        Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (idx[None, :] < ib), other=0.0)
        G += tl.dot(tl.trans(Vt), Vt, allow_tf32=False)

    # ---- M = T^{-1} = triu(G, 1) + diag(1/tau)  (upper triangular) ----
    tau_vec = tl.load(TAUb + idx * sTauN, mask=idx < ib, other=1.0)
    inv_tau = 1.0 / tau_vec
    Mmat = tl.where(idx[:, None] < idx[None, :], G, 0.0)
    Mmat = tl.where(idx[:, None] == idx[None, :], inv_tau[:, None], Mmat)

    Tmat = tl.zeros((IBN, IBN), dtype=dt)
    if INVERT:
        # ---- invert M -> T by back-substitution (rows ib-1 .. 0) ----
        for jj in tl.static_range(0, IBN):
            i = IBN - 1 - jj
            if i < ib:
                Mrow = tl.sum(tl.where(idx[:, None] == i, Mmat, 0.0), axis=0)
                Mii = tl.sum(tl.where(idx == i, Mrow, 0.0))
                contrib = tl.sum(
                    tl.where(idx[:, None] > i, Mrow[:, None] * Tmat, 0.0), axis=0
                )
                Trow = -contrib / Mii
                Trow = tl.where(idx == i, 1.0 / Mii, Trow)
                Tmat = tl.where(idx[:, None] == i, Trow[None, :], Tmat)
    out = Tmat if INVERT else Mmat

    # ---- store T (INVERT) or M = T^{-1} (upper triangular) ----
    tl.store(
        Tb + idx[:, None] * sMm + idx[None, :] * sMn, out,
        mask=(idx[:, None] < ib) & (idx[None, :] < ib),
    )


# ===========================================================================
# Kernel 3: apply block reflector H = I - V T V^H on the left (DLARFB).
#   C <- C - V Y,  Y = T @ W1   (Q assembly,   UPPER=True)
#   C <- C - V Y,  Y = T^H @ W1 (trailing update, UPPER=False)
# with T pre-computed (inverted) by _larft_kernel -- plain GEMMs only, no
# per-tile serial triangular solve (blog w1431: inverse + GEMM).
# ===========================================================================
@triton.jit
def _larfb_kernel(
    V, TAU, TOUT, C, M, ib, P,
    sVb, sVm, sVn, sTauB, sTauN, sTb, sTm, sTn, sCb, sCm, sCn,
    RM: tl.constexpr, IBN: tl.constexpr, TN: tl.constexpr, UPPER: tl.constexpr,
    SOLVE: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_p = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = TOUT + pid_b * sTb
    Cb = C + pid_b * sCb

    dt = Cb.dtype.element_ty
    col_idx = tl.arange(0, IBN)
    p_idx = pid_p * TN + tl.arange(0, TN)
    pmask = p_idx < P
    num_tiles = (M + RM - 1) // RM

    # ---- load T (fp32, upper triangular) or M = T^{-1} (fp64) ----
    Msram = tl.load(
        Tb + col_idx[:, None] * sTm + col_idx[None, :] * sTn,
        mask=(col_idx[:, None] < ib) & (col_idx[None, :] < ib), other=0.0,
    )

    # ---- W1 = V^H C[:, p-tile] ----
    W1 = tl.zeros((IBN, TN), dtype=dt)
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0)
        Ct = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        W1 += tl.dot(tl.trans(Vt), Ct, allow_tf32=False)
    W1 = tl.where(col_idx[:, None] < ib, W1, 0.0)

    # ---- Y = T (T^H) @ W1 : one GEMM, no serial solve (fp32) ----
    # or in-kernel triangular substitution on M (fp64; fp64 GEMMs + serial
    # inversion are slower than the masked-reduction solve on this hardware).
    Y = tl.zeros((IBN, TN), dtype=dt)
    if SOLVE:
        if UPPER:
            for jj in tl.static_range(0, IBN):
                i = IBN - 1 - jj
                if i < ib:
                    Mrow = tl.sum(tl.where(col_idx[:, None] == i, Msram, 0.0), axis=0)
                    W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                    Mii = tl.sum(tl.where(col_idx == i, Mrow, 0.0))
                    contrib = tl.sum(
                        tl.where(col_idx[:, None] > i, Mrow[:, None] * Y, 0.0), axis=0
                    )
                    Yrow = (W1row - contrib) / Mii
                    Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
        else:
            for i in tl.static_range(0, IBN):
                if i < ib:
                    Mcol = tl.sum(tl.where(col_idx[None, :] == i, Msram, 0.0), axis=1)
                    W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, 0.0), axis=0)
                    Mii = tl.sum(tl.where(col_idx == i, Mcol, 0.0))
                    contrib = tl.sum(
                        tl.where(col_idx[:, None] < i, Mcol[:, None] * Y, 0.0), axis=0
                    )
                    Yrow = (W1row - contrib) / Mii
                    Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
    else:
        Tsram = Msram
        if not UPPER:
            Tsram = tl.trans(Tsram)
        Y = tl.dot(Tsram, W1, allow_tf32=False)
    Y = tl.where(col_idx[:, None] < ib, Y, 0.0)

    # ---- C[:, p-tile] -= V Y ----
    for t in range(num_tiles):
        rows = t * RM + tl.arange(0, RM)
        rmask = rows < M
        v_off = rows[:, None] * sVm + col_idx[None, :] * sVn
        c_off = rows[:, None] * sCm + p_idx[None, :] * sCn
        Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < ib), other=0.0)
        Ct = tl.load(Cb + c_off, mask=rmask[:, None] & pmask[None, :], other=0.0)
        Ct = Ct - tl.dot(Vt, Y, allow_tf32=False)
        tl.store(Cb + c_off, Ct, mask=rmask[:, None] & pmask[None, :])


# ===========================================================================
# Kernel 3b: fused Q assembly -- identity + all panels in ONE launch.
# Q <- (H_0 H_1 ... H_{P-1}) applied to identity: each CTA owns a TN-wide
# column slice of Q and loops the panels in reverse, loading V_p/T_p and
# applying Q <- Q - V_p (T_p (V_p^H Q)) with the same GEMM-only body as
# _larfb_kernel(UPPER=True).  Replaces _identity_kernel + P per-panel larfb
# launches (blog w1422's launch-count lesson: fewer, bigger kernels).
# ===========================================================================
@libentry()
@triton.jit
def _assemble_q_fused_kernel(
    V, TAU, Tbuf, Q, m, n, k, qcols, ib, P,
    sVb, sVm, sVn, sTauB, sTauN, sTb, sTm, sTn, sQb, sQm, sQn,
    RM: tl.constexpr, IBN: tl.constexpr, TN: tl.constexpr, SOLVE: tl.constexpr,
):
    pid_b = tle.program_id(0)
    pid_p = tle.program_id(1)
    Vb = V + pid_b * sVb
    Tb = Tbuf + pid_b * sTb
    Qb = Q + pid_b * sQb

    dt = Vb.dtype.element_ty
    zero = tl.full((), 0.0, dtype=dt)
    one = tl.full((), 1.0, dtype=dt)
    col_idx = tl.arange(0, IBN)
    p_idx = pid_p * TN + tl.arange(0, TN)
    pmask = p_idx < qcols
    rm = tl.arange(0, RM)

    # Q column slice = identity (rows 0..m-1 of the p-tile columns)
    for t in range((m + RM - 1) // RM):
        rows = t * RM + rm
        rmask = rows < m
        qt = tl.where(rows[:, None] == p_idx[None, :], one, zero)
        qt = tl.where(rmask[:, None] & pmask[None, :], qt, zero)
        tl.store(Qb + rows[:, None] * sQm + p_idx[None, :] * sQn, qt,
                 mask=rmask[:, None] & pmask[None, :])

    # apply panels in reverse: Q <- H_p Q
    for pp in range(P - 1, -1, -1):
        kk = pp * ib
        iba = ib
        if kk + iba > k:
            iba = k - kk
        num_tiles = (m - kk + RM - 1) // RM
        # W1 = V_p^H Q[kk:m, p-tile]
        W1 = tl.zeros((IBN, TN), dtype=dt)
        for t in range(num_tiles):
            rows = kk + t * RM + rm
            rmask = rows < m
            v_off = rows[:, None] * sVm + (kk + col_idx)[None, :] * sVn
            q_off = rows[:, None] * sQm + p_idx[None, :] * sQn
            Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < iba), other=zero)
            Qt = tl.load(Qb + q_off, mask=rmask[:, None] & pmask[None, :], other=zero)
            W1 += tl.dot(tl.trans(Vt), Qt, allow_tf32=False)
        W1 = tl.where(col_idx[:, None] < iba, W1, zero)
        # T_p (upper triangular, pre-built by _blocked_qr / larft)
        Tt = tl.load(
            Tb + (kk + col_idx)[:, None] * sTm + (kk + col_idx)[None, :] * sTn,
            mask=(col_idx[:, None] < iba) & (col_idx[None, :] < iba), other=zero,
        )
        Y = tl.zeros((IBN, TN), dtype=dt)
        if SOLVE:
            # fp64: Tbuf holds M = T^{-1} -> solve M Y = W1 by back-substitution
            for jj in tl.static_range(0, IBN):
                i = IBN - 1 - jj
                if i < iba:
                    Mrow = tl.sum(tl.where(col_idx[:, None] == i, Tt, zero), axis=0)
                    W1row = tl.sum(tl.where(col_idx[:, None] == i, W1, zero), axis=0)
                    Mii = tl.sum(tl.where(col_idx == i, Mrow, zero))
                    contrib = tl.sum(
                        tl.where(col_idx[:, None] > i, Mrow[:, None] * Y, zero), axis=0
                    )
                    Yrow = (W1row - contrib) / Mii
                    Y = tl.where(col_idx[:, None] == i, Yrow[None, :], Y)
        else:
            Y = tl.dot(Tt, W1, allow_tf32=False)
        Y = tl.where(col_idx[:, None] < iba, Y, zero)
        # Q[kk:m, p-tile] -= V_p Y
        for t in range(num_tiles):
            rows = kk + t * RM + rm
            rmask = rows < m
            v_off = rows[:, None] * sVm + (kk + col_idx)[None, :] * sVn
            q_off = rows[:, None] * sQm + p_idx[None, :] * sQn
            Vt = tl.load(Vb + v_off, mask=rmask[:, None] & (col_idx[None, :] < iba), other=zero)
            Qt = tl.load(Qb + q_off, mask=rmask[:, None] & pmask[None, :], other=zero)
            Qt = Qt - tl.dot(Vt, Y, allow_tf32=False)
            tl.store(Qb + q_off, Qt, mask=rmask[:, None] & pmask[None, :])


# ===========================================================================
# Kernel 4: copy the upper triangle of W into R (zero below).  R[i,j]=W[i,j] if i<=j.
# ===========================================================================
@libentry()
@triton.jit
def _triu_copy_kernel(W, ROUT, rm, n, sWb, sWm, sWn, sRb, sRm, sRn, BLOCK: tl.constexpr):
    pid_b = tle.program_id(0)  # batch
    pid_e = tle.program_id(1)  # element tile
    numel = rm * n
    offs = pid_e * BLOCK + tl.arange(0, BLOCK)
    mmask = offs < numel
    i = offs // n
    j = offs % n
    val = tl.load(W + pid_b * sWb + i * sWm + j * sWn, mask=mmask, other=0.0)
    tl.store(ROUT + pid_b * sRb + i * sRm + j * sRn, tl.where(i <= j, val, 0.0), mask=mmask)


# ===========================================================================
# Kernel 6: stack two n-wide matrices vertically into a (rows0+rows1) x n buffer
# ===========================================================================
@libentry()
@triton.jit
def _vstack_kernel(TOP, BOT, OUT, rows0, rows1, n,
                   sTb, sTm, sTn, sBb, sBm, sBn, sOb, sOm, sOn, BLOCK: tl.constexpr):
    pid_b = tle.program_id(0)
    pid_e = tle.program_id(1)
    total = (rows0 + rows1) * n
    offs = pid_e * BLOCK + tl.arange(0, BLOCK)
    m = offs < total
    i = offs // n          # output row
    j = offs % n
    from_top = i < rows0
    v_top = tl.load(TOP + pid_b * sTb + i * sTm + j * sTn, mask=m & from_top, other=0.0)
    v_bot = tl.load(BOT + pid_b * sBb + (i - rows0) * sBm + j * sBn, mask=m & (~from_top), other=0.0)
    v = tl.where(from_top, v_top, v_bot)
    tl.store(OUT + pid_b * sOb + i * sOm + j * sOn, v, mask=m)


# ===========================================================================
# Kernel 7: triangular solve  X = A R^{-1}  with R upper triangular (n x n)
# Each program handles one batch and solves  x R = a  for many rows of A.
# ===========================================================================
@libentry()
@triton.jit
def _trsm_kernel(A, R, X, m, n, sAb, sAm, sAn, sRb, sRm, sRn, sXb, sXm, sXn,
                 TM: tl.constexpr, BN: tl.constexpr):
    pid = tle.program_id(0)   # batch
    tt = tle.program_id(1)    # row-tile
    Ab = A + pid * sAb
    Rb = R + pid * sRb
    Xb = X + pid * sXb
    dt = Ab.dtype.element_ty

    n_idx = tl.arange(0, BN)  # 0..BN-1, covers columns
    # load R (n x n) into a (BN,BN) tile
    Rtile = tl.load(Rb + n_idx[:, None] * sRm + n_idx[None, :] * sRn,
                    mask=(n_idx[:, None] < n) & (n_idx[None, :] < n), other=0.0)

    rows = tt * TM + tl.arange(0, TM)
    rmask = rows < m
    a_tile = tl.load(Ab + rows[:, None] * sAm + n_idx[None, :] * sAn,
                     mask=rmask[:, None] & (n_idx[None, :] < n), other=0.0)
    x_tile = tl.zeros((TM, BN), dtype=dt)
    # forward substitution for  X R = A  (R upper triangular):
    #   x[i,j] = (a[i,j] - sum_{k<j} x[i,k] R[k,j]) / R[j,j],  j = 0..n-1
    for jcol in tl.static_range(0, BN):
        if jcol < n:
            Rj = tl.sum(tl.where(n_idx[None, :] == jcol, Rtile, 0.0), axis=1)  # R[:, jcol]
            a_j = tl.sum(tl.where(n_idx[None, :] == jcol, a_tile, 0.0), axis=1)  # a[:, jcol]
            contrib = tl.sum(
                tl.where(n_idx[None, :] < jcol, x_tile * Rj[None, :], 0.0), axis=1
            )  # sum_{k<jcol} x[i,k] R[k,jcol]
            rjj = tl.sum(tl.where(n_idx == jcol, Rj, 0.0))  # R[jcol, jcol]
            xj = (a_j - contrib) / rjj
            x_tile = tl.where(n_idx[None, :] == jcol, xj[:, None], x_tile)
    tl.store(Xb + rows[:, None] * sXm + n_idx[None, :] * sXn, x_tile,
             mask=rmask[:, None] & (n_idx[None, :] < n))


# ===========================================================================
# Python orchestration (memory/layout + kernel launches only)
# ===========================================================================
def _launch_geqrt_sram(W, V, tau, m, n, k, kk, ib, B):
    """SRAM-resident panel factorisation (single CTA per batch)."""
    ib_active = min(ib, k - kk)
    M = m - kk
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    _geqrt_sram_kernel[(B,)](
        W, V, tau, M, ib_active, kk, n, k,
        sWb, sWm, sWn, sVb, sVm, sVn, sTauB, sTauN,
        BM=triton.next_power_of_2(M), IBN=_PANEL_IB,
    )


def _launch_geqrt(W, V, tau, m, n, k, kk, ib, B):
    ib_active = min(ib, k - kk)
    M = m - kk
    nr = min(ib_active, k - kk)
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    _geqrt_kernel[(B,)](
        W, V, tau, M, kk, ib_active, n, k, nr,
        sWb, sWm, sWn, sVb, sVm, sVn, sTauB, sTauN,
        RM=_PANEL_RM, IBN=_PANEL_IB,
    )


def _launch_geqrt_mcta(W, V, tau, alpha_buf, xnorm_buf, w_sum, ctr,
                       m, n, k, kk, ib, NC, B):
    M = m - kk
    nr = min(ib, k - kk)
    p = kk // ib
    CHUNK = (M + NC - 1) // NC
    NUM_TILES = (CHUNK + _PANEL_RM - 1) // _PANEL_RM
    NSYNC = 2
    sWb, sWm, sWn = W.stride()
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sAB, sAM, _ = alpha_buf.stride()
    sXB, sXM, _ = xnorm_buf.stride()
    sWB, sWM, sWN, _ = w_sum.stride()
    sCtrB, sCtrM, _ = ctr.stride()
    _geqrt_mcta_kernel[(B, NC)](
        W, V, tau, alpha_buf, xnorm_buf, w_sum, ctr,
        M, kk, ib, n, k, nr, p,
        sWb, sWm, sWn, sVb, sVm, sVn, sTauB, sTauN,
        sAB, sAM, sXB, sXM, sWB, sWM, sWN, sCtrB, sCtrM,
        CHUNK=CHUNK, RM=_PANEL_RM, IBN=_PANEL_IB, NC=NC, NSYNC=NSYNC,
        NUM_TILES=NUM_TILES,
    )


def _launch_larft(V, tau, Tout, m, kk, ib, B):
    M = m - kk
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tout.stride()
    _larft_kernel[(B,)](
        V, tau, Tout, M, ib, sVb, sVm, sVn, sTauB, sTauN, sTb, sTm, sTn,
        RM=_PANEL_RM, IBN=_PANEL_IB, INVERT=V.element_size() == 4,
    )


def _launch_larfb(V, tau, Tp, C, m, p, ib, B, upper):
    """Apply block reflector; T loaded from Tp (pre-computed by _launch_larft)."""
    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tp.stride()
    sCb, sCm, sCn = C.stride()
    # TN=32 is a ~1.9x win for the fp32 GEMM path on large trailing updates;
    # fp64 keeps TN=16 (the solve path and fp64 tiles spill at TN=32).
    tn = _LARFB_TN if V.element_size() == 4 else 16
    grid_p = (p + tn - 1) // tn
    _larfb_kernel[(B, grid_p)](
        V, tau, Tp, C, m, ib, p,
        sVb, sVm, sVn, sTauB, sTauN, sTb, sTm, sTn, sCb, sCm, sCn,
        RM=_LARFB_RM, IBN=_PANEL_IB, TN=tn, UPPER=upper,
        SOLVE=V.element_size() != 4,
    )


def _fused_qr(W, A, orig_dtype, batch_shape, m, n, k, mode, B, out_Q=None, out_R=None):
    """Single-launch QR for matrices that fit in shared memory (m, n <= _FUSED_DIM).

    One kernel per matrix does the factorisation, R extraction and Q assembly.
    Writes directly into caller-provided out_Q/out_R when given (true out variant).
    """
    qcols = 0 if mode == "r" else (k if mode == "reduced" else m)
    rrows = k if mode in ("reduced", "r") else m
    dt = W.dtype
    dev = W.device
    # r-mode: the kernel still needs a typed 3-D Q pointer (put_Q=False, never
    # written); use a dummy, and return the caller's (0-element) out_Q as-is.
    if mode == "r":
        Q = torch.empty(B, m, 1, dtype=dt, device=dev)
    else:
        Q = out_Q if out_Q is not None else torch.empty(B, m, qcols, dtype=dt, device=dev)
    R = out_R if out_R is not None else torch.empty(B, rrows, n, dtype=dt, device=dev)
    tau = torch.empty(B, k, dtype=dt, device=dev)
    BM = triton.next_power_of_2(m)
    BN = triton.next_power_of_2(n)
    BQ = triton.next_power_of_2(max(qcols, 1))
    BK = triton.next_power_of_2(max(k, 1))
    sWb, sWm, sWn = W.stride()
    sQb, sQm, sQn = Q.stride()
    sRb, sRm, sRn = R.stride()
    sTauB, sTauN = tau.stride()
    _qr_fused_kernel[(B,)](
        W, Q, R, tau, m, n, k, qcols, rrows, mode != "r",
        sWb, sWm, sWn, sQb, sQm, sQn, sRb, sRm, sRn, sTauB, sTauN,
        BM=BM, BN=BN, BQ=BQ, BK=BK,
    )
    if mode == "r":
        return (out_Q if out_Q is not None else A.new_empty(0),
                R.to(orig_dtype).reshape(*batch_shape, k, n))
    Q_out = Q.to(orig_dtype).reshape(*batch_shape, m, qcols)
    R_out = R.to(orig_dtype).reshape(*batch_shape, rrows, n)
    return (Q_out, R_out)


def _blocked_qr(W, V, tau, Tbuf, m, n, k, ib=_PANEL_IB):
    """In-place blocked Householder QR; leaves R in the upper triangle of W."""
    B = W.shape[0]
    P = (k + ib - 1) // ib
    dt = W.dtype
    dev = W.device
    # _GEQRT_SRAM_MAX_M is calibrated for fp32.  fp64 doubles register pressure
    # (2 regs/elem) and the kernel keeps both the A tile and V_panel live, so
    # the SRAM kernel either spills (BM>=256 -> ~10x slower) or, for small
    # panels, loses to the low-register single-CTA geqrt that reloads from
    # global.  Net: geqrt_sram never wins for fp64 -> disable it.
    sram_max_m = _GEQRT_SRAM_MAX_M if W.element_size() == 4 else 0
    # multi-CTA scratch (one slot per (panel, column), used once -> zeroed once);
    # allocated only when at least one panel actually takes the multi-CTA path.
    needs_mcta = any(
        triton.next_power_of_2(m - kk) > sram_max_m for kk in range(0, k, ib)
    )
    NSYNC = 2
    if needs_mcta:
        alpha_buf = torch.zeros(B, P, ib, dtype=dt, device=dev)
        xnorm_buf = torch.zeros(B, P, ib, dtype=dt, device=dev)
        w_sum = torch.zeros(B, P, ib, ib, dtype=dt, device=dev)
        ctr = torch.zeros(B, P, ib * NSYNC, dtype=torch.int32, device=dev)
    for kk in range(0, k, ib):
        ib_active = min(ib, k - kk)
        M = m - kk
        bm = triton.next_power_of_2(M)
        if bm <= sram_max_m:
            # panel fits SRAM: single-CTA resident factorisation (no global re-reads)
            _launch_geqrt_sram(W, V, tau, m, n, k, kk, ib_active, B)
        else:
            # ceil(M/RM) CTAs -> CHUNK == RM -> the register-resident fast path
            # of _geqrt_mcta_kernel (each CTA loads its row chunk once, no
            # per-reflector global re-reads).  Measured ~1.6x faster than the
            # CHUNK>RM tile-loop path on the early panels of large matrices.
            nc = max(1, min(_MCTA_NC_MAX, (M + _PANEL_RM - 1) // _PANEL_RM))
            if nc >= _MCTA_MIN_NC:
                _launch_geqrt_mcta(W, V, tau, alpha_buf, xnorm_buf, w_sum, ctr,
                                   m, n, k, kk, ib_active, nc, B)
            else:
                _launch_geqrt(W, V, tau, m, n, k, kk, ib_active, B)
        Vp = V[:, kk:m, kk : kk + ib_active]
        taup = tau[:, kk : kk + ib_active]
        Tp = Tbuf[:, kk : kk + ib_active, kk : kk + ib_active]
        if kk + ib_active < n:
            _launch_larft(Vp, taup, Tp, m, kk, ib_active, B)
            C = W[:, kk:m, kk + ib_active : n]
            _launch_larfb(Vp, taup, Tp, C, m - kk, n - (kk + ib_active), ib_active, B, upper=False)


def _assemble_q(V, tau, Tbuf, m, n, k, qcols, ib, B, out):
    """Q <- (H_0 H_1 ... H_{P-1}) applied to identity; writes into `out` (B, m, qcols).

    T is already in Tbuf from _blocked_qr for every panel except the last (no
    trailing update there) -- build that one T, then apply identity + all
    panels in a single fused launch.
    """
    P = (k + ib - 1) // ib
    kk_last = (P - 1) * ib
    ib_last = min(ib, k - kk_last)
    Vp = V[:, kk_last:m, kk_last : kk_last + ib_last]
    taup = tau[:, kk_last : kk_last + ib_last]
    Tp = Tbuf[:, kk_last : kk_last + ib_last, kk_last : kk_last + ib_last]
    _launch_larft(Vp, taup, Tp, m, kk_last, ib_last, B)

    sVb, sVm, sVn = V.stride()
    sTauB, sTauN = tau.stride()
    sTb, sTm, sTn = Tbuf.stride()
    sQb, sQm, sQn = out.stride()
    grid_p = (qcols + _LARFB_TN - 1) // _LARFB_TN
    _assemble_q_fused_kernel[(B, grid_p)](
        V, tau, Tbuf, out, m, n, k, qcols, ib, P,
        sVb, sVm, sVn, sTauB, sTauN, sTb, sTm, sTn, sQb, sQm, sQn,
        RM=_LARFB_RM, IBN=_PANEL_IB, TN=_LARFB_TN,
        SOLVE=V.element_size() != 4,
    )
    return out


def _triu_copy(W, R, rm, n, B):
    sWb, sWm, sWn = W.stride()
    sRb, sRm, sRn = R.stride()
    grid_e = (rm * n + 1023) // 1024
    _triu_copy_kernel[(B, grid_e)](W, R, rm, n, sWb, sWm, sWn, sRb, sRm, sRn, BLOCK=1024)


def _vstack(top, bot, out, B):
    rows0 = top.shape[-2]
    rows1 = bot.shape[-2]
    n = top.shape[-1]
    sTb, sTm, sTn = top.stride()
    sBb, sBm, sBn = bot.stride()
    sOb, sOm, sOn = out.stride()
    total = (rows0 + rows1) * n
    grid_e = (total + 1023) // 1024
    _vstack_kernel[(B, grid_e)](top, bot, out, rows0, rows1, n,
                                sTb, sTm, sTn, sBb, sBm, sBn, sOb, sOm, sOn, BLOCK=1024)


# ---------------------------------------------------------------------------
# TSQR fast path (tall-skinny): local QR of row blocks + flat reduction of R's
# ---------------------------------------------------------------------------
def _tsqr(W, V, tau, Tbuf, m, n, k, mode, B, out_Q=None, out_R=None):
    """Returns (Q or None, R).  R is (B, n, n); Q (reduced) is (B, m, n)."""
    br = max(n + 1, min(m, _TSQR_BLOCK))
    num_blocks = (m + br - 1) // br
    dt = W.dtype
    dev = W.device
    IBN = triton.next_power_of_2(n)

    # ---- Phase 1: fused local QR of ALL blocks in one launch ----
    R_blocks = torch.empty(B, num_blocks, n, n, dtype=dt, device=dev)
    k_max = min(br, n)
    # V_local / TAU_local are fully written by the local-QR kernel before read,
    # so use torch.empty (avoids the zeros re-dispatch kernel under use_gems).
    V_local = torch.empty(B, m, n, dtype=dt, device=dev)
    TAU_local = torch.empty(B, num_blocks, n, dtype=dt, device=dev)
    sWb, sWm, sWn = W.stride()
    sRb, sRm, sRn = R_blocks.stride(0), R_blocks.stride(1), R_blocks.stride(2)
    sVb, sVm, sVn = V_local.stride()
    sTauB = TAU_local.stride(0)
    if br * n <= _TSQR_SRAM_ELEM:
        # narrow blocks: one register-resident CTA per block (no sync at all)
        # W is read-only here -- keep it pristine for the trsm (Q = A R^{-1}).
        W_pristine = W
        BM = triton.next_power_of_2(br)
        _tsqr_local_sram_kernel[(B, num_blocks)](
            W, R_blocks, V_local, TAU_local,
            m, n, br, num_blocks, k_max,
            sWb, sWm, sWn, sRb, sRm, sRn, sVb, sVm, sVn, sTauB,
            BM=BM, IBN=IBN,
            num_warps=max(4, min(16, (BM * IBN) // 4096)),
        )
    else:
        # multi-CTA local QR writes R into W's upper triangle -> clone first.
        W_pristine = W.clone()
        assert W.data_ptr() != W_pristine.data_ptr(), "W and W_pristine alias!"
        NC = max(1, min(_MCTA_NC_MAX, br // _PANEL_RM))
        CHUNK = (br + NC - 1) // NC
        NSYNC = 2
        alpha_buf = torch.zeros(B, num_blocks, n, dtype=dt, device=dev)
        xnorm_buf = torch.zeros(B, num_blocks, n, dtype=dt, device=dev)
        w_sum = torch.zeros(B, num_blocks, n, IBN, dtype=dt, device=dev)
        ctr = torch.zeros(B, num_blocks, n * NSYNC, dtype=torch.int32, device=dev)
        _tsqr_local_kernel[(B, num_blocks * NC)](
            W, R_blocks, V_local, TAU_local,
            alpha_buf, xnorm_buf, w_sum, ctr,
            m, n, br, k_max, num_blocks,
            sWb, sWm, sWn,
            sRb, sRm, sRn,
            sVb, sVm, sVn,
            sTauB,
            alpha_buf.stride(0), alpha_buf.stride(1),
            xnorm_buf.stride(0), xnorm_buf.stride(1),
            w_sum.stride(0), w_sum.stride(1), w_sum.stride(2),
            ctr.stride(0), ctr.stride(1),
            CHUNK=CHUNK, RM=_PANEL_RM, IBN=IBN, NC=NC, NSYNC=NSYNC,
        )

    # ---- Phase 2: flat reduction of R blocks ----
    # Mathematically the final R is just the R of QR([R_0; R_1; ...; R_{b-1}])
    # (stack the num_blocks local R factors into a (num_blocks*n x n) matrix and
    # factor once).  When that stack fits the fused SRAM kernel this is a single
    # launch instead of num_blocks-1 serial vstack+blocked_qr+triu iterations.
    Rm = num_blocks * n
    if num_blocks > 1 and Rm <= _FUSED_M and Rm * n <= _FUSED_ELEM:
        # R_blocks is not needed after phase 2 and the fused kernel only reads
        # its A tile, so pass a view -- no copy.
        Rstack = R_blocks.reshape(B, Rm, n)
        Racc = out_R if out_R is not None else torch.empty(B, n, n, dtype=dt, device=dev)
        tau_p2 = torch.empty(B, n, dtype=dt, device=dev)
        Qdummy = torch.empty(B, Rm, 1, dtype=dt, device=dev)
        sRsb, sRsm, sRsn = Rstack.stride()
        sQdb, sQdm, sQdn = Qdummy.stride()
        sRab, sRam, sRan = Racc.stride()
        sTauP2B, sTauP2N = tau_p2.stride()
        _qr_fused_kernel[(B,)](
            Rstack, Qdummy, Racc, tau_p2, Rm, n, n, 0, n, False,
            sRsb, sRsm, sRsn, sQdb, sQdm, sQdn, sRab, sRam, sRan, sTauP2B, sTauP2N,
            BM=triton.next_power_of_2(Rm), BN=triton.next_power_of_2(n),
            BQ=1, BK=triton.next_power_of_2(n),
        )
    else:
        # fall back: serial pairwise reduction (num_blocks==1, or stack too big for fused)
        Racc = R_blocks[:, 0, :, :]
        for bidx in range(1, num_blocks):
            Rb = R_blocks[:, bidx, :, :]
            rows = 2 * n
            Ws = torch.empty(B, rows, n, dtype=dt, device=dev)
            _vstack(Racc, Rb, Ws, B)
            _blocked_qr(Ws, V, tau, Tbuf, rows, n, min(rows, n))
            Racc = torch.triu(Ws)[:, :n, :n].contiguous()
        # land the result in the caller's out_R if provided
        if out_R is not None:
            out_R.copy_(Racc)
            Racc = out_R

    if mode == "r":
        return None, Racc

    # ---- Phase 3: Q = A @ R^{-1} ----
    # W was modified in-place by the local QR, so use a pristine copy for Q.
    Q = out_Q if out_Q is not None else torch.empty(B, m, n, dtype=dt, device=dev)
    sAb, sAm, sAn = W_pristine.stride()
    sRb2, sRm2, sRn2 = Racc.stride()
    sQb, sQm, sQn = Q.stride()
    grid_m = (m + 15) // 16
    _trsm_kernel[(B, grid_m)](W_pristine, Racc, Q, m, n, sAb, sAm, sAn,
                              sRb2, sRm2, sRn2, sQb, sQm, sQn,
                              TM=16, BN=IBN)
    return Q, Racc


# ===========================================================================
# Public op
# ===========================================================================
def _validate_mode(mode):
    if mode not in ("reduced", "complete", "r"):
        raise ValueError(
            f"linalg_qr: mode must be one of 'reduced', 'complete', 'r', got {mode!r}"
        )


def linalg_qr(A, mode="reduced", *, out=None):
    """Compute the QR decomposition of ``A``, matching ``torch.linalg.qr``.

    If ``out=(Q, R)`` is given the factors are written directly into those
    caller-owned buffers (no internal allocation / copy of the outputs) -- this
    is the path used by ``linalg_qr_out`` / ``torch.linalg.qr(A, out=...)``.
    """
    logger.debug("GEMS LINALG_QR")
    _validate_mode(mode)

    if A.dim() < 2:
        raise RuntimeError("linalg_qr: input must have at least 2 dimensions")
    if A.dtype not in (torch.float32, torch.float64):
        raise NotImplementedError(
            "FlagGems linalg_qr currently supports float32 and float64 inputs; "
            f"got dtype={A.dtype}"
        )

    orig_dtype = A.dtype
    batch_shape = A.shape[:-2]
    m, n = A.shape[-2], A.shape[-1]
    k = min(m, n)
    B = 1
    for d in batch_shape:
        B *= d

    # Read-only view of A.  The fused path and the TSQR register-resident path
    # never write W (kernels only read it), so no copy is needed there; the
    # mutating paths (blocked / multi-CTA TSQR) clone below.
    W = A.reshape(B, m, n)

    qcols = 0 if mode == "r" else (k if mode == "reduced" else m)
    rrows = k if mode in ("reduced", "r") else m
    # Resolve caller-provided output buffers (reshaped to the (B, ...) layout the
    # kernels write).  These are views of the user's tensors for the contiguous
    # case, so the kernels write the user's memory directly -- no alloc/copy.
    out_Q = out_R = None
    if out is not None:
        out_Q, out_R = out
        out_Q = out_Q.reshape(B, m, qcols) if qcols else out_Q.reshape(0)
        out_R = out_R.reshape(B, rrows, n)
    # TSQR only for tall-skinny that fused can't fit (m*n > _FUSED_ELEM or m > _FUSED_M);
    # smaller tall-skinny (e.g. 64×16, 128×32) are faster via the single-launch fused kernel.
    # TSQR also needs m >= _TSQR_MIN_M -- below that the blocked path's zero-sync SRAM
    # panels beat TSQR's per-block local-QR sync (e.g. 512x64, 256x64).
    is_ts = (m >= _TSQR_ASPECT * n) and (m >= _TSQR_MIN_M) and (n > 0) and (mode in ("reduced", "r"))
    # _FUSED_ELEM is calibrated for fp32; fp64 doubles register pressure in the
    # fused kernel (A tile + Q tile both live during Q assembly), so larger fp64
    # matrices spill and run ~2x slower than the blocked path (e.g. 64x64 fp64:
    # fused 1120us vs blocked 581us).  Cut the cap to ~1/4 for fp64 -- small /
    # batched matrices stay fused (single launch wins), only the larger singles
    # divert to blocked.
    fused_elem = _FUSED_ELEM if A.element_size() == 4 else _FUSED_ELEM // 4
    fits_fused = (m <= _FUSED_M and n <= _FUSED_DIM and m * n <= fused_elem
                  and (mode == "r" or qcols * m <= fused_elem))
    if is_ts:
        fits_fused = fits_fused and (m <= _FUSED_TALL_M)
    tall_skinny = is_ts and not fits_fused
    fused = fits_fused
    if fused:
        return _fused_qr(W, A, orig_dtype, batch_shape, m, n, k, mode, B, out_Q, out_R)

    # large matrices: allocate the blocked-QR scratch and pick TSQR vs blocked.
    # Use torch.empty (not torch.zeros): under use_gems, torch.zeros re-dispatches
    # to a gems zeros *kernel* (one extra launch each).  V/tau/Tbuf are fully
    # written by the geqrt/larft kernels before they are read, so no zero-init
    # is needed.
    V = torch.empty(B, m, k, dtype=W.dtype, device=W.device)
    tau = torch.empty(B, k, dtype=W.dtype, device=W.device)
    Tbuf = torch.empty(B, k, k, dtype=W.dtype, device=W.device)

    tall_skinny = is_ts
    if tall_skinny:
        Qm, Rm = _tsqr(W, V, tau, Tbuf, m, n, k, mode, B, out_Q, out_R)
        if mode == "r":
            return (out_Q if out_Q is not None else A.new_empty(0),
                    Rm.reshape(*batch_shape, n, n))
        return (Qm.reshape(*batch_shape, m, n), Rm.reshape(*batch_shape, n, n))

    # blocked Householder path (large matrices): kernels write W in place
    W = W.clone()
    _blocked_qr(W, V, tau, Tbuf, m, n, k)

    if mode == "r":
        R = out_R if out_R is not None else torch.empty(B, k, n, dtype=W.dtype, device=W.device)
        _triu_copy(W, R, k, n, B)
        return (out_Q if out_Q is not None else A.new_empty(0),
                R.reshape(*batch_shape, k, n))

    Q = out_Q if out_Q is not None else torch.empty(B, m, qcols, dtype=W.dtype, device=W.device)
    _assemble_q(V, tau, Tbuf, m, n, k, qcols, _PANEL_IB, B, Q)

    R = out_R if out_R is not None else torch.empty(B, qcols if mode == "complete" else k, n, dtype=W.dtype, device=W.device)
    _triu_copy(W, R, R.shape[-2], n, B)

    if mode == "reduced":
        return (Q.reshape(*batch_shape, m, k), R.reshape(*batch_shape, k, n))
    return (Q.reshape(*batch_shape, m, m), R.reshape(*batch_shape, m, n))


def linalg_qr_out(A, mode="reduced", *, Q, R):
    """``out=`` variant of :func:`linalg_qr` (matches ``aten::linalg_qr.out``).

    Writes the factors directly into the caller-provided ``Q`` / ``R`` tensors
    (no internal allocation / copy) and returns them.
    """
    logger.debug("GEMS LINALG_QR_OUT")
    return linalg_qr(A, mode, out=(Q, R))
