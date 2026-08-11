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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

DEVICE = flag_gems.device

# fp64 cases are skipped at runtime on backends that do not support fp64.
_TEST_DTYPES = [torch.float32, torch.float64]

# Shapes covering every routing path of the op: tiny/edge, square, tall, wide,
# panel-boundary, tall-skinny (TSQR), batched, and batched with extra dims.
QR_SHAPES = [
    (1, 1),
    (1, 7),
    (7, 1),
    (8, 8),
    (33, 33),
    (8, 3),
    (64, 32),
    (512, 64),
    (1024, 16),
    (4096, 4),
    (3, 8),
    (8, 32),
    (8, 2048),
    (64, 256),
    (2, 8, 8),
    (3, 13, 7),
    (4, 7, 13),
    (2, 4, 4, 4),
    # regression: partial last panel on the multi-CTA path used to index the
    # sync scratch with kk // ib_active (out of bounds) and corrupt memory
    (2048, 1030),
    (4096, 1000),
    # regression: wide-n tall matrices must take the blocked path (TSQR legacy
    # kernels are only safe for narrow n)
    (4096, 128),
    (4096, 1024),
]

QR_MODES = ["reduced", "complete", "r"]


def _harmonise_r_sign(res_R, ref_R):
    """Align R's column signs to the reference (QR sign ambiguity)."""
    ref_R = ref_R.to(device=res_R.device)
    rows = res_R.shape[-2]
    k = min(rows, res_R.shape[-1])
    sgn = (
        torch.sign(res_R.diagonal(dim1=-2, dim2=-1)[..., :k])
        * torch.sign(ref_R.diagonal(dim1=-2, dim2=-1)[..., :k])
    )
    full = torch.ones(*res_R.shape[:-2], rows, dtype=res_R.dtype, device=res_R.device)
    full[..., :k] = sgn
    return res_R * full.unsqueeze(-1)


def _assert_qr_valid(res_Q, res_R, ref_Q, ref_R, mode, dtype):
    """Check a gems (Q, R) factorisation against the torch reference.

    Verifies that R matches torch's R (sign-harmonised) and is upper
    triangular, and (for non-"r" modes) that Q @ R reconstructs the reference
    factorisation and Q is orthonormal.
    """
    # TF32 (NVIDIA Ampere+) inflates the verification matmuls (Q @ R, Q^H Q)
    # with ~1e-3 noise; turn it off so the checks reflect the real
    # factorisation error.
    torch.backends.cuda.matmul.allow_tf32 = False

    res_R_h = _harmonise_r_sign(res_R, ref_R.to(res_R.dtype))
    utils.gems_assert_close(res_R_h, ref_R.to(res_R.dtype), dtype)
    zeros = utils.to_reference(torch.zeros_like(res_R))
    utils.gems_assert_close(res_R.tril(-1), zeros, dtype)

    if mode == "r":
        return

    k = min(res_R.shape[-2], res_R.shape[-1])
    recon = res_Q @ res_R
    ref_recon = ref_Q @ ref_R
    utils.gems_assert_close(recon, ref_recon, dtype, reduce_dim=k)
    gram = res_Q.transpose(-1, -2) @ res_Q
    eye = torch.eye(
        res_Q.shape[-1], dtype=res_Q.dtype, device=res_Q.device
    ).expand_as(gram)
    expected_eye = utils.to_reference(eye)
    utils.gems_assert_close(gram, expected_eye, dtype, reduce_dim=res_Q.shape[-1])


@pytest.mark.linalg_qr
@pytest.mark.parametrize("shape", QR_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("mode", QR_MODES)
def test_linalg_qr(shape, dtype, mode):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    if mode == "complete" and shape[-2] < shape[-1]:
        pytest.skip("complete mode requires m >= n")

    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_Q, ref_R = torch.linalg.qr(ref_inp, mode=mode)
    with flag_gems.use_gems():
        res_Q, res_R = torch.linalg.qr(inp, mode=mode)

    _assert_qr_valid(res_Q, res_R, ref_Q, ref_R, mode, dtype)


@pytest.mark.linalg_qr_out
@pytest.mark.parametrize("shape", QR_SHAPES)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
@pytest.mark.parametrize("mode", QR_MODES)
def test_linalg_qr_out(shape, dtype, mode):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    if mode == "complete" and shape[-2] < shape[-1]:
        pytest.skip("complete mode requires m >= n")

    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    ref_inp = utils.to_reference(inp)

    ref_Q, ref_R = torch.linalg.qr(ref_inp, mode=mode)

    res_Q = torch.empty(ref_Q.shape, dtype=dtype, device=DEVICE)
    res_R = torch.empty(ref_R.shape, dtype=dtype, device=DEVICE)
    with flag_gems.use_gems():
        out_Q, out_R = torch.linalg.qr(inp, mode=mode, out=(res_Q, res_R))

    # The out variant must write in place and return the same tensors.
    assert out_Q.data_ptr() == res_Q.data_ptr()
    assert out_R.data_ptr() == res_R.data_ptr()

    _assert_qr_valid(out_Q, out_R, ref_Q, ref_R, mode, dtype)


@pytest.mark.linalg_qr
@pytest.mark.parametrize("shape", [(33, 33), (4, 64, 40), (4, 8, 32)])
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_qr_non_contiguous(shape, dtype):
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    full = torch.randn(
        shape[:-2] + (shape[-2] * 2, shape[-1] * 2), dtype=dtype, device=DEVICE
    )
    inp = full[..., ::2, ::2]
    assert not inp.is_contiguous()
    ref_inp = utils.to_reference(inp)

    ref_Q, ref_R = torch.linalg.qr(ref_inp)
    with flag_gems.use_gems():
        res_Q, res_R = torch.linalg.qr(inp)

    _assert_qr_valid(res_Q, res_R, ref_Q, ref_R, "reduced", dtype)


@pytest.mark.linalg_qr
def test_linalg_qr_invalid_mode():
    inp = torch.randn(4, 4, device=DEVICE)
    with pytest.raises(ValueError):
        with flag_gems.use_gems():
            torch.linalg.qr(inp, mode="nonsense")


@pytest.mark.linalg_qr
@pytest.mark.parametrize(
    "shape",
    [
        (64, 32),      # fused path
        (512, 512),    # blocked path
        (2048, 1030),  # blocked, multi-CTA panels + partial last panel
        (16384, 64),   # TSQR legacy (multi-CTA local kernel)
    ],
)
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_qr_input_not_mutated(shape, dtype):
    """linalg_qr must not modify its input tensor (matches torch.linalg.qr).

    The blocked path factors a clone, but the TSQR legacy multi-CTA local
    kernel used to write R into the input's own storage.
    """
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    orig = inp.clone()

    with flag_gems.use_gems():
        torch.linalg.qr(inp, mode="reduced")

    assert torch.equal(inp, orig)


@pytest.mark.linalg_qr
@pytest.mark.parametrize("shape", [(64, 32), (256, 128), (600, 64)])
@pytest.mark.parametrize("dtype", _TEST_DTYPES)
def test_linalg_qr_zero_column(shape, dtype):
    """Inputs with an exact zero column must not produce NaN.

    A zero column has a zero Householder tail norm (tau == 0); an unguarded
    0/0 reflector tail used to leak NaN into the fused kernel's Q assembly.
    R row signs are arbitrary for a zero diagonal, so this checks
    reconstruction and orthonormality instead of R equality.
    """
    if dtype == torch.float64 and not utils.fp64_is_supported:
        pytest.skip("fp64 is not supported on this device")
    inp = torch.randn(shape, dtype=dtype, device=DEVICE)
    inp[..., 1] = 0

    with flag_gems.use_gems():
        res_Q, res_R = torch.linalg.qr(inp, mode="reduced")

    assert not torch.isnan(res_Q).any() and not torch.isnan(res_R).any()
    torch.backends.cuda.matmul.allow_tf32 = False
    k = min(shape[-2], shape[-1])
    utils.gems_assert_close(
        res_Q @ res_R, utils.to_reference(inp), dtype, reduce_dim=k
    )
    gram = res_Q.transpose(-1, -2) @ res_Q
    eye = torch.eye(res_Q.shape[-1], dtype=res_Q.dtype, device=res_Q.device)
    utils.gems_assert_close(
        gram, utils.to_reference(eye), dtype, reduce_dim=res_Q.shape[-1]
    )
    zeros = utils.to_reference(torch.zeros_like(res_R))
    utils.gems_assert_close(res_R.tril(-1), zeros, dtype)
