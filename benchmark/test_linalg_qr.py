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

from . import base


def _gems_qr_out(A, out=None, mode="reduced"):
    # adapt torch.linalg.qr's `out=(Q, R)` convention to the gems op signature
    Q, R = out
    return flag_gems.linalg_qr_out(A, mode, Q=Q, R=R)

# Representative shapes covering all routing paths:
#   fused (small square / tall / wide), blocked (large square), TSQR (tall-skinny).
# Wide shapes (m < n) are included since they exercise a different Q/R layout.
QR_SHAPES = [
    # square: tiny → large
    (8, 8),
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (4096, 4096),
    # tall (m > n): fused, routing-win, TSQR
    (128, 32),
    (512, 64),
    (4096, 4),
    (8192, 8),
    # wide (m < n)
    (8, 32),
    (16, 64),
    (32, 128),
    (64, 256),
    # batched
    (64, 8, 8),
    (128, 32, 32),
    (32, 128, 128),
    (4, 1024, 1024),
]


class QRBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QR_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield (torch.randn(shape, dtype=cur_dtype, device=self.device),)


@pytest.mark.linalg_qr
def test_linalg_qr():
    bench = QRBenchmark(
        op_name="linalg_qr",
        torch_op=torch.ops.aten.linalg_qr,
        gems_op=flag_gems.linalg_qr,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()


class QROutBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = QR_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            A = torch.randn(shape, dtype=cur_dtype, device=self.device)
            *batch, m, n = shape
            k = min(m, n)
            Q = torch.empty(*batch, m, k, dtype=cur_dtype, device=self.device)
            R = torch.empty(*batch, k, n, dtype=cur_dtype, device=self.device)
            yield (A, {"out": (Q, R)})


@pytest.mark.linalg_qr_out
def test_linalg_qr_out():
    bench = QROutBenchmark(
        op_name="linalg_qr_out",
        torch_op=torch.linalg.qr,
        gems_op=_gems_qr_out,
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
