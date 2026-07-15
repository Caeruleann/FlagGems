import pytest
import torch

from . import base, consts, utils


def composed_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    """NPU-native pairwise_distance via basic torch ops (sub+abs+pow+sum).
    Supports arbitrary p, unlike torch_npu's LpNormV2 which only accepts {0,1,2}.
    Used as the benchmark reference when aten would crash on p not in {0,1,2,inf,-inf}."""
    diff = torch.abs(x1 - x2 + eps)
    if p == float("inf"):
        return torch.amax(diff, dim=-1, keepdim=keepdim)
    elif p == float("-inf"):
        return torch.amin(diff, dim=-1, keepdim=keepdim)
    elif p == 0.0:
        return torch.sum(diff != 0, dim=-1, keepdim=keepdim, dtype=torch.float32).to(
            x1.dtype
        )
    else:
        return torch.pow(
            torch.sum(torch.pow(diff, p), dim=-1, keepdim=keepdim), 1.0 / p
        ).to(x1.dtype)


# torch_npu's native pairwise_distance only supports p in {0, 1, 2} -- inf,
# -inf, and arbitrary real p all crash (core dump). Fall back to the composed
# version for any p outside {0, 1, 2} so the benchmark can sweep arbitrary p.
_ATEN_SUPPORTED_P = (0.0, 1.0, 2.0)


def safe_pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False):
    if base.vendor_name == "ascend" and p not in _ATEN_SUPPORTED_P:
        return composed_pairwise_distance(x1, x2, p=p, eps=eps, keepdim=keepdim)
    return torch.nn.functional.pairwise_distance(
        x1, x2, p=p, eps=eps, keepdim=keepdim
    )


def pairwise_distance_input_fn(shape, dtype, device):
    # x1 and x2 must be broadcastable. Use identical (M, N) shapes so that
    # the op computes one p-distance per row -> M pairs of D-dim vectors,
    # matching both torch.nn.functional.pairwise_distance and the gems kernel
    # (which does N, D = x1.shape and reduces over D).
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Arbitrary real p is supported; sweep several p values plus eps.
        for p in (float("-inf"), float("inf"), 0.0, 1.0, 2.0, 6.6):
            yield inp1, inp2, {"p": p}
    else:
        yield inp1, inp2  # default p=2.0


class PairwiseDistanceBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # Keep the parent's large-N 2-D shapes, then add the small-N-large-D
        # regime: one program per row => few rows => SM underutilization (the
        # case a 2-D / split-K grid targets). (1, D) is equivalent to a 1-D
        # single pair here (grid = 1 program).
        shapes = super().set_more_shapes()
        shapes += [(1, 65536), (8, 65536), (64, 65536), (1, 10000000)]
        return shapes


@pytest.mark.pairwise_distance
def test_pairwise_distance():
    bench = PairwiseDistanceBenchmark(
        op_name="pairwise_distance",
        input_fn=pairwise_distance_input_fn,
        torch_op=safe_pairwise_distance,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
