import pytest
import torch

from . import base, consts, utils


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
        torch_op=torch.nn.functional.pairwise_distance,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
