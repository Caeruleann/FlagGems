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
    yield inp1, inp2

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # The gems kernel only implements the p=2 (Euclidean) path, so vary
        # eps (still hits the same kernel) rather than p.
        yield inp1, inp2, {"eps": 1e-6}
        yield inp1, inp2, {"eps": 0.0}


@pytest.mark.pairwise_distance
def test_pairwise_distance():
    bench = base.GenericBenchmark2DOnly(
        op_name="pairwise_distance",
        input_fn=pairwise_distance_input_fn,
        torch_op=torch.nn.functional.pairwise_distance,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
