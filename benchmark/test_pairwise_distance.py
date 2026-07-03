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
    yield inp1, inp2  # default p=2.0

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Arbitrary real p is supported; sweep several p values plus eps.
        for p in (1.0, 3.0, 4.0, 0.5):
            yield inp1, inp2, {"p": p}
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
