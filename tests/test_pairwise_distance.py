import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

# The gems kernel currently implements only the p=2 (Euclidean) path, so every
# case uses p=2.0 (also the default of torch.nn.functional.pairwise_distance).
# The implementation indexes x2 row-by-row like x1, so it does not broadcast;
# both inputs therefore share the same shape.
SHAPES = [
    (7,),  # 1-D: a single pair of D-dim vectors -> scalar output
    (256,),  # 1-D, larger feature dim
    (2, 3),
    (64, 64),
    (1024, 256),
]


@pytest.mark.pairwise_distance
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-8, 0.0])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_pairwise_distance_accuracy(shape, eps, keepdim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = torch.nn.functional.pairwise_distance(
        ref_x1, ref_x2, p=2.0, eps=eps, keepdim=keepdim
    )
    with flag_gems.use_gems():
        res_out = torch.nn.functional.pairwise_distance(
            x1, x2, p=2.0, eps=eps, keepdim=keepdim
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
