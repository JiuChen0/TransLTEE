import pytest
import numpy as np
from ..utils import wasserstein
import tensorflow as tf

@pytest.mark.parametrize("seq_len, lam, its, sq, backpropT", [
    (3, 10, 10, False, False),
    (4, 20, 20, True, True),
    # add more parameter sets here as needed
])
def test_wasserstein(seq_len, lam, its, sq, backpropT):

    # Create dummy inputs
    X_ls = np.random.rand(10, seq_len, 5).astype(np.float32)
    t = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
    p = 0.5

    # Call the function
    result = wasserstein(seq_len, X_ls, t, p, lam=lam, its=its, sq=sq, backpropT=backpropT)

    # Add your assertions here.
    assert isinstance(result, tf.Tensor)
    assert result.dtype == tf.float32
