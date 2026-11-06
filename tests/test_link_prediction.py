import numpy as np
import pytest
from gnn_engineering.tasks import dot_product_scores, sample_negative_edges


def test_dot_product_scores():
    z = np.array([[1., 0.], [0., 2.], [1., 1.]])
    edges = np.array([[0, 1], [1, 2]])
    np.testing.assert_allclose(dot_product_scores(z, edges), [0., 2.])


def test_negative_sampler_excludes_positive_and_self_edges():
    pos = np.array([[0, 1], [1, 2]])
    neg = sample_negative_edges(4, pos, 3, seed=2)
    assert neg.shape == (2, 3)
    for u, v in neg.T:
        assert u != v
        assert (min(u, v), max(u, v)) not in {(0, 1), (1, 2)}


def test_impossible_negative_request_rejected():
    with pytest.raises(ValueError):
        sample_negative_edges(2, np.array([[0], [1]]), 1)
