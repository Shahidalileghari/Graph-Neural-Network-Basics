import pytest
from gnn_engineering.metrics import accuracy, macro_f1, hits_at_k, mean_reciprocal_rank


def test_classification_metrics():
    assert accuracy([0, 1, 1], [0, 1, 0]) == pytest.approx(2 / 3)
    assert macro_f1([0, 1], [0, 1]) == pytest.approx(1.0)


def test_ranking_metrics():
    assert hits_at_k([1, 4, 9], 5) == pytest.approx(2 / 3)
    assert mean_reciprocal_rank([1, 2, 4]) == pytest.approx((1 + .5 + .25) / 3)
