import numpy as np
import pytest
from gnn_engineering.layers import gcn_aggregate


def test_isolated_node_keeps_feature_with_self_loop():
    x = np.array([[2.0], [5.0]])
    edge_index = np.empty((2, 0), dtype=np.int64)
    out = gcn_aggregate(x, edge_index)
    np.testing.assert_allclose(out, x)


def test_two_node_symmetric_graph_aggregates():
    x = np.array([[1.0], [3.0]])
    edges = np.array([[0, 1], [1, 0]], dtype=np.int64)
    out = gcn_aggregate(x, edges)
    np.testing.assert_allclose(out, [[2.0], [2.0]])


def test_invalid_edge_rejected():
    with pytest.raises(ValueError):
        gcn_aggregate(np.ones((2, 1)), np.array([[0], [2]]))
