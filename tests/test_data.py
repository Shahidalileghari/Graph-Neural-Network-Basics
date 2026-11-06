import numpy as np
import pytest
from gnn_engineering.data import validate_edge_index, validate_node_features, validate_masks


def test_validate_features_and_edges():
    validate_node_features(np.ones((3, 4)))
    validate_edge_index(np.array([[0, 1], [1, 2]]), 3)


def test_invalid_features_rejected():
    with pytest.raises(ValueError):
        validate_node_features(np.array([1, 2]))


def test_overlapping_masks_rejected():
    with pytest.raises(ValueError):
        validate_masks([1, 0], [1, 0], [0, 1], 2)
