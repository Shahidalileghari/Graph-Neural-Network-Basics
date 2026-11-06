"""Tiny dependency-light link scoring and negative sampling demonstration."""
import numpy as np
from gnn_engineering.tasks import dot_product_scores, sample_negative_edges


def main():
    positive = np.array([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=np.int64)
    embeddings = np.random.default_rng(7).normal(size=(5, 8))
    negatives = sample_negative_edges(5, positive, 4, seed=7)
    print("positive scores:", dot_product_scores(embeddings, positive))
    print("sampled negative edges:", negatives)
    print("negative scores:", dot_product_scores(embeddings, negatives))


if __name__ == "__main__":
    main()
