# Common GNN pitfalls

1. **Edge leakage:** held-out target edges accidentally remain in message-passing adjacency.
2. **Feature leakage:** features encode labels or information unavailable at inference time.
3. **Split confusion:** transductive node classification and inductive generalization answer different questions.
4. **Negative-sampling bias:** easy random negatives can inflate link-prediction scores.
5. **Duplicate/directed edges:** decide whether graph semantics are directed, undirected, weighted, or multi-relational.
6. **Oversmoothing:** repeated propagation can make node representations increasingly similar.
7. **Oversquashing:** long-range information is compressed through limited message channels.
8. **No simple baseline:** a GNN may not outperform an MLP when features already explain the target.
9. **Benchmark overclaim:** a single split or seed is not robust evidence of generalization.
10. **Dense adjacency at scale:** the teaching NumPy layer is not intended for large sparse graphs.
