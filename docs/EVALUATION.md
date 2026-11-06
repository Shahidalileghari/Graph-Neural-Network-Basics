# Evaluation protocol

## Node classification

Report at least accuracy and macro-F1. For imbalanced classes, accuracy alone can hide poor minority-class performance. Tune hyperparameters on validation data only, then evaluate the test split once.

Compare:
1. Majority-class baseline.
2. Feature-only MLP.
3. GCN using graph structure and node features.

## Link prediction

Report ROC-AUC and average precision for binary edge classification, plus ranking metrics such as Hits@K or MRR when the evaluation protocol is ranking-based. State the negative-sampling distribution and ratio.

## Reproducibility checklist

- Record random seed and package versions.
- Save dataset version and exact split protocol.
- State whether the task is transductive or inductive.
- Explain which edges are visible during message passing.
- Exclude held-out positive edges from the encoder graph when required by the benchmark protocol.
- Do not select thresholds or hyperparameters using test labels.
- Report mean and standard deviation across multiple seeds for serious comparisons.
