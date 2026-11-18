# XGBoost --- Complete Theory Notes

These notes cover XGBoost from its motivation and mathematical
foundations through tree construction, regularization, training
strategy, and hyperparameter reasoning. They intentionally contain **no
implementation code**.

## Learning sequence

1.  [Introduction and Intuition](01-Introduction-and-Intuition.md)
2.  [Objective Function and
    Regularization](02-Objective-and-Regularization.md)
3.  [Gradients, Hessians, and Taylor
    Expansion](03-Gradient-Hessian-and-Taylor-Expansion.md)
4.  [Optimal Leaf Weights, Similarity Score, and Split
    Gain](04-Leaf-Weights-and-Split-Gain.md)
5.  [Tree Construction and Growth](05-Tree-Construction-and-Growth.md)
6.  [Training Strategy and Hyperparameter
    Tuning](06-Training-and-Hyperparameter-Tuning.md)

## The central idea

XGBoost builds an additive ensemble of decision trees. At each boosting
iteration, it adds a tree that improves the current predictions. It uses
a regularized objective and a second-order approximation of the loss.
The first derivative (gradient) describes the local slope; the second
derivative (Hessian) describes local curvature. Gradients and Hessians
are aggregated inside leaves to calculate optimal leaf values and
evaluate candidate splits.

The complete mathematical chain is:

**Current predictions → loss → gradients and Hessians → aggregate G and
H per leaf → optimal leaf weight → leaf score → split gain → choose a
tree structure → update predictions.**

## How to study these notes

-   Understand the meaning of every symbol before memorizing a formula.
-   Re-derive the leaf weight and split gain formulas on paper.
-   Work through both the regression and binary-classification examples.
-   Explain why each regularization parameter exists, not merely what it
    is called.
-   Treat hyperparameter values as hypotheses to validate, not universal
    constants.

## Scope and terminology

The notes focus on the tree booster and the standard regularized
second-order formulation. Exact behavior and available options can vary
by XGBoost version, objective, and tree method. Distinguish mathematical
intuition from implementation-specific details.
