# 1. Introduction and Intuition

## 1.1 What is XGBoost?

XGBoost stands for **Extreme Gradient Boosting**. It is an optimized
gradient-boosting library whose tree booster constructs an ensemble of
decision trees sequentially.

It is not a single decision tree. It is a collection of trees whose
outputs are combined to produce one prediction.

For a regression problem, the model prediction can be represented as:

`yhat_i = f_0(x_i) + f_1(x_i) + ... + f_K(x_i)`

Here:

-   `x_i` is the feature vector for observation `i`.
-   `f_0` is the initial prediction.
-   `f_k` is the contribution made by the tree added at boosting
    iteration `k`.
-   `K` is the number of boosting rounds.

For classification, the additive model is generally built in
**raw-margin space**. For binary logistic classification, the raw margin
is transformed into a probability using the sigmoid function.

## 1.2 Why do we need boosting?

A decision tree partitions feature space into regions and assigns a
prediction to each leaf. A shallow tree is often a weak learner: it
captures only a limited set of patterns.

Boosting builds a stronger model by adding learners sequentially. Each
new learner is trained in the context of the errors or loss left by the
current ensemble.

The sequence matters:

1.  Build an initial prediction.
2.  Evaluate how well the current model predicts.
3.  Determine how the loss changes with the current prediction.
4.  Build a tree whose leaf outputs correct the current model.
5.  Add the tree's contribution to the ensemble.
6.  Repeat.

The new tree is not trained independently and then averaged with all
previous trees. It is fitted to improve the current ensemble.

## 1.3 Gradient Boosting and XGBoost

XGBoost is based on the gradient-boosting framework, but adds a
particular regularized objective and efficient tree-building machinery.

A useful comparison:

  -----------------------------------------------------------------------
  Aspect                  General gradient        XGBoost tree booster
                          boosting idea           
  ----------------------- ----------------------- -----------------------
  Model                   Additive ensemble       Additive ensemble

  Training                Sequential              Sequential

  Improvement signal      Loss gradients /        First- and second-order
                          negative gradients      loss information

  Tree structure          Usually decision trees  Regularized decision
                                                  trees

  Complexity control      Shrinkage, tree depth,  Explicit
                          subsampling and other   tree-complexity penalty
                          controls may be used    plus training controls

  Split evaluation        Depends on              Derived from
                          implementation          regularized objective
                                                  using aggregated
                                                  gradients and Hessians
  -----------------------------------------------------------------------

Do not say that traditional gradient boosting has no regularization.
Shrinkage, shallow trees, subsampling, and other constraints can
regularize a gradient-boosted model. XGBoost makes tree complexity an
explicit part of its standard objective.

## 1.4 What does "gradient boosting" mean?

The word *gradient* refers to using derivatives of the loss function to
determine how predictions should change.

Suppose the current prediction for observation `i` is `yhat_i`. The loss
is `l(y_i, yhat_i)`. The derivative with respect to the current
prediction tells us the local direction in which the loss increases.

The negative gradient points in the local direction of decreasing loss.
In squared-error regression, the negative gradient is proportional to
the residual. For other loss functions, it is a loss-dependent quantity.

The tree learns a mapping from features to these correction values. It
does not simply memorize one global correction for every observation; it
partitions observations into leaves and assigns a correction to each
region.

## 1.5 Why does XGBoost use second-order information?

A first-order method uses the gradient, which tells us the local slope.
XGBoost's standard tree formulation also uses the Hessian, which tells
us local curvature.

A second-order approximation gives a local quadratic representation of
the loss:

`loss change ≈ gradient term + curvature term`

The quadratic term makes it possible to derive an analytical optimal
output for a leaf under the regularized approximation.

This is closely related to Newton-style optimization. It does not mean
that XGBoost performs ordinary unconstrained Newton optimization over
every model parameter. It uses gradient and Hessian statistics to
optimize the leaf values for a proposed tree structure.

## 1.6 What does a tree contribute?

A tree maps an observation to one leaf. Each leaf has a numeric weight.

For observation `i`, the tree output is the weight of the leaf into
which `x_i` falls:

`f_t(x_i) = w_j` when observation `i` belongs to leaf `j`.

The new ensemble prediction is:

`new_prediction = old_prediction + learning_rate × tree_output`

The learning rate is also called `eta` or shrinkage. It scales the
contribution of each newly added tree.

A tree can therefore make different corrections for different regions of
feature space.

## 1.7 Regression versus binary classification

### Regression

With squared-error loss, the model's prediction is a numerical value. A
tree contributes a numerical correction directly to that prediction.

For example, if the current prediction is 60 and a leaf's tree output is
-10, with learning rate 0.3 the updated prediction is:

`60 + 0.3 × (-10) = 57`

### Binary classification

For binary logistic classification, the additive model is built on a raw
score or margin `z`. The probability is:

`p = 1 / (1 + exp(-z))`

A tree contributes a correction to `z`, not directly to the probability.
After the margin is updated, the sigmoid function maps it to a
probability.

This distinction is essential: **the tree output is a raw-margin
adjustment in logistic classification**.

## 1.8 High-level training flow

1.  Choose an objective/loss function.
2.  Initialize the model's raw predictions.
3.  Calculate the gradient and Hessian for each observation.
4.  Consider candidate tree splits.
5.  Aggregate gradients and Hessians in the candidate child nodes.
6.  Calculate the regularized gain of each candidate split.
7.  Grow the tree subject to constraints.
8.  Calculate optimal leaf weights.
9.  Scale the tree contribution using the learning rate.
10. Update predictions and continue for more boosting rounds.

## 1.9 Mental model to retain

XGBoost repeatedly asks:

-   Given the predictions made so far, how is the loss changing?
-   Which observations should receive similar corrections?
-   Which feature split separates observations that benefit from
    different corrections?
-   Is the improvement large enough to justify the added tree
    complexity?

The rest of the mathematics answers these questions precisely.
