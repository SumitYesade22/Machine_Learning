# 2. Objective Function and Regularization

## 2.1 Why an objective function is needed

Training a model requires a measurable target. XGBoost must balance two
competing goals:

1.  Fit the training data well.
2.  Avoid making the ensemble unnecessarily complex.

A model that only minimizes training loss may create many leaves and
large leaf values to fit small details or noise. Such a model may
generalize poorly.

XGBoost expresses this trade-off through an objective containing a
training-loss term and a regularization term.

## 2.2 The regularized objective

At boosting iteration `t`, the objective can be written as:

`Obj^(t) = Σ_i l(y_i, yhat_i^(t-1) + f_t(x_i)) + Ω(f_t)`

Here:

-   `i` indexes training observations.
-   `y_i` is the true target.
-   `yhat_i^(t-1)` is the current ensemble prediction before adding tree
    `t`.
-   `f_t` is the new tree.
-   `l` is the chosen loss function.
-   `Ω(f_t)` penalizes the complexity of the new tree.

The objective is evaluated for the candidate new tree in the context of
the predictions already produced by the existing ensemble.

## 2.3 Tree complexity penalty

For the standard formulation:

`Ω(f) = γT + (1/2)λ Σ_j w_j²`

Where:

-   `T` = number of leaves in the tree.
-   `w_j` = output weight assigned to leaf `j`.
-   `γ` (gamma) = penalty associated with adding leaves.
-   `λ` (lambda) = L2 regularization strength on leaf weights.

This formula contains two distinct controls.

### Leaf-count penalty: gamma

`γT` penalizes the number of leaves.

A tree with more leaves pays a larger complexity cost. When a split
creates an additional leaf, the objective includes an additional leaf
penalty. In the split-gain formula, this appears as a subtraction of
`gamma` for the split.

Higher gamma means a split must produce a larger improvement in the loss
approximation to be worthwhile.

Gamma is not a penalty on the numerical size of a leaf weight. It
penalizes tree structure, specifically the number of leaves.

### Leaf-weight penalty: lambda

`(1/2)λ Σ_j w_j²` penalizes large leaf weights.

Higher lambda discourages large corrections. It stabilizes the
leaf-weight calculation and can reduce sensitivity to small or noisy
groups of observations.

Lambda is not a penalty on the number of leaves. It penalizes the
magnitude of leaf outputs.

## 2.4 Why are both penalties useful?

Consider two different forms of complexity:

-   Structural complexity: a tree has many leaves and creates many
    separate regions.
-   Numerical complexity: a leaf assigns an unusually large correction.

Gamma controls the first. Lambda controls the second.

A tree could have few leaves but very large weights; lambda addresses
that. Another tree could have many leaves with modest weights; gamma
addresses that.

## 2.5 From the full objective to a tractable approximation

At a boosting iteration, the new tree changes the current prediction by
`f_t(x_i)`. Directly optimizing every possible tree structure against
the original loss is difficult.

XGBoost uses a second-order Taylor approximation around the current
prediction. For observation `i`:

`l(y_i, yhat_i + f_t(x_i)) ≈ l(y_i, yhat_i) + g_i f_t(x_i) + (1/2) h_i f_t(x_i)^2`

Where:

-   `g_i` is the first derivative of the loss with respect to the
    current raw prediction.
-   `h_i` is the second derivative with respect to that prediction.

The term `l(y_i, yhat_i)` does not depend on the new tree, so it can be
treated as a constant while comparing candidate new trees.

The simplified objective becomes:

`Obj^(t) ≈ Σ_i [g_i f_t(x_i) + (1/2)h_i f_t(x_i)^2] + Ω(f_t)`

This is the key transformation. Instead of repeatedly carrying the
original loss expression through every split calculation, the algorithm
uses gradient and Hessian statistics.

## 2.6 Group observations by leaf

Suppose the candidate tree has `T` leaves. Let `I_j` be the set of
observations assigned to leaf `j`.

Every observation in that leaf receives the same tree output `w_j`.

Therefore:

`f_t(x_i) = w_j` for every `i` in `I_j`.

Define the aggregate gradient and Hessian for leaf `j`:

`G_j = Σ_(i in I_j) g_i`

`H_j = Σ_(i in I_j) h_i`

The approximate objective can then be expressed as a sum over leaves:

`Obj^(t) ≈ Σ_j [G_j w_j + (1/2)(H_j + λ)w_j²] + γT`

This form is central to XGBoost. A candidate leaf can be summarized by
two numbers: its total gradient `G` and total Hessian `H`.

## 2.7 L1 regularization: alpha

The objective may also include an L1 penalty:

`Ω(f) = γT + (1/2)λ Σ_j w_j² + α Σ_j |w_j|`

Where `alpha` is the L1 regularization strength.

L1 regularization can push small optimal leaf weights exactly to zero
through a soft-thresholding operation. It can therefore encourage sparse
leaf outputs.

With both L1 and L2 regularization, define the soft-thresholding
function:

`S(G, α) = sign(G) × max(|G| - α, 0)`

The optimal leaf weight becomes:

`w* = -S(G, α) / (H + λ)`

If `|G| <= α`, then `S(G, α) = 0`, so the optimal leaf weight is zero.

When alpha is zero, this reduces to the L2-only formula:

`w* = -G / (H + λ)`

## 2.8 What the parameters mean in practice

  -----------------------------------------------------------------------
  Parameter               What it penalizes or    Increasing it generally
                          controls                does what?
  ----------------------- ----------------------- -----------------------
  `gamma`                 Number of leaves /      Makes splitting more
                          split complexity        conservative

  `lambda`                Squared magnitude of    Shrinks leaf
                          leaf weights            corrections

  `alpha`                 Absolute magnitude of   Can set small leaf
                          leaf weights            corrections to zero
  -----------------------------------------------------------------------

These parameters interact. For example, increasing lambda changes the
score of both parent and child nodes, so it can change which split
appears most beneficial.

## 2.9 Important distinction

Do not confuse:

-   `gamma`: the minimum loss reduction required to make a split
    worthwhile in the regularized gain calculation.
-   `min_child_weight`: a constraint on the sum of Hessians in a child.
-   `max_depth`: a limit on tree depth.
-   `lambda`: L2 regularization on leaf weights.
-   `alpha`: L1 regularization on leaf weights.

They all control complexity or conservatism, but they do so through
different mechanisms.
