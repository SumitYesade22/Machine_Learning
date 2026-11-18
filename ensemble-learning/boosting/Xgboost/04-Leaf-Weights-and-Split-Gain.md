# 4. Optimal Leaf Weights, Similarity Score, and Split Gain

## 4.1 From observations to a leaf

A decision tree sends observations into leaves. Every observation in the
same leaf receives the same output from that tree.

For a leaf `j`, define:

`G_j = Σ_(i in I_j) g_i`

`H_j = Σ_(i in I_j) h_i`

Where `I_j` is the set of observations assigned to leaf `j`.

`G_j` is the sum of gradients. `H_j` is the sum of Hessians.

The leaf does not need to retain every individual derivative for the
basic weight calculation; the aggregate pair `(G_j, H_j)` is sufficient
under the standard approximation.

## 4.2 Deriving the optimal leaf weight

Using L2 regularization and considering one leaf, its relevant objective
is:

`Obj(w) = G w + (1/2)(H + λ)w²`

Here:

-   `w` is the leaf output.
-   `G` is the sum of gradients in the leaf.
-   `H` is the sum of Hessians in the leaf.
-   `λ` is the L2 regularization parameter.

Differentiate with respect to `w`:

`dObj/dw = G + (H + λ)w`

Set the derivative to zero:

`G + (H + λ)w = 0`

Rearrange:

`(H + λ)w = -G`

Therefore:

`w* = -G / (H + λ)`

This is the optimal leaf weight for the regularized quadratic
approximation, assuming the denominator is positive.

## 4.3 Interpreting the formula

`w* = -G / (H + λ)`

-   The negative sign means the update generally moves against the
    aggregate gradient.
-   Larger absolute `G` can produce a larger correction.
-   Larger `H` indicates greater local curvature and reduces the
    magnitude of the correction.
-   Larger `λ` also reduces the magnitude of the correction.

Lambda therefore shrinks leaf outputs toward zero.

The formula is Newton-like: it uses a gradient-like numerator and
curvature-like denominator.

## 4.4 Regression interpretation

For squared-error loss:

`g_i = yhat_i - y_i`

`h_i = 1`

Thus, in a leaf containing `n` observations:

`G = Σ_i (yhat_i - y_i)`

`H = n`

The optimal leaf weight becomes:

`w* = -Σ_i(yhat_i-y_i) / (n+λ)`

Equivalently:

`w* = Σ_i(y_i-yhat_i) / (n+λ)`

If `λ=0`, this is the average residual in the leaf. This links XGBoost's
leaf update to the residual-fitting intuition of gradient boosting for
squared-error regression.

## 4.5 Similarity score

Substitute the optimal weight back into the leaf objective:

`Obj(w*) = G w* + (1/2)(H+λ)(w*)²`

Using `w* = -G/(H+λ)`:

First term:

`G w* = -G²/(H+λ)`

Second term:

`(1/2)(H+λ)(w*)² = (1/2)G²/(H+λ)`

Therefore:

`Obj(w*) = -(1/2)G²/(H+λ)`

Define the leaf score:

`S = G²/(H+λ)`

Then:

`Obj(w*) = -(1/2)S`

Some presentations call `S` the similarity score; the optimized
objective contribution is `-S/2`. Keep this factor of one-half clear
when comparing formulas.

The word *similarity* here does not mean similarity between two data
points. It is a score derived from the optimized regularized objective
contribution of a leaf.

## 4.6 Why the score is useful

For a proposed tree structure, each leaf can be summarized by its score.
A structure with child leaves that achieve a better combined score than
the parent can reduce the approximated objective.

This gives a way to evaluate splits without repeatedly optimizing each
leaf weight from scratch.

## 4.7 Deriving split gain

Suppose a parent node is split into left and right children.

Parent aggregates:

`G_P = G_L + G_R`

`H_P = H_L + H_R`

The parent score is:

`S_P = G_P²/(H_P+λ)`

The child scores are:

`S_L = G_L²/(H_L+λ)`

`S_R = G_R²/(H_R+λ)`

The split creates one additional leaf, so the regularized gain includes
the gamma penalty.

The gain is:

`Gain = (1/2)[S_L + S_R - S_P] - γ`

Expanded:

`Gain = (1/2)[G_L²/(H_L+λ) + G_R²/(H_R+λ) - G_P²/(H_P+λ)] - γ`

A positive gain means the split improves the regularized local objective
approximation under this convention. A larger gain indicates a more
beneficial split among the candidates being compared.

## 4.8 Numerical example: binary classification

Consider four observations with initial probability `p=0.5`.

For logistic loss:

`g = p-y`

`h = p(1-p)`

  Observation     Feature value   Target y   Gradient g   Hessian h
  ------------- --------------- ---------- ------------ -----------
  A                           1          0         +0.5        0.25
  B                           2          0         +0.5        0.25
  C                           3          1         -0.5        0.25
  D                           4          1         -0.5        0.25

Set `λ=1` and `γ=0`.

### Parent node

`G_P = 0.5 + 0.5 - 0.5 - 0.5 = 0`

`H_P = 0.25 × 4 = 1`

`S_P = 0²/(1+1) = 0`

### Candidate split: feature \<= 2

Left child contains A and B:

`G_L = 1`

`H_L = 0.5`

`S_L = 1²/(0.5+1) = 1/1.5 = 0.6667`

Right child contains C and D:

`G_R = -1`

`H_R = 0.5`

`S_R = (-1)²/(0.5+1) = 0.6667`

### Gain

`Gain = (1/2)(0.6667 + 0.6667 - 0) - 0`

`Gain ≈ 0.6667`

The split separates the two negative examples from the two positive
examples and has positive gain.

### Leaf weights

Left:

`w_L = -1/(0.5+1) = -0.6667`

Right:

`w_R = -(-1)/(0.5+1) = +0.6667`

With learning rate `eta=0.3`, the applied changes are approximately
`-0.2` and `+0.2` in raw-margin space.

## 4.9 Numerical example: regression

Targets are 40, 50, 70, and 80. Suppose the current prediction for every
observation is 60.

For half-squared-error loss:

`g = yhat-y`

`h=1`

  Observation     Feature value   Target   Current prediction     g   h
  ------------- --------------- -------- -------------------- ----- ---
  A                           1       40                   60   +20   1
  B                           2       50                   60   +10   1
  C                           3       70                   60   -10   1
  D                           4       80                   60   -20   1

Set `λ=1`, `γ=0`.

Parent: `G_P=0`, `H_P=4`, `S_P=0`.

For split `feature <= 2`:

Left: `G_L=30`, `H_L=2`, `S_L=900/3=300`.

Right: `G_R=-30`, `H_R=2`, `S_R=900/3=300`.

Gain:

`Gain = (1/2)(300+300-0)=300`

Weights:

`w_L=-30/(2+1)=-10`

`w_R=-(-30)/(2+1)=+10`

With `eta=0.3`, the prediction changes by -3 on the left and +3 on the
right. Predictions become 57, 57, 63, and 63.

## 4.10 L1-regularized leaf weight

When alpha is included, the optimal weight is:

`w* = -S(G, α)/(H+λ)`

Where:

`S(G, α) = sign(G) × max(|G|-α, 0)`

If the absolute aggregate gradient is no larger than alpha, the
numerator becomes zero and the leaf output is zero.

The corresponding score uses the thresholded gradient:

`S_leaf = [S(G, α)]²/(H+λ)`

The exact objective convention still includes the one-half factor in the
optimized objective.

## 4.11 What to remember

-   `G` summarizes first-order loss information.
-   `H` summarizes second-order curvature information.
-   The optimal leaf weight is `-G/(H+lambda)` for L2-only
    regularization.
-   The leaf score is `G²/(H+lambda)`.
-   The optimized objective contribution is `-1/2` times that score.
-   Split gain compares the two child scores with the parent score and
    subtracts gamma.
-   The score is an objective-derived quantity, not a distance or
    similarity between observations.
