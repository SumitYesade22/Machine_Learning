# 3. Gradients, Hessians, and Taylor Expansion

## 3.1 Why derivatives enter boosting

At each boosting iteration, the ensemble already has predictions. The
next tree must improve those predictions.

The loss function tells us how bad a prediction is. Its derivatives tell
us how the loss responds to a small change in the prediction.

For one observation:

`l = l(y, yhat)`

The first derivative is:

`g = ∂l / ∂yhat`

The second derivative is:

`h = ∂²l / ∂yhat²`

In XGBoost's logistic objectives, these derivatives are taken with
respect to the raw margin, not the final probability.

## 3.2 Gradient: local slope

The gradient measures the local rate and direction of change of the loss
as the prediction changes.

-   Positive gradient: increasing the prediction locally increases the
    loss, so a negative update is indicated.
-   Negative gradient: increasing the prediction locally decreases the
    loss, so a positive update is indicated.
-   Gradient near zero: the loss is locally stationary with respect to
    that prediction.

The negative gradient is the local direction of steepest decrease in the
loss.

For squared-error loss, the negative gradient is the residual. For other
losses, it is not generally equal to the raw residual.

## 3.3 Hessian: local curvature

The Hessian in the one-dimensional prediction case is the second
derivative.

It describes how quickly the gradient changes as the prediction changes.

-   Positive Hessian: locally convex curvature.
-   Larger Hessian: stronger local curvature.
-   Small Hessian: flatter local curvature.

The Hessian is not simply another error value. It describes the shape of
the loss around the current prediction.

For multi-dimensional parameter optimization, a Hessian is a matrix of
second partial derivatives. In the common per-observation scalar
prediction formulation used here, it is a scalar second derivative.

## 3.4 Taylor expansion

For a small change `u` in a function's input, a second-order Taylor
approximation is:

`f(x + u) ≈ f(x) + f'(x)u + (1/2)f''(x)u²`

Apply this to the loss, where the change in prediction is the new tree
output `f_t(x_i)`:

`l(y_i, yhat_i + f_t(x_i)) ≈ l(y_i, yhat_i) + g_i f_t(x_i) + (1/2)h_i f_t(x_i)²`

Interpretation:

1.  `l(y_i, yhat_i)` is the loss before adding the tree.
2.  `g_i f_t(x_i)` is the first-order effect of the update.
3.  `(1/2)h_i f_t(x_i)²` is the second-order curvature correction.

The quadratic term makes the approximation more informative than a
purely linear approximation.

## 3.5 Why the second-order approximation is useful

If we used only the gradient term, we would know the local direction but
would not explicitly model curvature in the approximation.

Including the Hessian creates a quadratic expression in the leaf output.
A quadratic function is straightforward to minimize analytically.

That is how the optimal leaf-weight formula becomes possible.

This is the conceptual connection to Newton's method: Newton-style
updates use both slope and curvature. XGBoost uses this information
within the optimization of tree leaf values and split structure.

## 3.6 Derivatives for squared-error regression

Take the loss:

`l(y, yhat) = (1/2)(y - yhat)²`

Differentiate with respect to `yhat`:

`g = ∂l/∂yhat = yhat - y`

Differentiate again:

`h = ∂²l/∂yhat² = 1`

Example: actual target is 40 and current prediction is 60.

-   `g = 60 - 40 = 20`
-   `h = 1`
-   Negative gradient is `-20`, which points toward reducing the
    prediction.

The Hessian is constant because the second derivative of this particular
squared-error loss with respect to the prediction is always 1. It is not
constant because the model is linear.

If the loss is defined without the factor `1/2`, the gradient and
Hessian are scaled by 2. The convention must be kept consistent
throughout a derivation.

## 3.7 Derivatives for binary logistic classification

For binary classification, let the model produce a raw margin `z`.

Convert the margin to probability:

`p = sigmoid(z) = 1 / (1 + exp(-z))`

For a binary target `y` in `{0, 1}`, the logistic loss is:

`l(y, p) = -[y log(p) + (1-y) log(1-p)]`

When differentiating with respect to the raw margin `z`, the derivatives
simplify to:

`g = p - y`

`h = p(1-p)`

The gradient is the difference between predicted probability and target.
The Hessian depends on the current probability.

### Example at initial probability 0.5

If `p = 0.5`:

`h = 0.5 × 0.5 = 0.25`

For a positive example `y=1`:

`g = 0.5 - 1 = -0.5`

For a negative example `y=0`:

`g = 0.5 - 0 = +0.5`

A positive example has a negative gradient, indicating that increasing
the raw margin reduces loss. A negative example has a positive gradient,
indicating that decreasing the raw margin reduces loss.

## 3.8 Why the logistic Hessian changes

`h = p(1-p)`

-   At `p=0.5`, Hessian is `0.25`, its maximum.
-   As `p` approaches 0 or 1, Hessian approaches 0.

This means the local curvature of logistic loss depends on the current
probability.

Consequently, the sum of Hessians is not generally equal to the number
of observations in a node.

## 3.9 Regression and classification compared

  Property                      Squared-error regression   Binary logistic classification
  ----------------------------- -------------------------- --------------------------------
  Model's additive space        Numerical prediction       Raw margin / logit
  Gradient                      `yhat - y`                 `p - y`
  Hessian                       `1`                        `p(1-p)`
  Is Hessian constant?          Yes, for this loss         No
  Tree output changes           Numerical prediction       Raw margin
  Final probability transform   Not required               Sigmoid

## 3.10 Initial prediction for binary logistic classification

A common optimal constant initial probability is the positive-class
proportion:

`p0 = number of positive observations / total observations`

The corresponding initial raw margin is:

`z0 = log(p0 / (1-p0))`

This is the log-odds.

If the classes are balanced, `p0=0.5`, so `z0=0`.

The log-odds is the starting raw score; it is not the probability
itself.

## 3.11 How the derivatives are used next

For every observation, XGBoost calculates `g_i` and `h_i`.

When observations are assigned to a candidate leaf, it sums them:

`G = Σ_i g_i`

`H = Σ_i h_i`

These aggregates summarize the local loss behavior of that group. The
next note derives how they determine the leaf's optimal weight and how
they are used to compare splits.
