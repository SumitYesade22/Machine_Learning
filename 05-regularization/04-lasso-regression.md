# Lasso Regression

## 1. What is Lasso Regression?

Lasso Regression is a regularized regression algorithm that minimizes prediction error while penalizing large model coefficients.

Lasso stands for **Least Absolute Shrinkage and Selection Operator**.

It is based on Linear Regression, but adds an L1 penalty to the loss function.

The main objectives of Lasso are:

- Minimize prediction error.
- Reduce model complexity.
- Reduce overfitting when appropriate.
- Shrink coefficients toward zero.
- Perform feature selection by making some coefficients exactly zero.

Lasso balances prediction accuracy and model complexity.

---

## 2. Why do we need Lasso?

Consider a Linear Regression model:

`y = w1*x1 + w2*x2 + w3*x3 + b`

Suppose we have 100 features, but only a few of them are useful for predicting the target.

Ordinary Linear Regression may assign coefficients to all 100 features.

Some of these coefficients may represent noise or weak relationships.

This can make the model unnecessarily complex and increase the risk of overfitting.

Lasso addresses this problem by penalizing the coefficients.

It can reduce the coefficients of less useful features to exactly zero.

When a coefficient becomes zero, that feature no longer contributes to the prediction.

This is called **feature selection**.

---

## 3. Lasso Loss Function

Ordinary Linear Regression minimizes Mean Squared Error (MSE).

`MSE = (1/n) * Σ(y_i - y_pred_i)^2`

Lasso adds an L1 penalty:

`Lasso Loss = MSE + λ * Σ|w_j|`

Where:

- `y_i` = actual target value
- `y_pred_i` = predicted target value
- `w_j` = coefficient of feature j
- `λ` = regularization parameter
- `n` = number of observations

The L1 penalty is:

`L1 Penalty = λ * Σ|w_j|`

### What does lambda do?

Lambda controls the strength of regularization.

| Lambda | Effect |
|---|---|
| `λ = 0` | No regularization; equivalent to ordinary Linear Regression |
| Small lambda | Small penalty; coefficients shrink slightly |
| Larger lambda | Stronger penalty; coefficients shrink more |
| Very large lambda | Coefficients may become zero, potentially causing underfitting |

As lambda increases, the model is encouraged to use smaller coefficients.

However, increasing lambda does not guarantee better performance on unseen data.

---

## 4. Why does Lasso use absolute values?

Lasso uses:

`λ * Σ|w_j|`

Instead of:

`λ * Σw_j²`

The absolute value penalty is called L1 regularization.

For a single coefficient `m`:

`Penalty = λ|m|`

The absolute value function is:

`|m| = m` when `m > 0`

`|m| = -m` when `m < 0`

`|m| = 0` when `m = 0`

Its graph has a sharp corner at zero.

This sharp corner is important because it allows the optimized coefficient to become exactly zero.

---

## 5. How does Lasso make a coefficient exactly zero?

Consider a simple regression model:

`y = mx + b`

Here, `m` is the coefficient of feature `x`.

Suppose ordinary Linear Regression estimates:

`m_OLS = 2`

Lasso tries to minimize:

`Total Loss = MSE + λ|m|`

Two objectives are competing:

- MSE wants coefficients that produce accurate predictions.
- L1 penalty encourages coefficients to become smaller.

During optimization, Lasso searches for the coefficient that minimizes the total loss.

If a feature is useful enough, its coefficient remains nonzero.

If the feature contributes too little to reducing the prediction error, the penalty can make zero the optimal coefficient.

### Important

Lasso does not simply check whether a coefficient is small and then remove it.

It optimizes the complete objective:

`Prediction Error + Regularization Penalty`

The coefficient becomes zero when zero is the optimal solution for that objective.

---

## 6. Mathematical Derivation

### Image placeholder

![Lasso Regression Derivation](images/lasso.png)

### Understanding the three cases

The derivation considers three possible values of the coefficient:

- `m > 0`
- `m = 0`
- `m < 0`

The absolute value term behaves differently in each case.

### Case 1: m > 0

When `m > 0`:

`|m| = m`

The L1 penalty becomes a linear term.

The derivative of the loss function is set to zero to obtain the optimal positive coefficient.

The resulting coefficient must satisfy the condition:

`m > 0`

If the calculated value is zero or negative, it is not a valid solution for this case.

### Case 2: m < 0

When `m < 0`:

`|m| = -m`

The L1 penalty becomes a different linear term.

The derivative of the loss function is set to zero to obtain the optimal negative coefficient.

The resulting coefficient must satisfy the condition:

`m < 0`

If the calculated value is zero or positive, it is not a valid solution for this case.

### Case 3: m = 0

At `m = 0`, the absolute value function is not differentiable.

Therefore, the ordinary derivative cannot be used to check whether zero is the optimal coefficient.

Instead, the subgradient condition is used.

For the loss convention:

`J(m) = SSE + λ|m|`

The condition for zero to be optimal is:

`|Sxy| <= λ/2`

Where:

`Sxy = Σ(x_i - x_bar)(y_i - y_bar)`

When this condition is satisfied:

`m_Lasso = 0`

This means that the feature's contribution to reducing the prediction error is not strong enough to overcome the L1 penalty.

### Important clarification

The OLS slope formula is:

`m_OLS = Sxy / Sxx`

Where:

`Sxx = Σ(x_i - x_bar)^2`

This formula gives the coefficient estimated by ordinary Linear Regression.

It is not the Lasso coefficient when `m = 0`.

For the zero case, the correct condition is:

`|Sxy| <= λ/2`

The OLS slope is useful as an intermediate quantity, but it is not the final coefficient.

---
![Lasso](images/lasso1.png)

## 7. Soft-Thresholding

The three cases can be combined into one expression.

`m_OLS = Sxy / Sxx`

`m_Lasso = sign(m_OLS) * max(|m_OLS| - λ/(2Sxx), 0)`

This is called **soft-thresholding**.

It means:

- If the coefficient is positive, subtract a threshold.
- If the coefficient is negative, add a threshold toward zero.
- If its absolute value is below the threshold, make it exactly zero.

### Example

Suppose:

`m_OLS = 0.8`

And the threshold is:

`λ/(2Sxx) = 1`

Then:

`m_Lasso = sign(0.8) * max(0.8 - 1, 0)`

`m_Lasso = 0`

The coefficient becomes exactly zero.

If instead:

`m_OLS = 3`

Then:

`m_Lasso = sign(3) * max(3 - 1, 0)`

`m_Lasso = 2`

The coefficient remains nonzero.

### Why does the sharp corner matter?

The L1 penalty has a sharp corner at zero.

When the prediction error and penalty are combined, the minimum of the total loss can occur exactly at that corner.

This allows coefficients to become exactly zero.

However, the sharp corner alone does not guarantee that a coefficient becomes zero.

Zero must satisfy the optimality condition.

---

## 8. Lasso vs Ridge Regression

| Property | Ridge Regression | Lasso Regression |
|---|---|---|
| Regularization | L2 | L1 |
| Penalty | `λΣw_j²` | `λΣ|w_j|` |
| Shrinks coefficients | Yes | Yes |
| Coefficients exactly zero | Generally no | Yes |
| Feature selection | No automatic feature elimination | Can perform feature selection |
| Multicollinearity | Often distributes weights across correlated features | May select one or a subset of correlated features |
| Main benefit | Coefficient stabilization and variance reduction | Shrinkage and feature selection |

### Ridge

`Ridge Loss = MSE + λΣw_j²`

Ridge shrinks coefficients toward zero, but they generally remain nonzero.

### Lasso

`Lasso Loss = MSE + λΣ|w_j|`

Lasso can shrink coefficients exactly to zero.

---

## 9. Does Lasso reduce overfitting?

Lasso can reduce overfitting by reducing model complexity.

It can:

- Shrink coefficients.
- Remove weak features.
- Reduce variance.
- Improve generalization performance.

But Lasso does not guarantee that overfitting will be reduced.

If lambda is too high, the model may become too simple and underfit.

The objective is to find a suitable balance between prediction error and regularization.

---

## 10. Bias-Variance Tradeoff

Regularization generally introduces bias while reducing variance.

### Without regularization

A model may fit the training data very closely.

It can have:

- Low training error.
- High variance.
- Poor performance on unseen data.

### With Lasso

The penalty shrinks coefficients and may remove features.

This can:

- Increase bias.
- Reduce variance.
- Improve performance on unseen data.

The objective is not to minimize variance alone.

The objective is to minimize generalization error.

A simplified error decomposition is:

`Expected Test Error ≈ Bias² + Variance + Irreducible Noise`

Lasso can improve test performance when the reduction in variance is greater than the increase in bias.

---

## 11. When should we use Lasso?

Lasso is useful when:

- We have many features.
- We suspect that only a subset of features is relevant.
- We want automatic feature selection.
- We want to reduce model complexity.
- We want to control overfitting through regularization.

Lasso may not be ideal when:

- Many correlated features are equally important.
- We need to retain all relevant features.
- Strong regularization causes underfitting.

For highly correlated features, Lasso may select one feature and set the coefficients of others to zero. The selected feature can change across different training samples.

---

## 12. Important Practical Points

### Feature scaling

Lasso is sensitive to feature scale.

If one feature has values ranging from 0 to 100000 and another ranges from 0 to 1, their coefficients may be penalized differently.

Therefore, feature scaling is generally important.

Common choices include:

- StandardScaler
- MinMaxScaler

### Choosing lambda

Lambda is a hyperparameter.

We generally choose it using cross-validation.

The goal is to find a suitable regularization strength based on validation performance.

### Intercept

The intercept is generally not penalized.

The penalty is applied to feature coefficients, not the intercept.

---

## 13. Summary

Lasso Regression is Linear Regression with an L1 penalty.

`Lasso Loss = MSE + λΣ|w_j|`

Its main properties are:

- It minimizes prediction error while penalizing large coefficients.
- It shrinks coefficients toward zero.
- It can make coefficients exactly zero.
- It can perform feature selection.
- It can reduce overfitting by reducing model complexity.
- It can increase bias while reducing variance.
- Lambda controls the strength of regularization.
- A very large lambda can cause underfitting.

**Key takeaway:**

Lasso does not remove a feature simply because its coefficient is small. It optimizes prediction error plus a penalty. When zero satisfies the optimality condition, Lasso can set that feature's coefficient exactly to zero.
