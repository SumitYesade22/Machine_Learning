# Ridge Regression (L2 Regularization)

---

## 1. Introduction

Ridge Regression is a regularized version of Linear Regression that uses L2 regularization.

It adds a penalty to the loss function based on the squared magnitude of the model coefficients.

The main objectives of Ridge Regression are:

- Prevent coefficients from becoming unnecessarily large.
- Reduce model variance.
- Improve model stability.
- Reduce overfitting.
- Improve generalization performance on unseen data.

Ridge Regression is particularly useful when:

- The model has many features.
- Features are highly correlated.
- Ordinary Least Squares (OLS) produces unstable coefficients.
- The model is overfitting the training data.

### Definition

Ridge Regression minimizes the prediction error while penalizing large coefficients.

The regularization strength is controlled by the parameter `lambda`, also commonly called `alpha` in Scikit-learn.

---

## 2. Why Ridge Regression is Needed

### Ordinary Linear Regression

Ordinary Linear Regression estimates coefficients by minimizing the sum of squared prediction errors.

For a model with two features:

```text
y_hat = b0 + b1*x1 + b2*x2
```

The OLS objective is:

```text
J(b) = MSE
```

OLS focuses only on minimizing the prediction error on the training data.

If the model has many features or highly correlated features, the estimated coefficients can become large or unstable.

This can increase the risk of overfitting.

### Ridge Regression

Ridge Regression adds a penalty for large coefficients.

```text
J(b) = MSE + lambda * (b1^2 + b2^2)
```

The model now considers two things:

1. How well it predicts the target.
2. How large its coefficients are.

The objective is to find coefficients that balance prediction accuracy and coefficient shrinkage.

---

## 3. Ridge Regression Cost Function

For a model with multiple features:

```text
y_hat = b0 + b1*x1 + b2*x2 + ... + bp*xp
```

The Ridge Regression objective is:

```text
J(b) = MSE + lambda * sum(bj^2)
```

More explicitly:

```text
J(b) = (1/n) * sum((yi - y_hat_i)^2)
        + lambda * sum(bj^2)
```

Where:

- `n` = number of training observations.
- `yi` = actual target value.
- `y_hat_i` = predicted target value.
- `bj` = coefficient of feature j.
- `lambda` = regularization strength.

The intercept `b0` is generally not penalized.

### L2 Penalty

The penalty term is:

```text
L2 penalty = lambda * sum(bj^2)
```

The coefficients are squared before being added.

This is why Ridge Regression is called L2 Regularization.

### Important

Different implementations may use different scaling conventions for MSE and the penalty term.

Therefore, the numerical value of `lambda` is dependent on the convention used in the objective function.

---

## 4. How Ridge Regression Shrinks Coefficients

Consider a model with two coefficients:

```text
y_hat = b0 + b1*x1 + b2*x2
```

The Ridge penalty is:

```text
Penalty = lambda * (b1^2 + b2^2)
```

Suppose the OLS coefficients are:

```text
b1 = 10
b2 = 5
```

Ridge Regression penalizes these coefficients.

After optimization, the coefficients may become:

```text
b1 = 8.5
b2 = 4.2
```

The coefficients have moved closer to zero.

This is called coefficient shrinkage.

### Why Shrinkage Helps

Large coefficients can make predictions highly sensitive to changes in input features.

By penalizing large coefficients, Ridge Regression can reduce this sensitivity.

This can improve generalization when the original coefficients were unstable or excessively large.

### Important Clarification

Ridge Regression does not blindly decrease every coefficient in every situation.

The final coefficients depend on both:

- The data-fitting term.
- The regularization penalty.

The optimization finds the coefficients that minimize their combined objective.

---

## 5. Mathematical Derivation for Simple Linear Regression

Consider a simple Linear Regression model:

```text
y_hat = b + m*x
```

Where:

- `m` = slope.
- `b` = intercept.

For centered data, the OLS slope is:

```text
m_OLS = sum((xi - x_mean)*(yi - y_mean))
        / sum((xi - x_mean)^2)
```

For Ridge Regression, the slope becomes:

```text
m_Ridge = sum((xi - x_mean)*(yi - y_mean))
          / (sum((xi - x_mean)^2) + lambda)
```

The intercept is:

```text
b_Ridge = y_mean - m_Ridge*x_mean
```

### Interpretation

In the Ridge slope formula, lambda is added to the denominator.

As lambda increases:

```text
Denominator increases
        |
        v
Slope magnitude generally decreases
```

This produces coefficient shrinkage.

### Important Assumption

This formula uses centered data and a particular convention for the Ridge penalty.

Other definitions of the loss function may produce a different scaling of lambda.

---

## 6. Gradient Descent in Ridge Regression

Ridge Regression can also be optimized using Gradient Descent.

Consider the objective:

```text
J(w) = MSE + lambda * sum(wj^2)
```

The gradient is:

```text
Gradient = Data Gradient + 2*lambda*w
```

For one coefficient:

```text
Gradient = Data Gradient + 2*lambda*w
```

### Update Rule

```text
w_new = w_old - eta*(Data Gradient + 2*lambda*w_old)
```

Where:

- `eta` = learning rate.
- `w_old` = current coefficient.
- `Data Gradient` = gradient of the data-fitting term.

Rearranging:

```text
w_new = (1 - 2*eta*lambda)*w_old
        - eta*(Data Gradient)
```

### Where Shrinkage Happens

The regularization contribution multiplies the old weight by:

```text
(1 - 2*eta*lambda)
```

When the learning rate and lambda are chosen appropriately, this term pulls the coefficient toward zero.

### Important Clarification

The complete update includes the data gradient as well.

Therefore, a coefficient does not necessarily decrease in magnitude on every iteration.

The regularization component pulls it toward zero, while the data gradient can move it in either direction.

---

## 7. Geometrical Interpretation of Ridge Regression

Ridge Regression can be understood geometrically using the coefficient space.

Consider a model with two coefficients:

```text
b1 and b2
```

These coefficients form two axes.

### OLS Estimate

OLS minimizes the prediction error without any coefficient constraint.

Its solution is called:

```text
OLS estimate
```

The OLS solution may have large coefficients.

### Ridge Constraint

Ridge Regression can be expressed as a constrained optimization problem:

```text
Minimize RSS

Subject to:

sum(bj^2) <= t
```

For two coefficients:

```text
b1^2 + b2^2 <= t
```

This constraint forms a circular region in the coefficient space.

The Ridge solution is the point that minimizes the prediction error while satisfying the coefficient constraint.

### Why It Is Called Ridge

The OLS solution may lie outside the constrained region.

Ridge Regression finds a solution within the permitted region.

The solution often lies on the boundary of the constraint region when the constraint is active.

This constrained region is associated with the Ridge estimate.

### Geometrical Interpretation of Lambda

- Small lambda: weak constraint, solution remains closer to OLS.
- Large lambda: stronger constraint, coefficients shrink more.
- Very large lambda: coefficients approach zero.

The intercept is generally not constrained.

---

## 8. Effect of Lambda (Alpha)

Lambda controls the strength of regularization.

### Lambda = 0

```text
Ridge Regression = Ordinary Linear Regression
```

There is no regularization penalty.

### Small Lambda

The model applies a weak penalty.

Coefficients may shrink slightly.

### Large Lambda

The model applies a strong penalty.

Coefficients shrink more toward zero.

### Extremely Large Lambda

The model may become too constrained and fail to learn the relationship adequately.

This can cause underfitting.

### Summary

| Lambda | Regularization Strength | Possible Effect |
|---|---|---|
| 0 | None | OLS |
| Small | Weak | Slight coefficient shrinkage |
| Large | Strong | Greater coefficient shrinkage |
| Extremely large | Very strong | Possible underfitting |

The best lambda is generally selected using validation or cross-validation.

---

## 9. Feature Scaling in Ridge Regression

Feature scaling is important in Ridge Regression because the penalty is applied to coefficient magnitudes.

Consider two features:

```text
Feature 1: Area
Range = 500 to 5000

Feature 2: Bedrooms
Range = 1 to 5
```

The features have different scales.

A coefficient's magnitude depends partly on the scale of its corresponding feature.

For example:

```text
Price = b1*Area + b2*Bedrooms
```

The coefficient for Area may be numerically much smaller than the coefficient for Bedrooms simply because Area is measured using larger numbers.

Ridge penalizes coefficients, not feature importance directly.

Therefore, applying Ridge without scaling can produce an unfair penalty across features measured on different scales.

### Standardization

A common approach is StandardScaler.

```text
z = (x - mean(x)) / standard_deviation(x)
```

After standardization:

```text
Mean = 0
Standard Deviation = 1
```

The features are placed on comparable scales.

This makes the coefficient penalty more comparable across features.

### Important

Feature scaling should be fitted using the training data only.

The same scaler is then applied to validation and test data.

This avoids data leakage.

---

## 10. Ridge Regression and Multicollinearity

### What Is Multicollinearity?

Multicollinearity occurs when two or more input features are highly correlated.

Example:

```text
House size in square feet
House size in square meters
```

These features contain nearly the same information.

### Problem in OLS

When features are highly correlated, OLS coefficients can become unstable.

Small changes in training data may produce large changes in the estimated coefficients.

### How Ridge Helps

Ridge Regression penalizes large coefficients.

This can make the coefficient estimates more stable in the presence of multicollinearity.

### Important

Ridge does not remove correlated features.

It generally keeps all features in the model, but shrinks their coefficients.

---

## 11. Bias-Variance Tradeoff

Ridge Regression changes the bias-variance balance.

### Without Regularization

The model may fit the training data closely.

However, if the coefficients are unstable, the model may have high variance.

### With Ridge Regularization

The coefficients are constrained.

This can:

- Increase bias.
- Reduce variance.
- Improve generalization performance.

### Important

Ridge Regression does not guarantee better performance.

If regularization is too strong, the model may underfit.

The objective is to select a suitable lambda that balances bias and variance.

---

## 12. Important Properties of Ridge Regression

### 1. Coefficients Shrink Toward Zero

Ridge penalizes the squared magnitude of coefficients.

### 2. Coefficients Generally Do Not Become Exactly Zero

Unlike Lasso, Ridge generally does not perform feature selection.

### 3. Larger Coefficients Receive Larger Penalty Gradients

The derivative of the penalty term is proportional to the coefficient:

```text
Gradient of penalty = 2*lambda*w
```

Therefore, the regularization contribution is larger in magnitude for larger coefficients.

### 4. Reduces Variance

Ridge can reduce coefficient instability and improve generalization.

### 5. Can Increase Bias

Constraining coefficients may prevent the model from fitting the training data as closely.

### 6. Useful for Multicollinearity

Ridge can stabilize coefficient estimates when features are highly correlated.

### 7. Requires Hyperparameter Tuning

The regularization strength should be selected using validation data or cross-validation.

---

## 13. Ridge Regression vs Linear Regression

| Linear Regression | Ridge Regression |
|---|---|
| Minimizes prediction error. | Minimizes prediction error plus L2 penalty. |
| No regularization. | Uses regularization. |
| Coefficients may become large or unstable. | Coefficients are shrunk toward zero. |
| Can have high variance. | Can reduce variance. |
| May overfit. | Can help reduce overfitting. |
| Does not penalize coefficient magnitude. | Penalizes squared coefficient magnitude. |

---

## 15. Important Points to Remember

- Ridge Regression is Linear Regression with L2 regularization.
- It adds a penalty proportional to the sum of squared coefficients.
- Lambda or alpha controls the strength of regularization.
- Larger lambda generally produces stronger coefficient shrinkage.
- Coefficients shrink toward zero but generally do not become exactly zero.
- Ridge can reduce variance and improve stability.
- Ridge is useful when features are highly correlated.
- Feature scaling is important because Ridge penalizes coefficient magnitudes.
- Ridge may increase bias while reducing variance.
- Excessive regularization can cause underfitting.
- Ridge does not guarantee better performance than OLS.
- The best regularization strength should be selected using validation data or cross-validation.

