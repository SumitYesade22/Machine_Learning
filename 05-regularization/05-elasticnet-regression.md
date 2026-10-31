# Elastic Net Regression

## 1. Introduction

Elastic Net Regression is a regularized linear regression algorithm that combines L1 (Lasso) and L2 (Ridge) regularization.

It is used to:

- Reduce overfitting by penalizing large coefficients.
- Perform feature selection by allowing coefficients to become exactly zero.
- Improve coefficient stability, especially when features are highly correlated.
- Handle datasets containing many features.

Elastic Net combines the benefits of Lasso and Ridge Regression.

---

## 2. Why Elastic Net?

Consider a dataset with several features:

- House area
- Carpet area
- Built-up area
- Number of rooms

Some features may be highly correlated and provide similar information.

In such cases:

- Ordinary Least Squares (OLS) may produce unstable coefficient estimates.
- Lasso can select one feature and set other correlated feature coefficients to zero, but its selection can be unstable.
- Ridge shrinks coefficients and helps reduce instability, but does not perform feature selection.

Elastic Net combines L1 and L2 regularization to address these limitations.

---

## 3. Objective Function Derivation

### Step 1: Ordinary Linear Regression

The linear regression prediction equation is:

`y_hat_i = w0 + w1*x_i1 + w2*x_i2 + ... + wp*x_ip`

The prediction error for an observation is:

`e_i = y_i - y_hat_i`

To measure the total prediction error, we use Sum of Squared Errors (SSE):

`SSE = Σ(y_i - y_hat_i)²`

The OLS objective function is:

`J(w,b) = SSE`

The goal is to minimize SSE.

### Step 2: Ridge Regression

Ridge Regression adds an L2 penalty to the SSE objective:

`J(w,b) = SSE + λΣw_j²`

The L2 penalty discourages large coefficient magnitudes.

### Step 3: Lasso Regression

Lasso Regression adds an L1 penalty:

`J(w,b) = SSE + λΣ|w_j|`

The L1 penalty encourages sparsity and can set coefficients exactly to zero.

### Step 4: Elastic Net

Elastic Net combines the L1 and L2 penalties:

`J(w,b) = SSE + λ₁Σ|w_j| + λ₂Σw_j²`

This objective includes:

- SSE for prediction error.
- L1 penalty for sparsity and feature selection.
- L2 penalty for coefficient shrinkage.

### Image Placeholder

![Elastic Net objective function derivation](images/elastic_net.png)

Replace the image path with the location of your derivation image in the GitHub repository.

---

## 4. Hyperparameters

### Lambda (λ)

Lambda controls the overall regularization strength.

- Small lambda: weaker regularization.
- Large lambda: stronger regularization.
- Lambda = 0: no regularization.

Increasing lambda generally shrinks coefficients more strongly.

### Alpha (α)

In the mathematical formulation, alpha controls the balance between L1 and L2 penalties.

`J(w) = MSE + λ[αΣ|w_j| + (1-α)Σw_j²]`

| Alpha | Result |
|---|---|
| α = 1 | Lasso |
| α = 0 | Ridge |
| 0 < α < 1 | Elastic Net |

### Scikit-learn Parameter Names

Scikit-learn uses:

- `alpha`: Overall regularization strength.
- `l1_ratio`: Balance between L1 and L2 penalties.

| l1_ratio | Result |
|---|---|
| 1 | Lasso |
| 0 | Ridge |
| Between 0 and 1 | Elastic Net |

---

## 5. Coefficient Behavior

Elastic Net can shrink coefficients and set some coefficients exactly to zero.

For example:

| Feature | OLS coefficient | Elastic Net coefficient |
|---|---:|---:|
| Area | 25 | 18 |
| Bedrooms | 10 | 0 |
| Location | 30 | 22 |
| Age | -15 | -8 |
| Balcony | 2 | 0 |

Here, the coefficients of Bedrooms and Balcony have become zero.

This means the model does not use these features to make predictions.

This behavior is called **feature selection**.

However, Elastic Net does not guarantee that every unnecessary feature will be removed. The result depends on the dataset and hyperparameters.

---

## 6. Elastic Net and Multicollinearity

Multicollinearity occurs when two or more features are highly correlated.

For example:

- House area
- Carpet area

These features may contain overlapping information.

### Why OLS can become unstable

When features are highly correlated, several different combinations of coefficients can produce similar predictions.

Small changes in the dataset can lead to large changes in the estimated coefficients.

### Ridge Behavior

Ridge adds an L2 penalty:

`λΣw_j²`

This discourages large coefficient magnitudes.

Ridge does not eliminate multicollinearity, but it can reduce the instability of coefficient estimates.

### Lasso Behavior

Lasso adds an L1 penalty:

`λΣ|w_j|`

Lasso may select one correlated feature and set another to zero.

However, the selected feature may change when the data changes slightly.

### Elastic Net Behavior

Elastic Net combines both penalties.

- L1 can set coefficients to zero.
- L2 shrinks coefficients and can improve stability among correlated features.

Therefore, Elastic Net can be useful when feature selection is required and the dataset contains highly correlated features.

It does not guarantee that all correlated features will be retained or that every coefficient will be stable.

---

## 7. Ridge vs Lasso vs Elastic Net

| Property | Ridge | Lasso | Elastic Net |
|---|---|---|---|
| Regularization | L2 | L1 | L1 + L2 |
| Shrinks coefficients | Yes | Yes | Yes |
| Coefficients exactly zero | Generally no | Yes | Yes |
| Feature selection | No | Yes | Yes |
| Handles correlated features | Helps stabilize coefficients | May select one feature | Combines shrinkage and feature selection |
| Hyperparameters | Lambda | Lambda | Lambda and alpha balance |

---

## 8. Advantages

1. Combines the benefits of Ridge and Lasso.
2. Can perform feature selection.
3. Helps reduce overfitting.
4. Can improve coefficient stability when features are highly correlated.
5. Useful for high-dimensional datasets.

---

## 9. Limitations

1. Requires tuning regularization parameters.
2. Features should generally be scaled before applying regularization.
3. Large regularization strength can cause underfitting.
4. Does not automatically learn nonlinear relationships.
5. Feature selection results depend on data and hyperparameters.

---

## 10. When to Use Elastic Net?

Elastic Net is particularly useful when:

- The dataset contains many features.
- Some features are highly correlated.
- Feature selection is required.
- Overfitting needs to be controlled.
- Lasso's feature selection may be unstable because of correlated predictors.

If all features are useful and only coefficient shrinkage is required, Ridge may be more suitable.

If feature selection is needed and predictors are not strongly correlated, Lasso may also be suitable.

---

## 11. Interview Answer

Elastic Net is a regularized linear regression algorithm that combines L1 and L2 penalties.

The L1 component encourages sparsity and can perform feature selection by setting some coefficients exactly to zero. The L2 component shrinks coefficient magnitudes and can help stabilize estimates when features are highly correlated.

Elastic Net is particularly useful when a dataset contains many correlated features and feature selection is also required.

---

## 12. Key Takeaways

- Elastic Net combines Ridge and Lasso.
- L1 penalty can set coefficients exactly to zero.
- L2 penalty shrinks coefficient magnitudes.
- Elastic Net can help with multicollinearity-related coefficient instability.
- It does not guarantee better performance than Ridge or Lasso in every dataset.
- Hyperparameter tuning is necessary to select suitable regularization strength and L1/L2 balance.