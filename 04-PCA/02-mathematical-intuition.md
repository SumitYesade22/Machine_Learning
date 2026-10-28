# Principal Component Analysis (PCA)

## 1. What is PCA?

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional representation while preserving as much variance as possible.

PCA creates new features called **Principal Components (PCs)**.

- PC1 captures the maximum possible variance.
- PC2 captures the maximum remaining variance, subject to being orthogonal to PC1.
- PC3 captures the maximum remaining variance, subject to being orthogonal to PC1 and PC2.
- The same process continues for the remaining components.

PCA does not select existing features. It creates new features by combining the original features.

---

## 2. Why do we need PCA?

Suppose a dataset contains multiple numerical features.

Some features may contain similar information or may be highly correlated.

For example, a dataset may contain:

- Height in centimeters
- Height in meters
- Weight
- Age

Some features may contain redundant information.

PCA can combine information from multiple features into fewer components.

### Benefits

- Reduces dimensionality.
- Reduces redundancy in correlated features.
- Helps visualize high-dimensional data.
- May reduce computational requirements for downstream models.

### Limitation

The new components are combinations of original features, so they may be difficult to interpret.

---

## 3. PCA and Variance

Variance measures how much the values of a feature spread around their mean.

For a feature X:

`Variance = Average squared deviation from the mean`

Mathematically:

`Var(X) = (1/n) * Σ(X_i - μ)²`

where:

- `X_i` = individual observation
- `μ` = mean of the feature
- `n` = number of observations

### Why does PCA use variance?

PCA assumes that directions with higher variance contain more information about the structure of the data.

Therefore, PCA searches for directions along which the data varies the most.

Important:

High variance does not always mean high predictive importance. PCA is an unsupervised technique and does not consider the target variable.

---

### Why do we Standardize Features Before PCA?

PCA identifies directions of maximum variance. However, variance is affected by the scale and units of features.

For example:
- Age: 18–60
- Salary: 20,000–1,00,000

Salary may have much larger variance than age simply because its numerical scale is larger.

If we apply PCA directly, features with larger scales may dominate the principal components.

Therefore, we often standardize features before PCA:

`X_scaled = (X - μ) / σ`

After standardization:
- Each feature has mean 0.
- Each feature has standard deviation 1.
- Features are placed on a comparable scale.

### When should we standardize?

- When features have different units or scales.
- When we want each feature to contribute on a comparable scale.
- When the original feature scales are not meaningful for determining importance.

### When may standardization not be necessary?

- When all features already have comparable scales.
- When differences in feature variance are meaningful and should be preserved.

**Important:** Standardization is not always mandatory. It depends on the dataset and the objective.

`StandardScaler` standardizes the data, whereas `PCA` in scikit-learn automatically centers the data but does not automatically standardize it.

Why Mean-Centering in PCA?
Mean-centering subtracts each feature's mean from its values, shifting the data's mean point to the origin (0, 0).

This allows PCA to find principal components that capture the maximum variance around the data's mean, rather than being influenced by its absolute position relative to the origin.

## 4. Geometrical Intuition

Imagine a dataset containing two numerical features.

The data points form an elongated cloud in a two-dimensional space.

PCA finds a direction along which the data points have maximum variance.

- PC1 is the direction of maximum variance.
- PC2 is perpendicular to PC1.
- PC2 captures the maximum remaining variance.

If PC1 captures most of the variance, we may represent the data using only PC1.

---

## 5. Centering the Data

Before applying PCA, we generally center the data by subtracting the mean of each feature.

For a feature X:

`X_centered = X - μ`

After centering:

- The mean of each feature becomes zero.
- The origin represents the mean of the original data.
- Variance and covariance are measured around the data mean.

### Why is centering needed?

PCA searches for directions of maximum variance around the mean of the data.

Centering makes the origin coincide with the data mean, which allows us to represent the data using its covariance matrix.

### Important: Centering vs Standardization

**Centering:**

`X_centered = X - μ`

**Standardization:**

`X_scaled = (X - μ) / σ`

Centering subtracts the mean.

Standardization subtracts the mean and divides by the standard deviation.

---

## 6. When is centering required?

### Scenario 1: Manually calculating variance and covariance

When we calculate variance and covariance using their individual formulas, the mean is already subtracted inside the formula.

For example:

`Var(X) = (1/n) * Σ(X_i - μ)²`

`Cov(X,Y) = (1/n) * Σ(X_i - μ_X)(Y_i - μ_Y)`

Therefore, we do not need to separately center the data before calculating these values.

The formulas already account for the means.

### Scenario 2: Calculating covariance using matrix multiplication

When using the matrix formula:

`C = (1/n) * XᵀX`

the data must be centered first.

Otherwise, `XᵀX/n` represents the second-moment matrix around the origin, not the covariance matrix around the mean.

The correct formula is:

`C = (1/n) * X_centeredᵀ X_centered`

### Scenario 3: Using scikit-learn PCA

`sklearn.decomposition.PCA` automatically centers the input data.

Therefore, manual centering is not required before calling PCA.

However, PCA does not automatically standardize each feature to unit variance.

---

## 7. Covariance Matrix

The covariance matrix represents:

- Variance of each feature.
- Covariance between pairs of features.

For two features X₁ and X₂:

`C = [[Var(X₁), Cov(X₁,X₂)],`
`     [Cov(X₂,X₁), Var(X₂)]]`

### Interpretation

The diagonal elements represent the variance of individual features.

The off-diagonal elements represent covariance between features.

- Positive covariance: features tend to increase or decrease together.
- Negative covariance: one feature tends to increase when the other decreases.
- Zero covariance: no linear relationship is indicated.

Zero covariance does not necessarily imply independence.

### Matrix formula

For a centered data matrix X:

`C = (1/n) * XᵀX`

Here:

- Rows represent observations.
- Columns represent features.
- `n` represents the number of observations.

---

## 8. Projection onto a Direction

Suppose we have a centered data point represented by a vector:

`x_i`

We want to project it onto a direction represented by a unit vector:

`u`

The projected value is:

`z_i = uᵀx_i`

Here:

- `x_i` represents the original data point.
- `u` represents the direction.
- `z_i` represents the projected value along that direction.

After projection, the data is represented along a single direction.

PCA searches for the direction that maximizes the variance of these projected values.

---

## 9. Projected Variance

The projected data is:

`z_i = uᵀx_i`

Since the original data is centered, the mean of the projected data is also zero.

Therefore:

`Var(z) = (1/n) * Σz_i²`

Substituting the projection equation:

`Var(z) = (1/n) * Σ(uᵀx_i)²`

This can be written as:

`Var(z) = uᵀ C u`

where C is the covariance matrix.

Therefore:

**The variance of the data projected onto direction u is given by:**

`Var(z) = uᵀ C u`

---

## 10. PCA Optimization Equation

PCA wants to find the direction along which the projected data has maximum variance.

Therefore, the optimization problem is:

`Maximize: uᵀ C u`

Subject to:

`uᵀu = 1`

The unit-vector constraint is necessary because otherwise we could increase the projected variance simply by increasing the magnitude of u.

The optimization equation is:

`u* = argmax (uᵀ C u)`

subject to:

`uᵀu = 1`

This is the mathematical formulation of the first principal component.

---

## 11. Short Derivation: Why do Eigenvectors and Eigenvalues appear?

We want to maximize:

`uᵀ C u`

subject to:

`uᵀu = 1` this keeps the vector length to 1

Using the Lagrange multiplier method:

`L(u, λ) = uᵀ C u - λ(uᵀu - 1)`

Taking the derivative with respect to u:

`∂L/∂u = 2Cu - 2λu`

Setting the derivative equal to zero:

`2Cu - 2λu = 0`

Therefore:

`Cu = λu`

This is the eigenvalue-eigenvector equation of the covariance matrix.

### What does this mean?

The optimization problem leads to the eigenvectors and eigenvalues of the covariance matrix.

- Eigenvector u represents a direction.
- Eigenvalue λ represents the variance along that direction.

For a unit eigenvector:

`uᵀ C u = λ`

Therefore, the eigenvalue tells us the variance captured along its corresponding eigenvector.

---

## 12. How do we get the Principal Components?

After calculating the covariance matrix, we perform eigen decomposition:

`C u_i = λ_i u_i`

We obtain multiple eigenvalues and their corresponding eigenvectors.

We sort the eigenvalues in descending order:

`λ₁ ≥ λ₂ ≥ λ₃ ≥ ... ≥ λ_p`

The corresponding eigenvectors represent the principal component directions.

| Component | Meaning |
|---|---|
| PC1 | Eigenvector corresponding to the largest eigenvalue |
| PC2 | Eigenvector corresponding to the second-largest eigenvalue |
| PC3 | Eigenvector corresponding to the third-largest eigenvalue |

### Why is PC1 the eigenvector with the largest eigenvalue?

Because the eigenvalue represents the variance along its corresponding eigenvector.

The largest eigenvalue corresponds to the direction with maximum variance.

Therefore:

**PC1 = Eigenvector corresponding to the largest eigenvalue.**

---

## 13. Subsequent Principal Components

The first principal component captures the maximum variance.

The second principal component captures the maximum remaining variance, subject to being orthogonal to PC1.

The third principal component captures the maximum remaining variance, subject to being orthogonal to PC1 and PC2.

For a covariance matrix, the eigenvectors corresponding to distinct eigenvalues are orthogonal.

This ensures that the principal components represent different, uncorrelated directions of variation.

---

## 14. Explained Variance Ratio

Each eigenvalue represents the variance captured by its corresponding principal component.

The explained variance ratio is:

`EVR_i = λ_i / Σλ_j`

For example:

| Component | Explained Variance |
|---|---:|
| PC1 | 70% |
| PC2 | 20% |
| PC3 | 10% |

If we retain PC1 and PC2:

`Total variance retained = 70% + 20% = 90%`

Therefore, the two components preserve 90% of the total variance.

### Cumulative Explained Variance

Cumulative explained variance is the sum of explained variance ratios up to a particular component.

It is used to determine how many components should be retained.

For example, we may retain the smallest number of components that explain 95% of the total variance.

---

## 15. PCA Workflow

1. Select numerical features.
2. Handle missing values.
3. Standardize features if they have different scales or units.
4. Center the data.
5. Calculate the covariance matrix.
6. Perform eigen decomposition.
7. Sort eigenvalues in descending order.
8. Select the required principal components.
9. Project the data onto the selected components.

---

## 16. PCA vs Feature Selection

| PCA | Feature Selection |
|---|---|
| Creates new features | Selects existing features |
| Components are combinations of original features | Features retain their original meaning |
| May reduce interpretability | Usually more interpretable |
| Uses variance to find directions | Uses criteria to select relevant features |

---

![mathematical intuition](C:\Users\SUMIT\Desktop\Machine_learning\images\pca.png)

### One-line conclusion

**PCA maximizes projected variance. This optimization leads to the eigenvalue-eigenvector equation of the covariance matrix, where eigenvectors represent principal component directions and eigenvalues represent the variance captured along those directions.**

# PCA: Inverse Transformation, Reconstruction Error, and Information Loss

## 1. Inverse Transformation

Inverse transformation is used to reconstruct the original data from its PCA-transformed representation.

During PCA, the original data is projected onto a selected number of principal components, reducing its dimensionality.

The inverse transformation maps the PCA-transformed data back to the original feature space.

### Formula

`X_reconstructed = Z @ W.T + μ`

Where:
- `Z` = PCA-transformed data
- `W` = selected principal components (eigenvectors)
- `W.T` = transpose of the components matrix
- `μ` = mean of the original data
- `X_reconstructed` = reconstructed data

### Example

Suppose the original dataset has 3 features:

`X.shape = (100, 3)`

After applying PCA with 2 components:

`Z.shape = (100, 2)`

Inverse transformation reconstructs the data in the original 3-dimensional feature space:

`X_reconstructed.shape = (100, 3)`

However, the reconstructed values may differ from the original values because one principal component was discarded.

### Python

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

X_reconstructed = pca.inverse_transform(X_pca)
```

---

## 2. Reconstruction Error

Reconstruction error measures how much the reconstructed data differs from the original data.

When PCA reduces dimensionality, some information may be lost. Reconstruction error quantifies this difference.

### Formula

`Reconstruction Error = ||X - X_reconstructed||²`

A common metric is Mean Squared Reconstruction Error (MSE):

`MSE = (1/n) * Σ (xᵢ - x̂ᵢ)²`

Where:
- `xᵢ` = original data point
- `x̂ᵢ` = reconstructed data point
- `n` = number of data points

### Interpretation

- Lower reconstruction error → reconstructed data is closer to the original data.
- Higher reconstruction error → greater difference between original and reconstructed data.

### Python

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(X, X_reconstructed)
```

---

## 3. Information Loss in PCA

Information loss occurs when PCA discards principal components during dimensionality reduction.

Each principal component captures a certain amount of variance in the data.

When only a subset of principal components is retained, the variance represented by the discarded components is lost in the reduced representation.

### Explained Variance Ratio (EVR)

`EVRᵢ = λᵢ / Σλⱼ`

Where:
- `λᵢ` = eigenvalue of the i-th principal component
- `Σλⱼ` = total variance across all principal components

### Retained Information

`Retained Information = Σ EVR of selected components`

### Information Loss

`Information Loss = 1 - Retained Information`

Or, in percentage:

`Information Loss (%) = (1 - Retained Information) × 100`

### Example

Suppose PCA produces these explained variance ratios:

| Component | Explained Variance Ratio |
|---|---:|
| PC1 | 0.60 |
| PC2 | 0.30 |
| PC3 | 0.10 |

If we retain only PC1 and PC2:

`Retained Information = 0.60 + 0.30 = 0.90`

`Information Loss = 1 - 0.90 = 0.10`

Therefore, 90% of the total variance is retained, and 10% is discarded.

---

## Relationship Between These Concepts

1. PCA reduces dimensionality by retaining selected principal components.
2. Inverse transformation reconstructs data in the original feature space.
3. Reconstruction error measures the difference between original and reconstructed data.
4. Information loss represents the variance discarded by PCA.

### Important Points

- If all principal components are retained, PCA can reconstruct the original data exactly, up to numerical precision.
- If fewer components are retained, reconstruction is generally approximate.
- Retaining more components generally reduces reconstruction error and information loss.
- PCA information loss refers to discarded variance, not necessarily all types of information that may be important for a particular task.

# PCA: Limitations and Sensitivity to Outliers

## 1. Limitations of PCA

PCA is a dimensionality reduction technique that transforms the original features into a smaller set of principal components while retaining as much variance as possible.

However, PCA has some limitations:

### 1.1 Linear Transformation

PCA captures linear relationships between features.

It may not effectively represent complex nonlinear relationships in the data.

### 1.2 Loss of Interpretability

Principal components are linear combinations of the original features.

Therefore, the transformed features may be harder to interpret than the original features.

### 1.3 Information Loss

When fewer principal components are retained, some variance is discarded.

This may result in loss of information from the original dataset.

### 1.4 High Variance Does Not Mean High Predictive Importance

PCA selects components based on variance, not on their relationship with the target variable.

A feature with low variance may still be important for predicting the target.

### 1.5 Sensitive to Feature Scaling

PCA is affected by the scale of the features.

Features with larger numerical ranges may dominate the principal components.

Standardization is commonly applied when features have different scales and comparable contributions are desired.

---

## 2. Sensitivity to Outliers

PCA is sensitive to outliers because it relies on variance and covariance.

Variance uses squared deviations from the mean:

`Var(X) = Σ(xᵢ - μ)² / n`

Due to squaring, observations that are far from the mean can contribute disproportionately to the variance.

### Effect of Outliers on PCA

Outliers can:

- Increase the variance of a feature.
- Change the covariance matrix.
- Shift the principal component directions.
- Cause PCA to prioritize directions influenced by outliers rather than the main data distribution.

### Example

Suppose most observations are concentrated around:

`X = [10, 11, 12, 13, 14]`

But one outlier is present:

`X = [10, 11, 12, 13, 14, 100]`

The outlier contributes a large squared deviation from the mean.

As a result, the calculated variance increases substantially, potentially affecting the PCA directions.

---

## 3. Handling Outliers Before PCA

Before applying PCA:

1. Investigate extreme observations.
2. Determine whether they are genuine observations or data errors.
3. Apply an appropriate treatment if justified by the data and analysis objective.

Possible approaches include:

- Correcting data-entry or measurement errors.
- Removing invalid observations.
- Applying suitable transformations.
- Using robust scaling or robust PCA methods when appropriate.

**Important:** Genuine outliers should not be removed automatically. Their presence may represent meaningful information.

---

## Key Takeaway

PCA is a linear, variance-based dimensionality reduction technique. It can lose information, reduce interpretability, and be strongly affected by outliers because it relies on variance and covariance.