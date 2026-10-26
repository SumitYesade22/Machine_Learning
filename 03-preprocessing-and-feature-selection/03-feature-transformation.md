# Numerical Feature Transformations

## 1. Why Do We Use Transformations?

Numerical transformations are used to:

- Reduce skewness in numerical features.
- Compress large values and reduce the influence of extreme values.
- Make nonlinear relationships more linear.
- Sometimes stabilize variance when error variance increases with the magnitude of the target.
- Improve model performance when the original feature representation is unsuitable.

**Important:** Normally distributed input features are not a mandatory requirement for most ML algorithms. Apply transformations when they are useful, not just to make the data look normal.

## 2. Types of Transformations

### 1. Log Transformation

**Suitable for:** Strongly right-skewed, positive numerical data, such as income, sales, and house prices.

**Purpose:** Compresses large values more than small values, potentially reducing right skewness.

**Formula:**

`y = log(x)`

For non-negative data containing zeros:

`y = log(1 + x)`

**Limitations:**
- `log(x)` requires `x > 0`.
- `log(1 + x)` requires `x > -1`.
- May not improve every distribution or model.

### 2. Reciprocal Transformation

**Suitable for:** Positive numerical features when an inverse relationship is meaningful or appropriate.

**Purpose:** Maps large positive values to small values and can substantially change the shape of a distribution.

**Formula:**

`y = 1 / x`

**Limitations:**
- Undefined when `x = 0`.
- Small positive values can become extremely large.
- May amplify noise and is not a universal solution for skewness.

### 3. Square Root Transformation

**Suitable for:** Non-negative data with mild to moderate right skewness, particularly count data.

**Purpose:** Compresses large values less aggressively than a logarithm in many common cases.

**Formula:**

`y = √x`

**Example:** Number of complaints, visits, or events.

**Limitations:**
- Standard square root is not real-valued for negative inputs.
- May not sufficiently reduce strong skewness.

### 4. Square Transformation

**Suitable for:** Situations where a quadratic relationship is meaningful or when deliberately creating a nonlinear feature.

**Purpose:** Gives larger magnitudes more influence and can help represent curvature in a relationship.

**Formula:**

`y = x²`

**Limitations:**
- Usually increases the influence of large magnitudes.
- Can worsen right skewness.
- Is not generally used to reduce right skewness.

### 5. Box-Cox Transformation

**Suitable for:** Strictly positive numerical data with skewness that may benefit from a power transformation.

**Purpose:** Estimates a power parameter to find a suitable transformation of the data.

**Formula:**

`y = (x^λ − 1) / λ`, when `λ ≠ 0`

`y = log(x)`, when `λ = 0`

**Limitations:**
- Requires strictly positive inputs: `x > 0`.
- Does not guarantee a normal distribution.
- The estimated parameter must be learned from training data.

### 6. Yeo-Johnson Transformation

**Suitable for:** Skewed numerical data that may contain positive values, zeros, and negative values.

**Purpose:** Estimates a power parameter, similarly to Box-Cox, while supporting zero and negative inputs.

**Formula:** Uses a piecewise power transformation for positive and non-positive values.

**Limitations:**
- Does not guarantee normality or constant error variance.
- Its usefulness should be evaluated on the actual data and model.

## 3. Which Transformation for Which Distribution?

| Data pattern | Possible choice |
|---|---|
| Strong right skew, positive values | Log or Box-Cox |
| Mild to moderate right skew | Square root |
| Skewed data containing zero or negative values | Yeo-Johnson |
| Meaningful inverse relationship | Reciprocal |
| Meaningful quadratic relationship | Square |
| Left-skewed data | Evaluate suitable power transformations or domain-specific alternatives; no universal choice |

These are starting points, not strict rules. Inspect the data and validate the result.

## 4. Apply Transformations to X or y?

Transformations can be applied to either input features `X` or the target `y`, depending on the problem.

### Input Features (X)

Transform a feature when:
- It is strongly skewed and compression may help.
- Its relationship with the target is nonlinear and a transformation may improve linearity.
- Domain knowledge supports the transformation.

**Example:** Apply `log(Area)` if a log-transformed relationship between area and price is appropriate.

### Target Variable (y)

Transform the target when:
- The target is strongly skewed.
- A multiplicative or nonlinear relationship is more appropriate.
- Error variance increases with the target magnitude and a transformation may stabilize it.

**Example:** Predict `log(Price)` instead of `Price` when a log-scale model is appropriate.

After predicting on the transformed target scale, apply the inverse transformation to obtain predictions in the original units. For log targets, simply exponentiating the prediction does not generally produce an unbiased estimate of the original-scale mean.

## 5. Relationship with Linear Regression Assumptions

Transformations may help with some regression problems, but they do not automatically satisfy every assumption.

- **Linearity:** Transforming `X` or `y` may make the relationship more linear.
- **Homoscedasticity:** Transforming the target may help when error variance changes with the target magnitude.
- **Normality:** Classical inference concerns the conditional distribution of errors, not the requirement that every input feature be normally distributed.

Check residual plots and validation performance after fitting the model.

## 6. Which Algorithms Benefit?

- **Linear Regression:** Transformations can help with nonlinear relationships or heteroscedasticity.
- **Logistic Regression:** May help with skewed features or nonlinear feature relationships.
- **KNN and SVM:** Transformations may help, but feature scaling is also important.
- **Decision Trees, Random Forest, and XGBoost:** Usually do not require normally distributed features; transformations are often unnecessary.
- **Gaussian Naive Bayes:** Assumes Gaussian class-conditional distributions for continuous features; transformations may help in some cases but do not guarantee this assumption.

## 7. Important Implementation Rules

- Learn transformation parameters using training data only.
- Apply the same fitted transformation to validation and test data.
- Use a pipeline to avoid inconsistent preprocessing and data leakage.
- Compare validation performance before and after transformation.
- Do not transform every numerical feature blindly.

**Key Takeaway:** Choose a transformation based on the feature or target distribution, the relationship being modeled, input restrictions, and validation results. Normality is not a universal requirement for ML features.
