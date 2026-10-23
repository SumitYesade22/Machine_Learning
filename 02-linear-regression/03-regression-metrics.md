# Regression Metrics

Regression metrics evaluate how well a regression model predicts continuous numerical values. They help measure prediction errors and assess model performance.

## Notation

- \(n\): Number of observations
- \(y_i\): Actual value
- \(\hat{y}_i\): Predicted value
- \(\bar{y}\): Mean of actual target values
- \(k\): Number of independent variables (features), excluding the intercept

---

# 1. MAE (Mean Absolute Error)

## Definition

MAE measures the average absolute difference between actual and predicted values.

### Formula

\[
MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
\]

### Advantages

- Easy to understand and interpret.
- Expressed in the same unit as the target variable.
- Less sensitive to outliers than MSE because errors are not squared.
- Penalizes errors linearly, so each additional unit of error adds a constant amount.

### Disadvantages

- Less sensitive to large errors than MSE because errors are not squared.
- Not differentiable at zero, which can complicate some gradient-based optimization methods.
- Does not indicate the direction of errors because absolute values are used.

### Interpretation

Lower MAE indicates smaller average absolute prediction errors. An MAE of 5 means the average absolute error is 5 units of the target variable.

---

# 2. MSE (Mean Squared Error)

## Definition

MSE measures the average squared difference between actual and predicted values.

### Formula

\[
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

### Advantages

- Penalizes large errors more heavily than small errors.
- Smooth and differentiable, making it suitable for gradient-based optimization.
- Commonly used as a training loss for regression models.
- Minimizing the sum of squared errors is the basis of Ordinary Least Squares (OLS).

### Disadvantages

- Highly sensitive to outliers because errors are squared.
- Expressed in squared units of the target variable, making direct interpretation less intuitive.
- A few large errors can dominate the overall metric.

### Interpretation

Lower MSE indicates smaller average squared errors. An MSE of 25 means the average squared error is 25 target-units squared; it does not mean the average prediction error is 25 units.

---

# 3. RMSE (Root Mean Squared Error)

## Definition

RMSE is the square root of MSE. It measures the typical magnitude of prediction errors while penalizing large errors more heavily.

### Formula

\[
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Or:

\[
RMSE=\sqrt{MSE}
\]

### Advantages

- Expressed in the same unit as the target variable.
- Penalizes large errors more heavily than MAE.
- Widely used to evaluate regression models.

### Disadvantages

- Sensitive to outliers because it is based on squared errors.
- A few large errors can substantially increase the metric.
- Less robust to extreme observations than MAE.

### Interpretation

Lower RMSE generally indicates better predictive performance on the same evaluation dataset. RMSE should be interpreted in the context of the target variable's scale.

---

# 4. R² Score (Coefficient of Determination)

## Definition

R² compares the model's squared prediction errors with the total variation of the actual target values around their mean.

### Formula

\[
R^2=1-\frac{SS_{res}}{SS_{tot}}
\]

Where:

\[
SS_{res}=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

\[
SS_{tot}=\sum_{i=1}^{n}(y_i-\bar{y})^2
\]

- \(SS_{res}\): Residual sum of squares.
- \(SS_{tot}\): Total sum of squares.
- \(\bar{y}\): Mean of actual target values.

### Interpretation

- **R² = 1:** Perfect predictions on the evaluated dataset.
- **R² = 0:** The model's squared error equals the error obtained by predicting the target mean, under the standard definition.
- **R² < 0:** The model performs worse than predicting the target mean on that dataset.

A higher R² generally indicates a better fit to the evaluated data, but it does not guarantee better performance on unseen data.

### Advantages

- Unitless and independent of the target's measurement units.
- Provides a useful measure of model fit.
- Can help compare models evaluated on the same target and observations.

### Disadvantages

- In ordinary least-squares regression with an intercept, training R² cannot decrease when additional features are added; irrelevant features may therefore make the model appear better.
- Does not, by itself, reveal systematic overprediction or underprediction.
- A high R² does not guarantee good predictions on unseen data.
- Does not verify regression assumptions or establish causation.
- Can be negative on test data or for models whose predictions perform worse than the mean baseline.

**Important:** R² can also be calculated for nonlinear models. It is not restricted to linear regression, although its interpretation depends on the evaluation setup.

---

# 5. Adjusted R²

## Definition

Adjusted R² modifies R² by accounting for the number of predictors relative to the number of observations. It helps discourage adding features that do not sufficiently improve model fit.

### Formula

\[
R^2_{adj}=1-(1-R^2)\frac{n-1}{n-k-1}
\]

Where:

- \(n\): Number of observations.
- \(k\): Number of independent variables, excluding the intercept.
- \(R^2\): Ordinary R² score.

### Advantages

- Accounts for the number of predictors.
- Can decrease when a newly added feature does not improve the model sufficiently.
- Useful when comparing regression models with different numbers of predictors, fitted on the same target and observations.

### Disadvantages

- Does not guarantee better test-set performance.
- Does not automatically identify the most meaningful or causal predictors.
- Can be misleading when model assumptions are inappropriate or the sample is small relative to the number of predictors.

### Interpretation

Adjusted R² can increase or decrease when a feature is added. It increases only when the improvement in fit is sufficient to compensate for the additional predictor under the standard formula.

---

# 6. MAPE (Mean Absolute Percentage Error)

## Definition

MAPE measures the average absolute prediction error as a percentage of the actual value.

### Formula

\[
MAPE=\frac{100}{n}\sum_{i=1}^{n}
\left|\frac{y_i-\hat{y}_i}{y_i}\right|
\]

### Advantages

- Expressed as a percentage, which is often easy to communicate.
- Scale-independent for nonzero actual values.
- Useful when relative errors matter more than absolute errors.

### Disadvantages

- Undefined when an actual value is zero.
- Can become extremely large when actual values are close to zero.
- Can behave poorly when actual values are very small.
- Is asymmetric in its treatment of overprediction and underprediction.
- Not appropriate for every dataset, especially when targets include zero or negative values.

### Interpretation

A MAPE of 10% means the average absolute percentage error, calculated observation by observation, is 10%.

**Note:** MAPE is not necessarily a reliable measure when actual values can be zero or close to zero.

---

# 7. MSLE (Mean Squared Logarithmic Error)

## Definition

MSLE measures the mean squared difference between the logarithms of actual and predicted values, using a log transformation of \(1+y\).

### Formula

\[
MSLE=\frac{1}{n}\sum_{i=1}^{n}
\left[\ln(1+y_i)-\ln(1+\hat{y}_i)\right]^2
\]

### Advantages

- Reduces the influence of absolute differences at larger scales.
- Can be useful when relative differences matter and targets vary over several orders of magnitude.
- Penalizes underprediction and overprediction differently in terms of their original-scale differences.

### Disadvantages

- Standard MSLE requires actual and predicted values to be greater than or equal to zero.
- Less intuitive to interpret than MAE or RMSE.
- May not be appropriate when absolute errors are the main concern.

### Interpretation

Lower MSLE indicates smaller squared differences between the log-transformed actual and predicted values.

---

# 8. RMSLE (Root Mean Squared Logarithmic Error)

## Definition

RMSLE is the square root of MSLE.

### Formula

\[
RMSLE=\sqrt{MSLE}
\]

### Advantages

- Square root makes the metric easier to relate to the logarithmic scale than MSLE.
- Useful for targets with a wide range of positive values when relative differences matter.
- Reduces the dominance of large absolute errors compared with RMSE in many cases.

### Disadvantages

- Requires nonnegative actual and predicted values under the standard definition.
- Remains less intuitive than MAE or RMSE.
- May not reflect business costs when absolute errors matter more than relative errors.

### Interpretation

Lower RMSLE generally indicates better predictions on the logarithmic scale.

---

# 9. Median Absolute Error

## Definition

Median Absolute Error is the median of the absolute differences between actual and predicted values.

### Formula

\[
MedAE=\operatorname{median}
\left(|y_i-\hat{y}_i|\right)
\]

### Advantages

- Robust to extreme prediction errors compared with MAE and RMSE.
- Easy to interpret in the target's original unit.
- Useful when the typical error is more important than the average error.

### Disadvantages

- Can hide a small number of very large errors.
- Does not represent the average error.
- Does not indicate the direction of prediction errors.

### Interpretation

A lower Median Absolute Error indicates a smaller median absolute prediction error.

---

# 10. Explained Variance Score

## Definition

Explained Variance Score measures how much of the variance in the target is accounted for by the model's predictions, using the variance of the residuals.

### Formula

\[
EVS=1-\frac{\operatorname{Var}(y-\hat{y})}
{\operatorname{Var}(y)}
\]

This standard form assumes the variance of the actual target is nonzero.

### Advantages

- Unitless.
- Useful for assessing how well predictions capture variation in the target.
- Can complement error-based metrics.

### Disadvantages

- Does not necessarily penalize systematic prediction offsets.
- A high score does not guarantee small absolute prediction errors.
- Can be misleading when interpreted without other metrics.

### Interpretation

- **EVS = 1:** Residual variance is zero.
- **EVS = 0:** Residual variance equals the variance of the actual target.
- **EVS < 0:** Residual variance exceeds the variance of the actual target.

**Important:** Explained Variance Score and R² are related but not identical. For example, a model whose predictions differ from actual values by a constant offset can have an Explained Variance Score of 1 but an R² below 1.

---

# Comparison of Regression Metrics

| Metric | Measures | Better value | Sensitive to outliers? |
|---|---|---|---|
| MAE | Average absolute error | Lower | Less sensitive |
| MSE | Average squared error | Lower | Highly sensitive |
| RMSE | Root of average squared error | Lower | Highly sensitive |
| R² | Fit relative to the mean baseline | Usually higher | Can be influenced strongly |
| Adjusted R² | Fit adjusted for predictor count | Usually higher | Can be influenced strongly |
| MAPE | Average absolute percentage error | Lower | Can be unstable near zero |
| MSLE | Average squared log error | Lower | Less dominated by large absolute scales |
| RMSLE | Root of squared log error | Lower | Less dominated by large absolute scales |
| Median Absolute Error | Median absolute error | Lower | Relatively robust |
| Explained Variance | Variance captured relative to residual variance | Usually higher | Depends on residual distribution |

## Final Notes

- MAE, MSE, and RMSE measure prediction errors.
- R² and Adjusted R² evaluate model fit, with Adjusted R² accounting for the number of predictors.
- MAPE measures relative errors but has problems when actual values are zero or close to zero.
- MSLE and RMSLE are useful for some nonnegative targets where relative differences matter.
- Median Absolute Error is robust to extreme errors.
- Explained Variance measures residual variance but does not fully capture systematic prediction offsets.

No single metric is best for every problem. Select metrics based on the target variable, the cost of prediction errors, and the intended use of the model.