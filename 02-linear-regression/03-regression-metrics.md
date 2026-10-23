# Regression Metrics

Regression metrics evaluate how accurately a regression model predicts continuous target values. Different metrics measure prediction errors differently, so the choice depends on the dataset and the objective of the problem.

## Notation

- \(n\): Number of observations.
- \(y_i\): Actual value of the \(i\)-th observation.
- \(\hat{y}_i\): Predicted value of the \(i\)-th observation.
- \(\bar{y}\): Mean of the actual target values.
- \(k\): Number of independent variables (predictors), excluding the intercept.
- \(\ln\): Natural logarithm.

---

## 1. Mean Absolute Error (MAE)

**Definition:** MAE calculates the average absolute difference between actual and predicted values.

### Formula

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

### Advantages

- Easy to understand and interpret.
- Expressed in the same unit as the target variable.
- Less sensitive to extreme errors than MSE and RMSE.
- Penalizes errors linearly, so larger errors do not receive a quadratic penalty.

### Disadvantages

- Does not indicate the direction of errors because absolute values remove their signs.
- Penalizes large errors less heavily than MSE and RMSE.
- Not differentiable at zero error, although it can still be optimized using suitable optimization methods.

### Interpretation

Lower MAE indicates smaller average absolute prediction errors. An MAE of 5 means the average absolute prediction error is 5 units.

---

## 2. Mean Squared Error (MSE)

**Definition:** MSE calculates the average of the squared differences between actual and predicted values.

### Formula

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

### Advantages

- Penalizes large errors more heavily because errors are squared.
- Differentiable, making it convenient for gradient-based optimization.
- Commonly used as a loss function for training regression models.

### Disadvantages

- Highly sensitive to outliers and extreme prediction errors.
- Expressed in squared units of the target variable, making direct interpretation less intuitive.
- A single large error can significantly increase the metric.

### Interpretation

Lower MSE indicates smaller squared prediction errors. An MSE of 25 means the average squared error is 25 target-units squared.

---

## 3. Root Mean Squared Error (RMSE)

**Definition:** RMSE is the square root of MSE. It measures the typical magnitude of prediction errors while penalizing large errors more heavily.

### Formula

$$
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

### Advantages

- Expressed in the same unit as the target variable.
- Penalizes large prediction errors more heavily than MAE.
- Useful when large prediction errors are particularly costly.
- Widely used for evaluating regression models.

### Disadvantages

- Sensitive to outliers because errors are squared.
- Does not indicate the direction of prediction errors.
- A few extreme errors can dominate the metric.

### Interpretation

Lower RMSE indicates smaller errors, with larger errors receiving more weight. An RMSE of 5 means the root mean squared error is 5 target units.

---

## 4. R-squared (R²)

**Definition:** R², or the coefficient of determination, compares the model's prediction errors with the total variation in the actual target values around their mean.

### Formula

$$
R^2 = 1-\frac{SS_{\text{res}}}{SS_{\text{tot}}}
$$

Where:

$$
SS_{\text{res}} = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

$$
SS_{\text{tot}} = \sum_{i=1}^{n}(y_i-\bar{y})^2
$$

### Advantages

- Provides a measure of model fit relative to predicting the mean of the actual target values.
- Unitless, allowing interpretation without depending on the target's measurement units.
- Useful for comparing models evaluated on the same dataset and target.
- Can be calculated for linear and nonlinear regression models.

### Disadvantages

- On the training data, R² cannot decrease when predictors are added to ordinary least squares regression with an intercept. It may increase even when an added predictor is not genuinely useful.
- A high R² does not guarantee good predictions on unseen data.
- Does not indicate the direction of prediction errors.
- Can be negative on evaluation data when the model performs worse than the mean-prediction baseline.
- Does not establish causation or prove that the model is appropriate.

### Interpretation

- **R² = 1:** Perfect predictions.
- **R² = 0:** The model performs as well as predicting the mean of the actual target values, under the standard evaluation definition.
- **R² < 0:** The model performs worse than that mean-prediction baseline.

For example, R² = 0.80 means the model accounts for 80% of the observed variation relative to the mean baseline on the evaluated dataset. It does not mean that 80% of predictions are correct.

---

## 5. Adjusted R-squared (Adjusted R²)

**Definition:** Adjusted R² modifies R² by accounting for the number of predictors used in the model and the number of observations.

### Formula

$$
R^2_{\text{adj}} =
1-(1-R^2)\frac{n-1}{n-k-1}
$$

Where:

- \(n\): Number of observations.
- \(k\): Number of predictors, excluding the intercept.
- \(R^2\): Ordinary coefficient of determination.

### Advantages

- Accounts for the number of predictors in the model.
- Penalizes unnecessary model complexity.
- Can decrease when an added predictor does not improve the model sufficiently.
- Useful when comparing regression models with different numbers of predictors on the same dataset.

### Disadvantages

- Does not guarantee good performance on unseen data.
- Does not automatically identify which predictors are useful.
- Can be misleading when model assumptions are inappropriate or when predictors are selected using the same data without proper validation.
- Its standard formula requires \(n-k-1>0\).

### Interpretation

Unlike ordinary training R², Adjusted R² can increase or decrease when a predictor is added.

For example, adding a useful predictor may increase Adjusted R², while adding an irrelevant predictor may decrease it. This is because Adjusted R² considers model complexity as well as fit.

---

## 6. Mean Absolute Percentage Error (MAPE)

**Definition:** MAPE calculates the average absolute prediction error as a percentage of the actual value.

### Formula

$$
MAPE = \frac{100}{n}\sum_{i=1}^{n}
\left|\frac{y_i-\hat{y}_i}{y_i}\right|
$$

### Advantages

- Expressed as a percentage, making it easy to communicate.
- Independent of the target's measurement units.
- Can help compare forecasting errors across datasets with different scales when percentage errors are meaningful.

### Disadvantages

- Undefined when actual values are zero.
- Can become extremely large when actual values are close to zero.
- Can be misleading when actual target values are negative.
- Can behave poorly when actual values are a mixture of positive and negative values.
- Standard MAPE treats equal absolute errors equally for a given actual value, regardless of whether the prediction is above or below the actual value.

### Interpretation

Lower MAPE generally indicates smaller average absolute percentage errors when the metric is appropriate for the data.

For example, MAPE = 10% means the mean of the individual absolute percentage errors is 10%. It does not mean every prediction is off by exactly 10%.

---

## 7. Mean Squared Logarithmic Error (MSLE)

**Definition:** MSLE calculates the mean squared difference between the logarithms of the actual and predicted values, after adding 1 to each value.

### Formula

$$
MSLE = \frac{1}{n}\sum_{i=1}^{n}
\left[\ln(1+y_i)-\ln(1+\hat{y}_i)\right]^2
$$

### Advantages

- Places greater emphasis on relative differences than on absolute differences in many situations.
- Reduces the influence of large target magnitudes compared with ordinary squared error.
- Can be useful for nonnegative targets with a wide range of magnitudes, such as sales or demand.

### Disadvantages

- Standard MSLE requires actual and predicted values to be greater than or equal to zero.
- Less intuitive to interpret than MAE or RMSE.
- May not be suitable when absolute errors matter more than relative differences.
- Large relative errors can still be penalized substantially.

### Interpretation

Lower MSLE indicates smaller squared differences between the log-transformed actual and predicted values.

The metric is particularly useful when relative differences matter more than raw differences.

---

## 8. Root Mean Squared Logarithmic Error (RMSLE)

**Definition:** RMSLE is the square root of MSLE. It measures the root mean squared difference between log-transformed actual and predicted values.

### Formula

$$
RMSLE = \sqrt{
\frac{1}{n}\sum_{i=1}^{n}
\left[\ln(1+y_i)-\ln(1+\hat{y}_i)\right]^2
}
$$

### Advantages

- Places more emphasis on relative differences than on absolute differences in many applications.
- Less dominated by large target magnitudes than RMSE.
- Can be useful for nonnegative targets spanning several orders of magnitude.

### Disadvantages

- Standard RMSLE requires nonnegative actual and predicted values.
- Does not have the same direct interpretation as MAE or RMSE in the original target units.
- May not be appropriate when absolute errors are the main concern.

### Interpretation

Lower RMSLE indicates smaller log-scale prediction errors. A value of zero indicates perfect predictions for all observations.

---

## 9. Median Absolute Error (MedAE)

**Definition:** MedAE calculates the median of the absolute differences between actual and predicted values.

### Formula

$$
MedAE = \operatorname{median}
\left(|y_i-\hat{y}_i|\right)
$$

### Advantages

- Robust to extreme prediction errors compared with MAE and RMSE.
- Easy to interpret in the target's original unit.
- Useful when the typical error is more important than the average error.

### Disadvantages

- Can hide a small number of very large errors.
- Does not represent the average error.
- Does not indicate the direction of prediction errors.

### Interpretation

Lower MedAE indicates a smaller median absolute prediction error. For an odd number of observations, it is the middle value after sorting the absolute errors.

For an even number of observations, the median is typically the average of the two middle sorted values.

---

## 10. Explained Variance Score

**Definition:** Explained Variance Score measures how much of the variation in the actual target values is accounted for by the predictions, focusing on the variance of the prediction errors.

### Formula

$$
EVS = 1-\frac{\operatorname{Var}(y-\hat{y})}
{\operatorname{Var}(y)}
$$

Where:

- \(\operatorname{Var}(y-\hat{y})\): Variance of the prediction errors.
- \(\operatorname{Var}(y)\): Variance of the actual target values.

This standard formula assumes the variance of the actual target values is nonzero.

### Advantages

- Provides a unitless measure of how much variation in the target is explained.
- Useful for assessing the variation remaining in prediction errors.
- Can complement R² when evaluating regression models.

### Disadvantages

- Does not directly measure absolute prediction error.
- Does not fully capture systematic prediction bias.
- May be misleading if interpreted as equivalent to R² in every situation.

### Interpretation

An Explained Variance Score of 1 indicates that the prediction errors have zero variance. However, this does not necessarily mean predictions are perfect: all predictions could have the same constant offset.

For ordinary evaluation, a higher score is generally better.

---

## Quick Comparison

| Metric | Measures | Sensitive to large errors? | Original target units? |
|---|---|---|---|
| MAE | Average absolute error | Moderately | Yes |
| MSE | Average squared error | High | No |
| RMSE | Root mean squared error | High | Yes |
| R² | Fit relative to mean baseline | Based on squared errors | No |
| Adjusted R² | Fit adjusted for predictor count | Based on R² | No |
| MAPE | Average absolute percentage error | Depends on actual values | Percentage |
| MSLE | Mean squared log error | Emphasizes relative differences | No |
| RMSLE | Root mean squared log error | Emphasizes relative differences | No |
| MedAE | Median absolute error | Relatively low | Yes |
| Explained Variance | Variance remaining in errors | Depends on error variance | No |

## Important Notes

1. **Lower is generally better:** MAE, MSE, RMSE, MAPE, MSLE, RMSLE and MedAE are error metrics for which lower values generally indicate better predictions. R², Adjusted R² and Explained Variance are generally interpreted as better when higher.
2. **Compare models fairly:** Evaluate models on the same held-out dataset and use the same target scale.
3. **Choose metrics based on the problem:** Use MAE when average absolute error matters, RMSE when large errors deserve more weight, MAPE when percentage errors are meaningful, and MedAE when a robust measure of typical error is desired.
4. **Check the target domain:** MAPE is problematic with zero or near-zero actual values, while standard MSLE and RMSLE require nonnegative values.
5. **Metrics are complementary:** No single metric captures every aspect of regression performance. Use multiple appropriate metrics when needed.
