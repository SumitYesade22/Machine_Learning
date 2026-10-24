## 1. Linearity Assumption

The linearity assumption states that the expected value of the target variable has a linear relationship with the input feature(s), according to the specified regression model.

### How to Check Linearity

**Simple Linear Regression:**
- Plot the input feature against the target using a scatter plot.
- Check whether the relationship follows an approximately straight-line trend.

**Multiple Linear Regression:**
- Plot each input feature against the target individually for an initial visual check.
- Individual plots are not sufficient because they do not account for the effects of other features.
- Check residual plots against fitted values to identify possible nonlinear patterns in the combined model.
- Use partial residual plots when needed to investigate individual feature effects while accounting for other features.

### What If Linearity Is Violated?

- **Polynomial features:** Add squared or higher-order terms and check whether they improve the model.
- **Transformations:** Transform features or the target when appropriate.
- **Splines:** Use flexible curves to capture nonlinear patterns.
- **Cross-validation:** Compare the performance of linear and nonlinear models on unseen data.

### Important Points

- Actual data points do not need to lie exactly on a straight line.
- Monotonic does not necessarily mean linear; a relationship can increase consistently but still be curved.
- Polynomial Regression can model curved relationships while remaining linear in its coefficients.
- Individual feature-target plots are useful, but the combined model must also be assessed.
- Linearity and multicollinearity are different concepts; multicollinearity concerns strong linear relationships among input features.

## 2. Independence of Errors / Observations

### 1. What is the Assumption?

The errors (residuals) of different observations should be independent of each other.

- **Residual** = Actual value − Predicted value.
- One observation's error should not systematically depend on another observation's error.

**Example:** In daily sales prediction, if the model's errors are positive for several consecutive days and then negative for several consecutive days, this may indicate autocorrelation.

### 2. Why is It Important?

If errors are correlated, the model can still produce predictions and coefficient estimates, but the **usual standard errors may become unreliable**. Consequently, confidence intervals and p-values from regression coefficient t-tests may also be unreliable.

Correlated errors do not automatically mean that the coefficients are biased.

### 3. How to Detect It?

- **Residuals vs. Time/Observation Order:** Look for patterns in residuals rather than random variation.
- **Durbin–Watson Test:** Checks for first-order autocorrelation. A value around 2 suggests little first-order autocorrelation.
- **ACF Plot:** Helps identify autocorrelation across different time lags.

### 4. Connection with the t-test

In multiple linear regression:

\[
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \epsilon
\]

A **t-test** checks whether an individual regression coefficient is statistically different from zero, while keeping the other included features constant.

**Example:** Suppose we want to check whether the number of bedrooms significantly affects house prices after accounting for area.

- **Null hypothesis (\(H_0\)):** \(\beta_2 = 0\) — The number of bedrooms has no independent linear effect on the predicted price, holding area constant.
- **Alternative hypothesis (\(H_1\)):** \(\beta_2 \neq 0\) — The coefficient for bedrooms is nonzero, holding area constant.

**Interpretation:**
- If the p-value is less than the significance level (e.g., 0.05), reject the null hypothesis.
- If the p-value is greater than or equal to 0.05, fail to reject the null hypothesis.

**Important:** Statistical significance does not establish causation, and a nonsignificant coefficient does not prove that the feature has no relationship with the target.

### 5. Simple vs. Multiple Linear Regression

- **Simple Linear Regression:** The same independence assumption applies to the errors across observations.
- **Multiple Linear Regression:** The assumption still applies to errors across observations. Adding more features does not automatically resolve correlated errors.

### Key Takeaways

- Independence concerns errors across observations, not relationships between input features.
- Multicollinearity (relationships among input features) is a separate issue.
- Correlated errors can make conventional statistical inference unreliable.
- For time-series or repeated measurements, the data structure must be considered when choosing the model and standard errors.

## 3. Homoscedasticity (Constant Variance of Errors)

### 1. What is the Assumption?
The homoscedasticity assumption states that the error variance remains constant across observations, given the independent variables.

\[
\operatorname{Var}(\epsilon_i \mid X_i)=\sigma^2
\]

Where:
- \(\epsilon_i\) represents the error term for observation \(i\).
- \(X_i\) represents the independent variables for observation \(i\).
- \(\sigma^2\) represents the constant error variance.

If the error variance changes across observations, the assumption is violated. This is called **heteroscedasticity**.

### 2. Example

Suppose we are predicting house prices.

- Low-priced houses have prediction errors mostly within ±₹2 lakh.
- High-priced houses have prediction errors ranging from ±₹30 lakh.

If the error spread systematically increases with predicted price, it may indicate **heteroscedasticity**.

- **Homoscedasticity:** Error variance remains approximately constant.
- **Heteroscedasticity:** Error variance changes across observations or predictor levels.

### 3. Why Is It Important?

If heteroscedasticity exists:

- OLS coefficient estimates are not automatically biased, assuming the other relevant conditions hold.
- Conventional standard errors may be incorrectly estimated.
- Consequently, coefficient t-tests, p-values, and confidence intervals may become unreliable.

### 4. How to Detect It?

**A. Residuals vs. Fitted Plot**

Plot residuals against predicted values.

- Random scatter with approximately constant spread → consistent with homoscedasticity.
- Funnel-shaped pattern or systematically changing spread → possible heteroscedasticity.

**B. Breusch–Pagan Test**

A formal statistical test for heteroscedasticity.

- \(H_0\): Error variance is constant.
- \(H_1\): Error variance is related to the predictors used in the test.

Interpretation at significance level \(\alpha=0.05\):

- p-value < 0.05 → Reject \(H_0\); evidence of heteroscedasticity.
- p-value ≥ 0.05 → Fail to reject \(H_0\); insufficient evidence of heteroscedasticity.


### 6. How to Handle Heteroscedasticity?

- **Robust Standard Errors (HC3):** Adjust standard error estimation to account for heteroscedasticity.
- **Transformations:** Log transformation of the target may help when the variance increases with the target's scale.
- **Weighted Least Squares (WLS):** Useful when the variance structure can be modeled appropriately.

### Key Takeaways

- Homoscedasticity concerns the spread or variance of errors, not their independence.
- Independence concerns whether errors across observations are related.
- Both assumptions can affect conventional standard errors and statistical inference.
- A residual plot provides a visual diagnostic; the Breusch–Pagan test provides a formal test.
- Heteroscedasticity does not automatically mean that the regression model or its coefficient estimates are invalid.

## 4. Normality of Errors (Residuals)

### 1. What is the Assumption?

The normality assumption states that the errors are normally distributed around zero, conditional on the predictors.

$$
\epsilon \mid X \sim N(0,\sigma^2)
$$

Here:
- $\epsilon$ = Error term
- $X$ = Predictor variables
- $\sigma^2$ = Error variance

### 2. Example

Suppose we are predicting house prices.

$$
\text{Residual}=\text{Actual Price}-\text{Predicted Price}
$$

If the residuals approximately follow a bell-shaped distribution centered around zero, they are consistent with the normality assumption.

**Important:** The assumption concerns the conditional distribution of errors, not the normal distribution of the target variable or predictors.

### 3. Why Is It Important?

- OLS coefficient estimates are not automatically biased just because errors are non-normal.
- Normality supports exact small-sample inference using conventional t-tests, F-tests, and confidence intervals, under the other classical regression assumptions.
- With sufficiently large samples, approximate inference may still be valid under suitable conditions.

### 4. How to Detect It?

**A. Histogram of Residuals**

- An approximately bell-shaped distribution supports normality.
- Strong skewness or heavy tails may indicate departures from normality.

**B. Q-Q Plot**

- Points approximately following the reference line support normality.
- Systematic deviations from the line indicate possible non-normality.

**C. Shapiro–Wilk Test**

Hypotheses:

- $H_0$: The errors follow a normal distribution.
- $H_1$: The errors do not follow a normal distribution.

At significance level $\alpha=0.05$:

- p-value < 0.05 → Reject $H_0$; evidence against normality.
- p-value ≥ 0.05 → Fail to reject $H_0$; insufficient evidence against normality.

### 5. Key Takeaways

- Normality concerns the errors' conditional distribution, not the predictors or target variable individually.
- Normality and homoscedasticity are separate assumptions.
- Normality is particularly relevant for exact small-sample statistical inference.
- A Q-Q plot is a useful visual diagnostic.
- Non-normal errors do not automatically invalidate OLS coefficient estimates.

## 5. Multicollinearity

### Definition
Multicollinearity occurs when two or more independent features in a regression model are highly linearly related to each other.

**Example:** In house price prediction, `Area` and `Number_of_Rooms` may be strongly correlated because larger houses generally have more rooms.

### Why Is It a Problem?
When features contain overlapping information, it becomes difficult to estimate their individual contributions to the target variable.

- Regression coefficients can become unstable.
- Small changes in the dataset may cause large changes in coefficient estimates.
- Standard errors of coefficients may increase.
- Individual coefficient t-tests may show statistical insignificance even when the model is useful.
- Predictions can still be reasonably accurate, despite unstable coefficients.

**Important:** Multicollinearity does not automatically make OLS coefficients biased. Its main effects are reduced precision and difficulty interpreting individual coefficients.

### Perfect vs. High Multicollinearity

- **Perfect multicollinearity:** One feature is an exact linear combination of other features. OLS cannot uniquely estimate all individual coefficients.
- **High multicollinearity:** Features are strongly linearly related but not perfectly related. Coefficients can be estimated, but they may be unstable and imprecise.

### Detection Using VIF

VIF stands for **Variance Inflation Factor**. It measures how much the estimated variance of a regression coefficient is inflated due to the linear relationship between that feature and the other independent features.

Formula:

\[
VIF_j = \frac{1}{1 - R_j^2}
\]

Where:

- \(VIF_j\) = VIF of feature \(X_j\).
- \(R_j^2\) = R-squared obtained by regressing \(X_j\) on all the other independent features.

**Important:** This is a separate regression for each feature. The original target variable is not used to calculate VIF.

#### Interpretation of VIF

| VIF | Interpretation |
|---:|---|
| 1 | No linear relationship with the other features in the auxiliary regression |
| 1–5 | Generally considered acceptable |
| Above 5 | Potential multicollinearity concern |
| Above 10 | Often considered high multicollinearity |

These are rules of thumb, not strict universal thresholds.

#### Example

Suppose \(R_j^2=0.8\).

\[
VIF=\frac{1}{1-0.8}=5
\]

The coefficient estimator's variance is inflated by a factor of 5 relative to the corresponding variance without linear predictability from the other features, under the standard OLS setup.

Since standard error is the square root of variance, its inflation factor is \(\sqrt{5}\approx2.24\), not 5.

### How to Handle Multicollinearity

1. **Remove a redundant feature:** Consider dropping one of two highly correlated features when it is not essential.
2. **Combine related features:** Create a meaningful composite feature when appropriate.
3. **Use regularization:** Ridge regression can help stabilize coefficients.
4. **Keep the features:** If prediction is the main goal and performance is satisfactory, high VIF alone does not require dropping features.

Always consider domain knowledge, the objective of the analysis, and model performance before removing features.

### Key Takeaways

- Multicollinearity concerns relationships among independent features, not simply their relationships with the target.
- High VIF indicates greater variance inflation due to linear predictability from other features.
- High VIF can make coefficient estimates less precise and harder to interpret.
- Multicollinearity is particularly important when interpreting individual coefficients and conducting inference.
- The essential OLS requirement is **no perfect multicollinearity**; high multicollinearity is not always a reason to reject a model.
- Statistical significance, predictive accuracy, and coefficient interpretability are different concepts.

## 6. Zero Conditional Mean (Exogeneity)

### What Does It Mean?
The zero conditional mean assumption states that the expected value of the error term, given the independent variables, must be zero.

$$
E(\varepsilon \mid X) = 0
$$

**Purpose:** It helps ensure that OLS coefficient estimates are unbiased, provided the other required conditions are satisfied.

### What Happens If It Is Violated?
- OLS coefficients may become biased and inconsistent.
- The estimated effect of a feature may incorrectly include the effect of omitted variables.
- Using robust standard errors alone cannot fix this bias.

**Example:** While predicting exam marks using study hours, suppose prior knowledge also affects marks but is excluded from the model and is correlated with study hours. The estimated coefficient of study hours may be biased.

**Note:** Residual plots are diagnostic tools, not definitive tests of exogeneity.