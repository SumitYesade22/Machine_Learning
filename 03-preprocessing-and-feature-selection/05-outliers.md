# Outlier Detection and Treatment

## 1. Outlier Fundamentals

### 1.1 What is an Outlier?

An outlier is an observation that differs significantly from other observations in a dataset.

Example:

```python
salary = [25000, 30000, 28000, 35000, 32000, 500000]
```

Here, `500000` is a potential outlier because it is significantly higher than the other salary values.

However, an outlier is not necessarily an incorrect observation. It may represent a genuine extreme value or a data quality issue.

### 1.2 Outlier vs. Noise

**Outlier:** An observation that is unusually extreme compared to other observations.

**Noise:** Unwanted randomness or errors in data that do not represent the underlying pattern accurately.

Example:

```python
salary = [25000, 30000, 35000, 40000, 500000]
```

If `500000` is the actual salary of a senior executive, it is a genuine outlier.

If the actual salary was `50000` but `500000` was entered accidentally, it is a data error.

**Important:**
- Not every outlier is noise.
- Not every noise observation is necessarily an outlier.

### 1.3 Outlier vs. Anomaly

An outlier is an observation that is statistically unusual compared to the dataset.

An anomaly is an observation that is unusual compared to expected behavior or context.

These terms are related but are not always interchangeable.

Anomaly Detection is a separate topic from Outlier Detection.

### 1.4 Causes of Outliers

Outliers may occur due to:

#### 1. Data Entry Errors

Incorrect values entered manually.

Example:

```python
age = [22, 25, 28, 30, 250]
```

Here, `250` may be an incorrect age.

#### 2. Measurement Errors

Incorrect values recorded due to faulty measuring instruments or processes.

Example: A temperature sensor records `400°C` when the actual temperature is `40°C`.

#### 3. Sampling Errors

Incorrect or unsuitable observations included in the dataset.

Example: A commercial property accidentally included in a dataset intended only for residential properties.

#### 4. Genuine Extreme Values

Correct observations that are significantly different from the majority of the data.

Example: A CEO's salary being significantly higher than other employees' salaries.

#### 5. Natural Variability

Natural differences in real-world data.

Example: Luxury property prices being significantly higher than ordinary property prices.

### 1.5 Genuine Outliers vs. Data Errors

Before treating an outlier, investigate whether it is genuine or incorrect.

Consider:

```python
salary = [25000, 30000, 35000, 40000, 1000000]
```

The value `1000000` may represent:

- A genuine salary of a senior executive.
- A data entry error.
- An incorrect observation from another population.

The appropriate treatment depends on the data source and business context.

### 1.6 Univariate vs. Multivariate Outliers

#### Univariate Outlier

An observation that is unusual based on a single feature.

Example:

| Employee | Salary |
|---|---:|
| A | 25000 |
| B | 30000 |
| C | 35000 |
| D | 40000 |
| E | 500000 |

Here, `500000` is a potential outlier based on the Salary feature.

Common methods:
- Box Plot
- IQR
- Z-Score

#### Multivariate Outlier

An observation that is unusual when multiple features and their relationships are considered together.

Example:

| Employee | Age | Salary |
|---|---:|---:|
| A | 25 | 30000 |
| B | 28 | 35000 |
| C | 30 | 40000 |
| D | 55 | 35000 |

Employee D may be unusual considering the relationship between Age and Salary, even if neither feature is individually extreme.

**Key Difference:**

| Univariate | Multivariate |
|---|---|
| Considers one feature | Considers multiple features |
| Analyzes individual feature distribution | Analyzes relationships between features |
| Simpler to interpret | More complex to interpret |

### 1.7 Global vs. Local Outliers

#### Global Outlier

An observation that is unusual compared to the entire dataset.

Example:

```python
salary = [25000, 30000, 35000, 40000, 500000]
```

Here, `500000` is a potential global outlier.

#### Local Outlier

An observation that is unusual compared to its nearby observations or local group.

Example:

In a dataset containing property prices from different locations, a property may have a normal price globally but an unusual price compared to properties in its own location.

**Key Difference:**

| Global Outlier | Local Outlier |
|---|---|
| Compared with the entire dataset | Compared with its local neighborhood |
| Overall distribution is considered | Local group is considered |

### 1.8 When Should We Avoid Removing Outliers?

Outliers should not be removed blindly.

Removing genuine outliers may:

- Cause information loss.
- Distort the actual data distribution.
- Reduce representation of important observations.
- Introduce bias into the dataset.
- Affect model performance.

Example:

In fraud detection, unusually large transactions may be important observations.

Removing them could cause the model to lose information about potential fraudulent behavior.

---

## 2. Outlier Detection — Visualization and Statistical Concepts

### 2.1 Box Plot

A box plot represents the distribution of numerical data using quartiles.

- Q1 = 25th percentile
- Q2 = Median (50th percentile)
- Q3 = 75th percentile
- IQR = Q3 - Q1

The IQR method identifies potential outliers using:

```text
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

Observations outside these bounds are considered potential outliers.

**Important:** A box plot identifies potential outliers. It does not prove that they are incorrect values.

### 2.2 Histogram

A histogram represents the frequency distribution of a numerical feature.

It helps understand:

- Distribution shape.
- Symmetry or skewness.
- Spread of values.
- Potential extreme values.
- Whether the data has multiple peaks.

A histogram can provide clues about potential outliers, but it does not independently confirm them.

### 2.3 KDE Plot

Kernel Density Estimation (KDE) provides a smooth estimate of the probability density distribution.

It helps visualize:

- Distribution shape.
- Peaks and valleys.
- Spread.
- Tail behavior.

KDE is useful for understanding the distribution but is not a direct outlier detection method.

### 2.4 Scatter Plot

A scatter plot represents the relationship between two numerical features.

Example:

- X-axis: Age
- Y-axis: Salary

It helps identify:

- Isolated observations.
- Unusual feature relationships.
- Potential multivariate outliers.

### 2.5 Q-Q Plot

A Quantile-Quantile (Q-Q) plot compares the quantiles of sample data with the quantiles of a theoretical distribution, commonly the normal distribution.

If the points approximately follow a straight line, the sample distribution may be close to the theoretical distribution.

Deviations from the line, especially in the tails, may indicate differences in tail behavior or potential extreme observations.

---

## 3. Skewness and Kurtosis

### 3.1 Skewness

Skewness measures the asymmetry of a distribution.

#### Positive Skewness (Right-Skewed)

- Right tail is longer.
- Some unusually large values may be present.
- Mean is generally greater than median.

Examples:
- Salary.
- House prices.

#### Negative Skewness (Left-Skewed)

- Left tail is longer.
- Some unusually small values may be present.
- Mean is generally less than median.

Example:
- Marks in an easy exam where most students score high and a few score low.

#### Zero Skewness

A skewness value near zero indicates an approximately symmetric distribution.

| Skewness | Interpretation |
|---|---|
| Near 0 | Approximately symmetric |
| Positive | Right-skewed |
| Negative | Left-skewed |
| Large positive | Strong right skew |
| Large negative | Strong left skew |

**Important:** Skewness does not confirm the presence of outliers. It describes distribution asymmetry.

### 3.2 Kurtosis

Kurtosis describes the tail behavior of a distribution relative to a normal distribution.

Pandas:

```python
df["salary"].kurt()
```

Pandas `.kurt()` calculates excess kurtosis by default.

#### Mesokurtic

- Excess kurtosis = 0
- Tail behavior equivalent to a normal distribution.

#### Leptokurtic

- Excess kurtosis > 0
- Heavier tails than a normal distribution.
- Potentially higher occurrence of extreme values.

#### Platykurtic

- Excess kurtosis < 0
- Lighter tails than a normal distribution.
- Potentially lower occurrence of extreme values.

| Excess Kurtosis | Type | Interpretation |
|---:|---|---|
| 0 | Mesokurtic | Normal-equivalent tail behavior |
| Positive | Leptokurtic | Heavier tails |
| Negative | Platykurtic | Lighter tails |

**Important:** High kurtosis does not automatically mean outliers are present. Kurtosis summarizes tail behavior but does not identify or confirm individual outliers.

### 3.3 Skewness vs. Kurtosis

| Skewness | Kurtosis |
|---|---|
| Measures distribution asymmetry | Describes tail behavior |
| Indicates direction of skew | Indicates relative tail heaviness |
| Positive or negative | Positive, negative, or zero excess kurtosis |

---

## 4. Key Takeaways

- An outlier is not necessarily an error.
- Genuine outliers should not be removed blindly.
- Univariate outliers are identified using individual features.
- Multivariate outliers are identified using multiple features and their relationships.
- Global outliers are unusual compared with the entire dataset.
- Local outliers are unusual compared with their local neighborhood.
- Visualization helps understand the distribution and potential extreme values.
- Skewness describes asymmetry.
- Kurtosis describes tail behavior.
- Outlier detection and outlier treatment are different processes.


# Statistical Detection Methods

# 1. IQR Method (Interquartile Range)

The IQR method is a statistical technique used to identify potential outliers in numerical data. It is based on the spread of the middle 50% of observations and is less sensitive to extreme values than methods based on the mean and standard deviation.

## Key Terms

- **Q1 (First Quartile):** The 25th percentile of the data.
- **Q3 (Third Quartile):** The 75th percentile of the data.
- **IQR (Interquartile Range):** The difference between Q3 and Q1.

## Formula

`IQR = Q3 - Q1`

`Lower Bound = Q1 - 1.5 * IQR`

`Upper Bound = Q3 + 1.5 * IQR`

Observations below the lower bound or above the upper bound are flagged as potential outliers.

## When to Use

- Suitable for numerical data.
- Useful for identifying potential outliers in both symmetric and skewed distributions.
- Commonly used with box plots for exploratory data analysis.
- Does not require the data to follow a normal distribution.

## Limitations

- May flag genuine extreme observations as potential outliers.
- The 1.5 × IQR threshold is a rule of thumb, not a universal rule.
- Identifying an outlier does not automatically mean it should be removed or modified.

**Note:** The IQR method is used for outlier detection. Decisions about removing, capping, or transforming outliers belong to the outlier treatment stage.

## 2. Z-Score Method

The Z-Score method is a statistical technique used to identify potential outliers by measuring how many standard deviations an observation is away from the mean.

### Formula

`Z-Score = (x - Mean) / Standard Deviation`

- `x`: Individual observation
- `Mean`: Mean of the data
- `Standard Deviation`: Spread of the data

### Outlier Threshold

A common rule of thumb:

`|Z-Score| > 3`

Observations beyond this threshold are flagged as potential outliers.

### When to Use

- Suitable for approximately normal or symmetric distributions.
- Useful when the data is not strongly skewed.
- Commonly used in statistical analysis.

### Limitations

- Sensitive to extreme values because it uses mean and standard deviation.
- Generally not preferred for highly skewed or heavy-tailed data.

---

## 3. Modified Z-Score Method (MAD)

The Modified Z-Score method is a robust statistical technique used to identify potential outliers using the median and Median Absolute Deviation (MAD).

### MAD (Median Absolute Deviation)

`MAD = Median(|x - Median(x)|)`

It measures the typical absolute distance of observations from the median.

### Formula

`Modified Z-Score = 0.6745 * (x - Median) / MAD`

### Outlier Threshold

A common rule of thumb:

`|Modified Z-Score| > 3.5`

Observations beyond this threshold are flagged as potential outliers.

### When to Use

- Suitable for data containing extreme values.
- Useful for skewed or heavy-tailed distributions.
- Less sensitive to extreme observations than the standard Z-Score method.

### Limitations

- The threshold is a rule of thumb, not a universal rule.
- May still flag genuine extreme observations.
- Requires care when MAD is zero.

**Note:** Both methods identify potential outliers. Removing or modifying flagged observations is part of outlier treatment, not detection.

## 1. IQR Method

The Interquartile Range (IQR) method identifies potential outliers using the spread of the middle 50% of data.

**Formulas:**
- `IQR = Q3 - Q1`
- `Lower Bound = Q1 - 1.5 * IQR`
- `Upper Bound = Q3 + 1.5 * IQR`

Values outside these bounds are flagged as potential outliers.

**Use:** Suitable for skewed data and less sensitive to extreme values than the Z-Score method.

## 2. Tukey’s Fences

Tukey’s Fences is an extension of the IQR method that classifies potential outliers into mild and extreme outliers.

**Formulas:**
- `Inner Lower Fence = Q1 - 1.5 * IQR`
- `Inner Upper Fence = Q3 + 1.5 * IQR`
- `Outer Lower Fence = Q1 - 3 * IQR`
- `Outer Upper Fence = Q3 + 3 * IQR`

Values outside the inner fences are potential mild outliers, while values outside the outer fences are potential extreme outliers.

**Note:** Tukey’s Fences uses the same inner bounds as the IQR method; it adds outer fences to distinguish extreme outliers.

## Grubbs’ Test

Grubbs’ Test is a formal statistical test used to detect a **single potential outlier** in approximately normally distributed data.

It uses a standardized deviation similar to the Z-Score method, but compares the calculated statistic against a critical value based on sample size and significance level.

- `H0`: No outlier is present.
- `H1`: One outlier is present.

**Formula:**

`G = max(|xi - mean|) / standard_deviation`

If `G` exceeds the critical value, the most extreme observation is flagged as a potential outlier.

**Limitations:**
- Designed to detect one outlier at a time.
- Assumes approximately normal data.
- Multiple outliers may mask each other.


**Note:** Grubbs’ Test is a statistical test, whereas the Z-Score method generally flags observations using a predefined threshold such as `|Z| > 3`.

## Generalized ESD Test

Generalized Extreme Studentized Deviate (ESD) Test is a statistical test used to detect **multiple potential outliers** in approximately normally distributed data.

Unlike Grubbs’ Test, which is designed to detect a single outlier, Generalized ESD can detect up to `r` potential outliers.

**Hypotheses:**
- `H0`: No outliers are present.
- `H1`: Up to `r` outliers are present.

**Concept:**
- Iteratively identifies the most extreme observation.
- Calculates a test statistic at each iteration.
- Compares the statistic with a critical value.
- Determines the number of statistically significant outliers.

**Limitation:** Assumes approximately normal data and requires specifying the maximum number of outliers (`r`) to test.

## Mahalanobis Distance

Mahalanobis Distance is a multivariate outlier detection method that measures how far an observation is from the multivariate mean while considering feature variances and correlations.

**Formula:**

`D² = (x - μ)ᵀ Σ⁻¹ (x - μ)`

Where:
- `x` = observation vector
- `μ` = mean vector
- `Σ⁻¹` = inverse covariance matrix

Observations with unusually large Mahalanobis distances are flagged as potential outliers.

**Use Cases:**
- Detecting outliers in multivariate numerical data.
- Identifying unusual combinations of features.
- Useful when features are correlated.

**Limitations:**
- Sensitive to outliers in mean and covariance estimates.
- Assumes approximately multivariate normal data when using chi-square thresholds.
- Requires numerical features and a reliable covariance matrix.

## Outlier Treatment Techniques

### 1. Trimming
Trimming removes observations identified as outliers from the dataset.

- **Use when:** Outliers are invalid or irrelevant, and data loss is acceptable.
- **Trade-off:** Reduces dataset size and may remove genuine information.

### 2. Capping / Winsorization
Capping replaces extreme values with predefined lower or upper bounds without removing rows.

- **Use when:** Observations should be retained, but the influence of extreme values needs to be reduced.
- **Trade-off:** Changes original values and may alter the distribution.

### 3. Mean Imputation
Mean imputation replaces an identified outlier with the mean of the feature.

- **Use when:** The outlier is invalid, the correct value is unknown, and the feature is approximately symmetric.
- **Trade-off:** Mean is sensitive to extreme values and can be distorted.

### 4. Median Imputation
Median imputation replaces an identified outlier with the median of the feature.

- **Use when:** The outlier is invalid, the correct value is unknown, and the feature is skewed or contains extreme values.
- **Trade-off:** Can reduce variation and create repeated values.

### 5. Quantile-Based Capping
- **Use when:** The dataset contains extreme values that may negatively affect analysis or model performance, but removing observations is undesirable.
- **Approach:** Caps values below or above selected quantile thresholds (e.g., P1 and P99).
- **Trade-off:** Retains all observations and limits extreme values, but changes genuine extreme values and may lose information about their magnitude.

### 6. Outlier Flagging
- **Use when:** Outliers are genuine observations, and their unusual nature may provide useful information for the model.
- **Approach:** Creates a binary indicator feature (1 = outlier, 0 = non-outlier) while retaining the original values.
- **Trade-off:** Preserves original data and allows the model to learn from outlier status, but does not reduce the influence of extreme values by itself.
