# Feature Scaling in Machine Learning

## 1. Why Do We Need Feature Scaling?

Feature scaling is a preprocessing technique used to bring numerical features to comparable scales.

Consider a dataset containing:

- Age: 18–60
- Salary: 20,000–200,000
- Experience: 0–30 years

These features have different numerical ranges. Some machine learning algorithms are sensitive to these differences.

### Why Can Larger-Scale Features Dominate?

Distance-based algorithms calculate distances using differences between feature values.

Euclidean distance formula:

`Distance = sqrt((x1 - y1)^2 + (x2 - y2)^2 + ... + (xp - yp)^2)`

Suppose two people have these differences:

- Age difference = 5
- Salary difference = 50,000

Squared differences:

- Age: `5^2 = 25`
- Salary: `50000^2 = 2,500,000,000`

The salary difference dominates the distance numerically. Consequently, KNN may behave as though salary is much more influential than age.

Feature scaling helps prevent features from dominating merely because they have larger numerical ranges.

**Important:** A larger numerical range does not automatically mean a feature is more important. Scaling addresses numerical magnitude, not actual predictive importance.

### Algorithms That Often Benefit from Scaling

- K-Nearest Neighbors (KNN)
- K-Means Clustering
- Support Vector Machines (SVM)
- Principal Component Analysis (PCA)
- Logistic Regression
- Ridge and Lasso Regression
- Neural Networks
- Models trained using gradient-based optimization

Tree-based models such as Decision Trees and Random Forests generally do not require feature scaling.

---

## 2. Standardization (Z-Score Scaling)

Standardization transforms a feature so that its mean becomes approximately 0 and its standard deviation becomes approximately 1.

### Formula

`z = (x - mean) / standard deviation`

Where:

- `x` = original value
- `mean` = mean of the feature
- `standard deviation` = standard deviation of the feature
- `z` = standardized value

### Example

Consider the values:

`10, 20, 30`

Mean = 20

Population standard deviation = approximately 8.165

Standardized values:

- 10 → approximately -1.225
- 20 → 0
- 30 → approximately 1.225

### Characteristics

- Mean becomes approximately 0.
- Standard deviation becomes approximately 1.
- Values are not restricted to a fixed range.
- Relative ordering and spacing are preserved.
- The distribution does not automatically become normal.
- Outliers can influence the mean and standard deviation.

### When to Use Standardization?

Standardization is a common starting point when an algorithm is sensitive to feature scales and no fixed output range is required.

It is commonly used with:

- KNN
- SVM
- PCA
- Logistic Regression
- Ridge and Lasso Regression
- Gradient-based optimization algorithms

### Python Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

`fit_transform()` learns the mean and standard deviation from the training data and transforms it. `transform()` applies the same learned parameters to the test data.

---

## 3. Normalization (Min-Max Scaling)

In many machine learning workflows, normalization refers to Min-Max Scaling. It transforms feature values into a specified range, commonly between 0 and 1.

### Formula

`x_scaled = (x - min) / (max - min)`

Where:

- `x` = original value
- `min` = minimum value of the feature
- `max` = maximum value of the feature
- `x_scaled` = normalized value

### Example

Consider the values:

`10, 20, 30`

Minimum = 10

Maximum = 30

Normalized values:

- 10 → 0
- 20 → 0.5
- 30 → 1

The minimum maps to 0, the maximum maps to 1, and intermediate values are scaled proportionally.

### Characteristics

- The default output range is [0, 1].
- Relative ordering and spacing are preserved.
- The distribution's shape is generally preserved.
- It does not make data normally distributed.
- It is sensitive to extreme minimum and maximum values.
- New observations can fall outside [0, 1] if they exceed the training data's range.

### When to Use Normalization?

Min-Max Scaling can be useful when a fixed feature range is desirable, including some neural network workflows.

It should not automatically be preferred over standardization. The appropriate method depends on the algorithm and dataset.

### Python Implementation

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 4. Robust Scaling

Robust Scaling uses the median and interquartile range (IQR) instead of the mean and standard deviation.

It is designed to reduce the influence of outliers on the scaling parameters.

### Formula

`x_scaled = (x - median) / IQR`

`IQR = Q3 - Q1`

Where:

- `median` = 50th percentile
- `Q1` = 25th percentile
- `Q3` = 75th percentile
- `IQR` = interquartile range

### Example

Consider the values:

`10, 20, 30, 40, 1000`

The value 1000 is an extreme observation.

Using the common percentile convention:

- Median = 30
- Q1 = 20
- Q3 = 40
- IQR = 40 - 20 = 20

Robust-scaled values:

- 10 → -1
- 20 → -0.5
- 30 → 0
- 40 → 0.5
- 1000 → 48.5

The extreme value remains extreme, but it does not distort the median and IQR as strongly as it would distort the mean and standard deviation.

**Important:** Robust Scaling does not remove outliers. It uses statistics that are less sensitive to extreme observations.

### Characteristics

- The median becomes 0.
- The IQR becomes 1 when the IQR is nonzero.
- Values are not restricted to a fixed range.
- It is generally less sensitive to outliers than StandardScaler and MinMaxScaler.
- It does not guarantee a normal distribution.
- Extreme observations can still affect the model.

### When to Use Robust Scaling?

Consider Robust Scaling when numerical features contain extreme values and you want scaling parameters that are less influenced by them.

### Python Implementation

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 5. Comparison of Scaling Methods

| Property | StandardScaler | MinMaxScaler | RobustScaler |
|---|---|---|---|
| Centering statistic | Mean | Minimum | Median |
| Scaling statistic | Standard deviation | Maximum - minimum | IQR |
| Typical range | Unbounded | [0, 1] | Unbounded |
| Outlier sensitivity | Sensitive | Highly sensitive to extremes | Less sensitive |
| Makes data normally distributed? | No | No | No |
| Common use | General scale-sensitive models | Fixed-range inputs | Data containing influential outliers |

### Do Scaling Methods Change the Distribution?

All three methods apply a feature-wise linear transformation, provided the scaling denominator is valid.

They generally preserve:

- Ordering of observations
- Relative spacing between observations
- Distribution shape, including skewness

They change the numerical scale and location of values, not the underlying ordering.

However, none of these methods guarantees a normal distribution.

---

## 6. Important Rules

### Rule 1: Fit the Scaler Only on Training Data

```python
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Do not fit the scaler on the entire dataset before splitting. This allows information from the test set to influence preprocessing parameters and can cause data leakage.

### Rule 2: Scaling Is Not Feature Selection

Scaling does not identify which features are important. Predictive importance depends on the relationship between the features, target, and model.

### Rule 3: Scaling Does Not Fix Skewness

A highly skewed feature generally remains skewed after standardization or Min-Max Scaling.

A separate transformation, such as a logarithm for suitable positive data, may help address skewness.

### Rule 4: Scaling Depends on the Algorithm

Scaling is often important for distance-based, gradient-based, and regularized models. It is usually unnecessary for tree-based models.

### Rule 5: Select Using Validation Performance

There is no universally best scaler. Compare appropriate alternatives using cross-validation and choose the one that performs best for the task.
