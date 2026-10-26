# Missing Data Mechanisms: MCAR, MAR and MNAR

Missing data mechanisms explain **why some values are missing** in a dataset. Understanding these mechanisms helps us choose an appropriate missing-value handling strategy.

## 1. MCAR — Missing Completely At Random

### Definition
Missingness is independent of both observed and unobserved data values.

In simple terms, values are missing due to a completely random process.

### Example
A college collects students' attendance and study hours. Due to a random server failure, some study-hour entries are lost, regardless of students' attendance or actual study hours.

### Key Points
- Missingness is not systematically related to any observed or unobserved variable.
- Missing values do not occur more frequently in any particular group due to a data-related pattern.
- Complete-case deletion may be reasonable in some situations, but it reduces the sample size.

---

## 2. MAR — Missing At Random

### Definition
Missingness depends on observed variables, but after accounting for those variables, it does not depend on the missing values themselves.

### Example
A college collects students' attendance and study hours.

| Student | Attendance | Study Hours |
|---|---:|---:|
| A | 95% | 5 |
| B | 90% | 4 |
| C | 40% | Missing |
| D | 35% | Missing |
| E | 45% | Missing |

Suppose students with low attendance are less likely to complete the study-hours question.

Here:
- Attendance is observed.
- Study hours are missing for some students.
- Missingness depends on the observed attendance variable.

This can be MAR **if, after accounting for attendance, missingness does not additionally depend on the actual unobserved study hours**.

### Key Points
- Missingness is related to one or more observed variables.
- Observed variables can help explain the missingness pattern.
- Methods such as multiple imputation can use observed variables to handle missing values.

---

## 3. MNAR — Missing Not At Random

### Definition
Missingness depends on the missing value itself or on an unobserved factor that is not adequately accounted for by the observed data.

### Example
Students who study very few hours are more likely to hide their actual study hours in a survey.

| Student | Actual Study Hours | Recorded Study Hours |
|---|---:|---|
| A | 6 | 6 |
| B | 5 | 5 |
| C | 1 | Missing |
| D | 0.5 | Missing |

Here, students with lower actual study hours are more likely to leave the question unanswered.

Therefore, missingness depends on the unobserved study-hour value itself, making this an MNAR mechanism.

### Key Points
- Missingness may depend on the value that is missing.
- Observed variables alone may not adequately explain the missingness mechanism.
- Ordinary imputation methods may produce biased estimates.
- Domain knowledge and sensitivity analysis may be needed.

---

## 4. Comparison

| Mechanism | What does missingness depend on? | Example |
|---|---|---|
| MCAR | Neither observed nor unobserved data values | Random server failure |
| MAR | Observed variable(s), after accounting for which missingness does not depend on the missing value | Missing study hours related to observed attendance |
| MNAR | Missing value itself or an unobserved factor | Students with low actual study hours hide their answers |

## 5. How to Identify the Mechanism

- **MCAR:** Little's MCAR test and analysis of missingness patterns may provide evidence, but cannot prove MCAR in every situation.
- **MAR:** Investigate whether missingness is associated with observed variables. This association alone does not prove MAR.
- **MNAR:** MAR and MNAR generally cannot be reliably distinguished using observed data alone. Additional information, domain knowledge, or sensitivity analysis may be required.

## 6. Impact on Missing-Value Handling

- **MCAR:** Complete-case deletion or simple imputation may be reasonable depending on the analysis and amount of missing data.
- **MAR:** Methods such as multiple imputation can incorporate observed variables associated with missingness.
- **MNAR:** Specialized models and sensitivity analysis may be required because standard methods can produce biased estimates.


**Important:** MAR and MNAR describe the missingness mechanism, not the percentage of missing values in a dataset.

## Univariate Imputation

Univariate imputation is a missing-value handling technique in which missing values in a feature are filled using information from the **same feature only**.

### Example

| Age |
|---:|
| 22 |
| 25 |
| NaN |
| 30 |
| 35 |

The missing value in `Age` is imputed using the available values of the `Age` column, without using other features such as salary or education.

**Key Point:** Univariate imputation uses information from a single feature to fill its missing values.

> **Important Principle of Imputation**
>
> **Before and after imputation, evaluate whether the important statistical properties and underlying patterns of the original data are reasonably preserved.**
>
> - **Distribution:** Shape, skewness, mean, median, and variance.
> - **Categorical Proportions:** Relative frequencies of different categories.
> - **Relationships Between Features:** Correlations and associations between variables.
> - **Variability and Outliers:** Spread of the data and meaningful extreme values.
> - **Overall Data Patterns:** Important trends and relationships relevant to the analysis or model.
>
> **Note:** Different imputation methods can affect these properties differently. The goal is to minimize unnecessary distortion, not to guarantee exact preservation of every property.

## 1. Mean Imputation

Mean imputation replaces missing values in a numerical feature with the **mean of its available (non-missing) values**.

**Formula:**

`Mean = Sum of non-missing values / Number of non-missing values`

### Advantages
- Simple and easy to implement.
- Preserves the number of rows in the dataset.
- Works reasonably well for numerical features with approximately symmetric distributions and few missing values.

### Disadvantages
- Sensitive to outliers because extreme values affect the mean.
- Reduces the variance of the feature.
- Can distort the original distribution.
- May weaken relationships between features.

### When to Use
- Numerical features with approximately symmetric distributions.
- Features with few missing values and no influential outliers.

---

## 2. Median Imputation

Median imputation replaces missing values in a numerical feature with the **median of its available (non-missing) values**.

**Formula:**

After sorting the non-missing values:
- Odd number of observations: middle value.
- Even number of observations: average of the two middle values.

### Advantages
- Robust to outliers.
- Suitable for skewed numerical distributions.
- Preserves the number of rows in the dataset.
- Less affected by extreme values than mean imputation.

### Disadvantages
- Reduces the variance of the feature.
- Can distort the original distribution.
- May weaken relationships between features.
- Does not account for relationships with other features.

### When to Use
- Skewed numerical features.
- Numerical features containing outliers.
- Features where the mean is not representative of a typical value.

---

## 3. Mean vs. Median Imputation

| Property | Mean | Median |
|---|---|---|
| Outlier sensitivity | High | Low |
| Skewed data | Usually less suitable | Usually more suitable |
| Approximately symmetric data | Often suitable | Also suitable |
| Computational complexity | Low | Low |
| Preserves variance exactly | No | No |

**Important:** Both methods are simple univariate imputation techniques. They do not account for the missingness mechanism or relationships with other features. If missingness is substantial, or if the data is MAR or MNAR, consider whether more advanced methods are appropriate. Mean or median imputation is not guaranteed to produce unbiased results.It can be usd in case of MCAR

## Categorical Imputation

Categorical imputation replaces missing values in categorical features with suitable categories or predicted values.

### Common Methods

1. **Mode Imputation:** Replaces missing values with the most frequent category.
2. **Constant Imputation:** Replaces missing values with a predefined category, such as `"Unknown"` or `"Missing"`.
3. **Missing Indicator:** Creates an additional binary feature to indicate whether the original value was missing.
4. **Predictive Imputation:** Uses other features and a classification model to predict missing categories.
5. **Random Sample Imputation:** Replaces missing values by randomly sampling from the observed categories.

### When to Use

- **Mode:** When missing values are few and the most frequent category is representative.
- **Constant:** When missingness should be represented as a separate category.
- **Missing Indicator:** When missingness itself may contain useful predictive information.
- **Predictive:** When other features can help predict the missing category.
- **Random Sample:** When maintaining approximate category proportions is important.

### Important Considerations

- Imputation can distort category proportions and relationships between features.
- Mode imputation can over-represent the most frequent category.
- Predictive imputation may introduce bias if the model is inappropriate.
- Choose the method based on the missingness mechanism, data characteristics, and modeling objective.


## Random Sample Imputation

Random sample imputation replaces missing values by randomly selecting values from the **observed (non-missing) values of the same feature**, with replacement.

### Advantages
- Can preserve the observed value distribution better than mean/median imputation.
- Does not replace every missing value with the same constant.
- Can be applied to numerical and categorical features.

### Disadvantages
- Does not guarantee preservation of feature relationships or the original distribution.
- Results can vary due to random sampling.
- May introduce bias if missingness depends on other variables or the missing values themselves.

### When to Use
- As a simple baseline when sampling from observed values is reasonable.
- Can be considered for MCAR, depending on the data and analysis objective.
- For MAR, consider conditional or group-wise sampling using relevant observed features.

### Missingness Indicator
A binary indicator can be added **before imputation** to identify which values were originally missing.

# Multivariate Imputation: KNN Imputer and Iterative Imputer

Multivariate imputation methods estimate missing values by utilizing information from multiple features in a dataset rather than relying only on the feature containing missing values.

The two important multivariate imputation techniques are:

1. K-Nearest Neighbors (KNN) Imputation
2. Iterative Imputation

---

# 1. K-Nearest Neighbors (KNN) Imputer

## 1.1 What Is KNN Imputation?

KNN Imputation is a multivariate imputation technique that estimates missing values using the values of similar observations, called nearest neighbors.

Similarity between observations is determined using a distance metric calculated from their available features.

The underlying assumption is that observations with similar feature values are likely to have similar values for the feature being imputed.

## 1.2 How Does KNN Imputation Work?

Suppose a dataset contains three numerical features:

| Student | Study Hours | Attendance (%) | Marks |
|---|---:|---:|---:|
| A | 2 | 60 | 40 |
| B | 4 | 80 | 60 |
| C | 6 | 90 | 80 |
| D | 4 | 80 | Missing |

The marks of Student D are missing.

The imputation process works as follows:

1. **Calculate distances:** Compare Student D with other observations using their available features.
2. **Identify nearest neighbors:** Select the K observations with the smallest usable distances.
3. **Retrieve available values:** Consider the neighbors whose marks are observed.
4. **Estimate the missing value:** Aggregate the available neighboring values.

For example, if the two nearest usable neighbors have marks of 60 and 80, and uniform weighting is used, the imputed value is:

Imputed Marks = (60 + 80) / 2 = 70

Therefore, the missing marks are estimated as 70.

## 1.3 How Is Distance Calculated?

A common distance metric is Euclidean distance.

For two observations with complete numerical features:

Distance = √[(x1 − y1)² + (x2 − y2)² + ... + (xn − yn)²]

However, missing data complicates distance calculation.

Scikit-learn's KNNImputer uses the `nan_euclidean` distance metric by default.

This metric:

- Uses features that are observed in both observations.
- Excludes features that are missing in either observation.
- Adjusts the distance to account for the number of features used relative to the total number of features.

Consequently, two observations can still have a usable distance even when some of their features are missing.

If two observations have no common observed features, their distance cannot be calculated using this metric.

## 1.4 What Happens When a Neighbor Also Has a Missing Value?

Suppose the target observation has missing marks, and three nearest neighbors are selected.

| Neighbor | Marks |
|---|---:|
| A | 60 |
| B | Missing |
| C | 90 |

Neighbor B cannot contribute a marks value because its marks are missing.

With uniform weighting, the available values are averaged:

Imputed Marks = (60 + 90) / 2 = 75

If a selected neighbor has a missing value for the feature being imputed, that neighbor is excluded from the aggregation for that feature. Other usable observations may contribute instead.

If the feature has no observed values available for imputation, a meaningful estimate cannot be obtained from neighbors.

## 1.5 How Is K Chosen?

K represents the number of nearest neighbors considered.

- **Small K:** More sensitive to individual observations and local variation.
- **Large K:** Uses more observations but may combine dissimilar samples and smooth away meaningful differences.

There is no universally optimal K. It should be selected using validation on the training data.

The selected K should provide a reasonable balance between local similarity and estimation stability.

## 1.6 Weighting Methods

KNN imputation can aggregate neighboring values in different ways.

**Uniform weighting:** Every usable neighbor contributes equally to the estimate.

**Distance weighting:** Closer neighbors receive greater weight than more distant neighbors.

Distance weighting can be useful when the closest observations are more representative of the target observation than the other selected neighbors.

## 1.7 Feature Scaling and KNN Imputation

KNN is sensitive to feature scales because distances depend on numerical differences.

Suppose a dataset contains:

- Age, ranging from 18 to 60.
- Annual income, ranging from 200,000 to 2,000,000.

Income may dominate the distance calculation because its numerical scale is much larger.

Consequently, scaling numerical features may be necessary before applying KNN imputation.

Scaling should be fitted using training data only and then applied consistently to validation and test data.

## 1.8 Advantages of KNN Imputation

- Utilizes relationships between multiple features.
- Can capture local patterns and nonlinear relationships without fitting an explicit predictive model.
- Does not require assuming that every feature follows a normal distribution.
- Can provide reasonable estimates when similar observations are available.
- Supports numerical multivariate imputation.

## 1.9 Limitations of KNN Imputation

- Can be computationally expensive for large datasets because distances must be calculated between observations.
- Sensitive to feature scaling and the choice of K.
- Performance may deteriorate when irrelevant features distort distance calculations.
- May perform poorly when observations are not meaningfully similar.
- Can be unreliable when many features are missing simultaneously.
- Can reduce variability when neighboring values are averaged.
- Standard KNNImputer is designed for numerical data, not direct categorical imputation.

## 1.10 When Should KNN Imputation Be Used?

Consider KNN imputation when:

- Similar observations are available in the dataset.
- Relationships between observations are meaningful.
- Numerical features have been appropriately scaled when necessary.
- Local patterns are useful for estimating missing values.

For example, missing battery capacity in a headphone dataset might be estimated using products with similar specifications, provided those similarities are meaningful and enough observed battery capacities are available.

---

# 2. Iterative Imputer

## 2.1 What Is Iterative Imputation?

Iterative Imputation is a multivariate imputation technique in which each feature containing missing values is predicted using the other available features.

Instead of finding the nearest observations, it fits predictive models to estimate missing values.

The process repeats across features over multiple rounds, allowing estimates to be updated iteratively.

In scikit-learn, this method is implemented through `IterativeImputer`.

By default, it uses Bayesian Ridge regression as its estimator.

## 2.2 How Does Iterative Imputation Work?

Suppose a dataset contains the following features:

| Student | Study Hours | Attendance (%) | Marks |
|---|---:|---:|---:|
| A | 2 | 60 | 40 |
| B | Missing | 90 | 80 |
| C | 6 | 80 | 90 |
| D | 4 | 70 | Missing |

Both Study Hours and Marks contain missing values.

The process works as follows.

### Step 1: Initial Imputation

Initially, missing values are filled with provisional estimates.

By default, scikit-learn uses the mean of each feature's observed values to initialize its missing values.

These initial values allow the subsequent predictive models to operate on a usable dataset.

### Step 2: Predict the First Feature

Suppose Study Hours is selected as the target feature.

A predictive model is trained using observations with known Study Hours values, with the other features serving as predictors.

Conceptually:

Study Hours = f(Attendance, Marks)

The trained model predicts Study Hours for observations where it is missing.

These predictions replace the initial estimates for the missing Study Hours values.

### Step 3: Predict the Next Feature

Marks can then be selected as the target feature.

Conceptually:

Marks = f(Study Hours, Attendance)

The model is trained using observations with known Marks values.

It then predicts the missing Marks values using the other features.

### Step 4: Repeat the Process

The imputer continues cycling through features containing missing values.

The newly estimated values can be used when predicting other features in subsequent steps.

This allows the initial estimates to be refined over successive rounds.

### Step 5: Stop the Iterations

The process stops when either:

- The convergence criterion is satisfied, or
- The maximum number of iterations is reached.

The default maximum is 10 iterations in scikit-learn.

Ten iterations are not always required; the algorithm may stop earlier if its stopping criterion is met.

## 2.3 What Happens When a Predictor Feature Is Also Missing?

Suppose an observation has both Study Hours and Marks missing, while Attendance is available.

Initially, the missing values receive provisional estimates.

When Study Hours is predicted, the model can use Attendance and the initialized Marks value as predictors.

Later, when Marks is predicted, the model can use Attendance and the updated Study Hours value.

The iterative process can progressively update these estimates.

However, the estimates are not guaranteed to recover the original values. Their quality depends on the observed data, predictive relationships, model assumptions, and missingness patterns.

## 2.4 Why Is It Called Iterative Imputation?

It is called iterative because the procedure repeatedly:

1. Selects a feature to impute.
2. Fits a predictive model using the other features.
3. Predicts that feature's missing values.
4. Moves to the next feature.
5. Repeats the cycle to update the estimates.

Each complete round through the features is an iteration, not a single missing cell being filled.

## 2.5 Role of the Predictive Model

The choice of estimator influences the imputation results.

Scikit-learn's default estimator is Bayesian Ridge regression.

Other estimators can be considered when appropriate, including:

- Linear regression for approximately linear relationships.
- Random forest models for nonlinear relationships and interactions.
- Other suitable regressors for numerical features.

The estimator should be selected based on the structure of the data and the relationships between features.

Using a more complex estimator does not automatically guarantee better imputation.

## 2.6 What Does Convergence Mean?

Convergence means that the imputed values have stopped changing substantially between successive rounds according to the algorithm's stopping criterion.

In scikit-learn, convergence is assessed using the relative change in imputed values and the configured tolerance.

If the criterion is met, the procedure can stop before reaching the maximum number of iterations.

If convergence is not reached, the algorithm stops when the maximum iteration limit is reached.

Convergence does not guarantee that the imputed values are accurate; it only indicates that the estimates have stabilized according to the specified criterion.

## 2.7 Advantages of Iterative Imputation

- Uses relationships between multiple features.
- Can capture predictive relationships that simple univariate imputation ignores.
- Can accommodate multiple columns containing missing values.
- Allows different predictive estimators to be considered.
- Can be effective when the missing features are predictable from the other observed features.

## 2.8 Limitations of Iterative Imputation

- Computationally more expensive than simple imputation.
- Results depend on the predictive model and its assumptions.
- Poorly predicted features can produce inaccurate imputations.
- Highly correlated or redundant features may cause modeling difficulties.
- Imputed values may become overly confident or fail to reflect the uncertainty associated with missing data.
- It does not guarantee unbiased estimates under every missingness mechanism.
- Standard IterativeImputer is primarily designed for numerical features; categorical data requires appropriate encoding or a categorical imputation approach.

## 2.9 When Should Iterative Imputation Be Used?

Consider Iterative Imputation when:

- Features have meaningful predictive relationships.
- Multiple numerical features contain missing values.
- One feature can reasonably be estimated from the others.
- A predictive-model-based approach is appropriate for the dataset.

For example, in a headphone dataset, missing playtime might be estimated using battery capacity, product category, and other relevant specifications if the observed data supports those relationships.

