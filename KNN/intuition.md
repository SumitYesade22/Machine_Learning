# K-Nearest Neighbors (KNN)

# 1. Introduction to KNN

## 1.1 What is K-Nearest Neighbors?

K-Nearest Neighbors (KNN) is a supervised, non-parametric, instance-based machine learning algorithm used for both classification and regression tasks.

It predicts the output for a new observation by identifying the K closest observations from the training dataset.

The fundamental assumption behind KNN is:

> Observations that are close to each other in the feature space are likely to have similar target values.

For example, consider a house price prediction problem.

If a house has similar characteristics to nearby houses, such as area, number of bedrooms, and location, its price may also be similar.

KNN uses this local similarity to make predictions.

### Key characteristics

- Supervised learning algorithm.
- Non-parametric algorithm.
- Instance-based learning.
- Lazy learning algorithm.
- Supports classification and regression.
- Relies on distance or similarity measurements.
- Requires feature scaling in most numerical applications.

## 1.2 How Does KNN Work?

KNN follows a simple prediction procedure.

1. Store the training dataset.
2. Receive a new observation for prediction.
3. Calculate the distance between the new observation and all relevant training observations.
4. Identify the K nearest observations.
5. Aggregate their target values to make a prediction.

The aggregation method depends on the task:

- Classification: Majority voting.
- Regression: Average of neighboring target values.

### Example: KNN Classification

Suppose we want to classify a new observation into Class A or Class B.

Let K = 5.

The algorithm identifies the five nearest training observations:

| Neighbor | Class |
|---|---|
| 1 | A |
| 2 | A |
| 3 | B |
| 4 | A |
| 5 | B |

Class A receives three votes, while Class B receives two votes.

Therefore:

**Predicted class = A**

### Example: KNN Regression

Suppose the target values of the three nearest neighbors are:

`[200, 250, 300]`

For uniform-weight KNN regression:

`Predicted value = (200 + 250 + 300) / 3`

`Predicted value = 250`

## 1.3 Instance-Based Learning

KNN is an instance-based learning algorithm.

Instead of learning a generalized mathematical function during training, it stores the training observations and uses them directly during prediction.

For example:

In Linear Regression, the model learns coefficients such as:

`y = b0 + b1*x1 + b2*x2`

KNN does not learn such a fixed global equation.

Instead, it compares a new observation with stored training observations and uses nearby examples to generate predictions.

### Important characteristics

- Training mainly involves storing observations.
- Predictions depend directly on stored training data.
- The algorithm performs most of its computational work during prediction.
- New observations can influence predictions without retraining a global mathematical model, although they must be incorporated into the stored dataset or index.

## 1.4 Lazy Learning vs Eager Learning

Machine learning algorithms can be broadly differentiated by when they perform most of their learning computation.

### Lazy Learning

Lazy learning algorithms defer most computation until prediction time.

KNN is a classic example.

Characteristics:

- Minimal model-building computation.
- Stores training observations.
- Performs distance calculations during prediction.
- Prediction can be computationally expensive.

## 1.5 KNN Classification vs KNN Regression

KNN supports two types of supervised learning tasks.

| Parameter | Classification | Regression |
|---|---|---|
| Target variable | Categorical | Numerical |
| Prediction method | Majority voting | Average or weighted average |
| Example | Spam detection | House price prediction |
| Output | Class label | Numerical value |

---

# 2. Distance Metrics

## 2.1 Why Do We Need Distance Metrics?

KNN identifies neighbors based on the distance between observations.

A distance metric determines how similarity or dissimilarity between observations is measured.

Different distance metrics can produce different nearest neighbors.

Therefore, choosing an appropriate metric is important for KNN performance.

## 2.2 Euclidean Distance

Euclidean distance measures the straight-line distance between two points.

It is the most commonly used distance metric for numerical data.

### Formula

For two observations:

`P = (p1, p2, ..., pn)`

`Q = (q1, q2, ..., qn)`

Euclidean distance:

`d(P,Q) = sqrt(sum((pi - qi)^2))`

### Example

Consider:

`P = (2, 3)`

`Q = (5, 7)`

`d(P,Q) = sqrt((5-2)^2 + (7-3)^2)`

`= sqrt(9 + 16)`

`= sqrt(25) = 5`

### Characteristics

- Measures straight-line distance.
- Sensitive to feature scales.
- Larger coordinate differences have a stronger influence because differences are squared.
- Commonly used for continuous numerical features.

## 2.3 Manhattan Distance

Manhattan distance measures the total absolute difference between coordinates.

It is also known as L1 distance or City Block distance.

### Formula

`d(P,Q) = sum(|pi - qi|)`

### Example

Consider:

`P = (2, 3)`

`Q = (5, 7)`

`d(P,Q) = |5-2| + |7-3|`

`= 3 + 4`

`= 7`

### Characteristics

- Calculates the sum of absolute coordinate differences.
- Does not square differences.
- Can be useful when movement or distance is represented along coordinate-wise paths.
- Also sensitive to feature scales.

## 2.4 Minkowski Distance

Minkowski distance is a generalized distance formula.

Euclidean and Manhattan distances are special cases of Minkowski distance.

### Formula

`d(P,Q) = (sum(|pi - qi|^p))^(1/p)`

Here, `p` determines the distance calculation.

### Important cases

**When p = 1:**

`d(P,Q) = sum(|pi - qi|)`

This becomes Manhattan distance.

**When p = 2:**

`d(P,Q) = sqrt(sum((pi - qi)^2))`

This becomes Euclidean distance.

### Example

For:

`P = (2, 3)`

`Q = (5, 7)`

When p = 1:

`Distance = 7`

When p = 2:

`Distance = 5`

## 2.5 Chebyshev Distance

Chebyshev distance measures the maximum absolute difference across all dimensions.

It is also known as L-infinity distance.

### Formula

`d(P,Q) = max(|pi - qi|)`

### Example

Consider:

`P = (2, 3)`

`Q = (5, 7)`

Absolute differences:

- X difference = 3
- Y difference = 4

Therefore:

`Distance = max(3, 4) = 4`

### Characteristics

- Considers only the largest coordinate difference.
- Useful in applications where the maximum movement or deviation across dimensions matters.

## 2.6 Hamming Distance

Hamming distance counts the number of positions at which two equal-length sequences differ.

It is commonly used with binary or categorical representations.

### Example

`P = 101101`

`Q = 100001`

Compare each position:

| Position | P | Q | Different? |
|---|---|---|---|
| 1 | 1 | 1 | No |
| 2 | 0 | 0 | No |
| 3 | 1 | 0 | Yes |
| 4 | 1 | 0 | Yes |
| 5 | 0 | 0 | No |
| 6 | 1 | 1 | No |

Total differences = 2.

Therefore:

`Hamming Distance = 2`

### Important consideration

Hamming distance is appropriate for representations where positions have meaningful equality or inequality.

Do not arbitrarily Label Encode nominal categories and then interpret numerical differences as meaningful distances.

For example:

- Red = 1
- Blue = 2
- Green = 3

The numerical difference between these labels does not imply that Green is more different from Red than Blue is.

## 2.7 Cosine Similarity and Cosine Distance

Cosine similarity measures the similarity between two vectors based on the angle between them.

It focuses on vector direction rather than their absolute magnitudes.

### Formula

`Cosine Similarity = (P . Q) / (||P|| * ||Q||)`

Where:

- `P . Q` = Dot product.
- `||P||` = Magnitude of vector P.
- `||Q||` = Magnitude of vector Q.

Cosine distance is commonly defined as:

`Cosine Distance = 1 - Cosine Similarity`

### Example

Consider two vectors:

`P = [1, 2]`

`Q = [2, 4]`

Both vectors point in the same direction.

Their cosine similarity is 1.

Therefore:

`Cosine Distance = 1 - 1 = 0`

They have zero cosine distance despite having different magnitudes.

### Applications

Cosine similarity is frequently used in:

- Text similarity.
- Document retrieval.
- Semantic search.
- High-dimensional vector representations.
- Embedding-based applications.

### Important distinction

Cosine distance is a commonly used dissimilarity measure, but it is not a metric in the strict mathematical sense for all vector spaces and definitions.

## 2.8 Comparison of Distance Metrics

| Metric | Calculation | Common Use |
|---|---|---|
| Euclidean | Straight-line distance | Continuous numerical data |
| Manhattan | Sum of absolute differences | Coordinate-wise distance |
| Minkowski | Generalized distance | Flexible numerical distance |
| Chebyshev | Maximum coordinate difference | Maximum deviation |
| Hamming | Number of differing positions | Binary/categorical representations |
| Cosine | Directional similarity | Text and vector embeddings |

## 2.9 Choosing the Appropriate Distance Metric

There is no universally best distance metric.

The choice depends on:

- Nature of the features.
- Data representation.
- Problem requirements.
- Feature scaling.
- Validation performance.

A practical approach is to evaluate different suitable metrics using the same preprocessing strategy and cross-validation splits.

## 2.10 Effect of Distance Metrics on Predictions

Changing the distance metric can change which observations are selected as nearest neighbors.

This can change:

- The selected neighborhood.
- Classification votes.
- Regression predictions.
- Decision boundaries.

Therefore, distance metric selection is an important hyperparameter optimization decision.

---

# 3. Feature Scaling and Curse of Dimensionality

## 3.1 Why Is Feature Scaling Important in KNN?

KNN relies on distance calculations.

If features have significantly different numerical scales, features with larger ranges can dominate the distance calculation.

### Example

Consider a house dataset:

| House | Area (sq. ft.) | Bedrooms |
|---|---:|---:|
| A | 1000 | 2 |
| B | 1200 | 3 |
| C | 1500 | 4 |

Suppose we calculate Euclidean distance between A and C.

`Distance = sqrt((1500-1000)^2 + (4-2)^2)`

`= sqrt(250000 + 4)`

`≈ 500.004`

The area difference contributes almost everything to the distance.

This happens because area has a much larger numerical scale than bedrooms.

Consequently, the algorithm may not properly account for features measured on smaller scales.

**Feature scaling reduces this scale-related distortion.**

## 3.2 Curse of Dimensionality

The curse of dimensionality refers to several problems that arise as the number of dimensions increases.

It affects distance-based algorithms such as KNN.

### Problem 1: Exponential Growth of Feature Space

As dimensions increase, the volume of the feature space grows rapidly.

For illustration, divide each dimension into 10 intervals.

| Dimensions | Possible Regions |
|---:|---:|
| 1 | 10 |
| 2 | 100 |
| 3 | 1,000 |
| 4 | 10,000 |
| 5 | 100,000 |
| 10 | 10,000,000,000 |

Formula:

`Number of regions = 10^D`

Where D is the number of dimensions.

The number of possible regions grows exponentially.

If the number of observations does not increase accordingly, the feature space becomes sparsely populated.

### Problem 2: Data Sparsity

As the dimensionality increases, a fixed number of observations covers a smaller proportion of the possible feature space.

Consequences:

- Observations become more spread out.
- Local neighborhoods may contain relatively distant points.
- Finding genuinely similar neighbors becomes difficult.
- Predictions may become less reliable.

### Problem 3: Distance Concentration

In high-dimensional spaces, distances between observations can become relatively similar under common data distributions.

The difference between nearest and farthest distances may become small relative to their magnitudes.

This phenomenon is called distance concentration.

Why is it a problem?

KNN relies on the assumption that nearby observations are more similar than distant observations.

If almost all observations have relatively similar distances, the distinction between near and far becomes less informative.

The mathematically nearest observation is still nearest, but its relative closeness may not indicate meaningful similarity.

### Problem 4: Irrelevant Features Distort Distances

Suppose a dataset has:

- Age
- Income
- Education
- Random_ID
- Unrelated numerical feature

If irrelevant features are included in distance calculations, they can distort the neighborhood structure.

KNN may select observations that are close according to the calculated distance but are not genuinely similar for the prediction task.

### Problem 5: Computational Complexity

For brute-force nearest-neighbor search:

`Distance calculation complexity = O(N * D)`

Where:

- N = number of training observations.
- D = number of features.

Increasing dimensionality also increases the computational cost of distance calculations.

## 3.7 Why Does Distance Concentration Affect KNN?

KNN needs meaningful local neighborhoods.

Suppose the nearest observation has distance 9.8 and the farthest has distance 10.2.

Although one observation is mathematically closer, the relative difference is small.

The nearest observation may not be substantially more similar than other observations.

This weakens the usefulness of distance-based neighborhood selection.

## 3.8 Handling the Curse of Dimensionality

### 1. Feature Selection

Remove irrelevant or redundant features.

Possible techniques:

- Variance Threshold
- Correlation-based feature selection
- Mutual Information
- Recursive Feature Elimination (RFE)
- Sequential Feature Selection

Feature selection reduces the number of dimensions while retaining useful information.

### 2. Dimensionality Reduction

Use dimensionality reduction techniques such as PCA.

PCA transforms the original features into a smaller set of principal components.

It can reduce dimensionality and redundancy.

However, PCA preserves directions of high variance, not necessarily the features or directions most predictive of the target.

Therefore, its impact should be evaluated using validation performance.

### 3. Domain Knowledge

Select features that are relevant to the prediction problem.

For example, in house price prediction, location and area may be more useful than an unrelated record identifier.

### 4. Appropriate Distance Metric

Choose a distance metric that reflects the actual meaning of similarity in the data.

---

# 4. KNN Classification

## 4.1 How Does KNN Classification Work?

KNN classification predicts the class of a new observation using the majority class among its K nearest training observations.

### Steps

1. Store the training observations and labels.
2. Receive a new observation.
3. Calculate distances to training observations.
4. Identify the K nearest observations.
5. Count the class labels among those neighbors.
6. Return the majority class.

## 4.2 Majority Voting

Majority voting means selecting the class that occurs most frequently among the K nearest neighbors.

### Formula

`Predicted Class = argmax_c Count(y_i = c)`

Where:

- c represents a possible class.
- `y_i` represents the class of a neighboring observation.

### Example

Let K = 5.

| Neighbor | Class |
|---|---|
| 1 | Spam |
| 2 | Spam |
| 3 | Not Spam |
| 4 | Spam |
| 5 | Not Spam |

Votes:

- Spam = 3
- Not Spam = 2

Predicted class = Spam.

## 4.3 Effect of K on Classification

The value of K determines how many neighbors participate in voting.

### Small K

Example: K = 1.

Only the nearest observation determines the prediction.

Characteristics:

- Highly local decision-making.
- Sensitive to noise.
- Sensitive to individual observations.
- High variance.
- Greater risk of overfitting.

### Large K

Example: K = 50.

A larger neighborhood contributes to the prediction.

Characteristics:

- Smoother decision boundaries.
- Less sensitive to individual observations.
- Higher bias in many situations.
- Greater risk of underfitting.

### Comparison

| Small K | Large K |
|---|---|
| More flexible | Less flexible |
| Higher variance | Lower variance |
| Lower bias generally | Higher bias generally |
| Noise-sensitive | Can ignore local patterns |
| Overfitting risk | Underfitting risk |

These are general tendencies, not guaranteed outcomes for every dataset.

## 4.4 Decision Boundaries

A decision boundary separates regions of the feature space assigned to different predicted classes.

KNN creates decision boundaries based on the class distribution of nearby observations.

### Effect of K

**Small K:**

- More irregular boundaries.
- Local observations have strong influence.
- Greater sensitivity to noise.

**Large K:**

- Smoother boundaries.
- More observations influence predictions.
- Local patterns may be lost.

## 4.5 Distance-Weighted Voting

In standard KNN classification, all K neighbors receive equal voting weight.

However, a closer neighbor may be more relevant than a farther neighbor.

Distance-weighted voting gives closer observations greater influence.

### Common Weighting Formula

`w_i = 1 / d_i`

Where:

- `w_i` = weight of neighbor i.
- `d_i` = distance of neighbor i from the query.

The weighted class score is:

`Score(c) = sum(w_i * I(y_i = c))`

Where `I(y_i = c)` is 1 when the neighbor belongs to class c, otherwise 0.

The class with the highest weighted score is selected.

### Example

Suppose three neighbors:

| Neighbor | Class | Distance | Weight |
|---|---|---:|---:|
| 1 | A | 1 | 1.00 |
| 2 | B | 2 | 0.50 |
| 3 | B | 4 | 0.25 |

Uniform voting:

- A = 1 vote.
- B = 2 votes.

Prediction = B.

Distance-weighted voting:

- A score = 1.00
- B score = 0.50 + 0.25 = 0.75

Prediction = A.

The closest neighbor changes the final prediction.

## 4.6 Tie Handling

A tie occurs when two or more classes receive the same number of votes.

Example:

K = 4.

- Class A = 2 votes.
- Class B = 2 votes.

There is no unique majority class.

Possible approaches include:

- Use an odd K in binary classification.
- Use distance-weighted voting.
- Apply a deterministic tie-breaking rule.
- Evaluate the behavior of the chosen implementation.

An odd K reduces the possibility of ties in binary classification, but it does not eliminate ties in multiclass classification.

Scikit-learn has implementation-specific behavior when class vote counts are tied, so it should not be assumed that every tie is resolved by choosing the closest class.

## 4.7 Binary vs Multiclass Classification

### Binary Classification

Two possible classes.

Example:

- Spam
- Not Spam

Prediction is based on votes from the two classes.

### Multiclass Classification

More than two possible classes.

Example:

- Cat
- Dog
- Horse

The class receiving the highest vote or weighted score is selected.

## 4.8 Choosing K Using Cross-Validation

There is no universally optimal value of K.

The best value depends on:

- Dataset size.
- Noise level.
- Feature distribution.
- Class distribution.
- Distance metric.
- Feature scaling.

A practical approach is to evaluate multiple values of K using cross-validation.

---

# 5. KNN Regression

## 5.1 How Does KNN Regression Work?

KNN regression predicts a numerical target by identifying the K nearest observations and aggregating their target values.

Unlike classification, it does not select a majority class.

Instead, it calculates an average or weighted average.

### Steps

1. Store the training observations and numerical target values.
2. Receive a new observation.
3. Calculate distances to training observations.
4. Select the K nearest observations.
5. Aggregate their target values.
6. Return the numerical prediction.

## 5.2 Mean-Based Prediction

In standard KNN regression, all selected neighbors receive equal weights.

The prediction is the arithmetic mean of their target values.

### Formula

`y_pred = (y1 + y2 + ... + yk) / K`

### Example

Suppose the target values of the three nearest neighbors are:

`[200, 250, 300]`

Then:

`y_pred = (200 + 250 + 300) / 3`

`y_pred = 250`

## 5.3 Distance-Weighted Regression

Closer observations can receive greater influence.

### Formula

`w_i = 1 / d_i`

Weighted prediction:

`y_pred = sum(w_i * y_i) / sum(w_i)`

Where:

- `w_i` = weight of neighbor i.
- `y_i` = target value of neighbor i.
- `d_i` = distance from query to neighbor i.

### Example

| Neighbor | Target | Distance | Weight |
|---|---:|---:|---:|
| 1 | 200 | 1 | 1.00 |
| 2 | 250 | 2 | 0.50 |
| 3 | 300 | 4 | 0.25 |

Weighted prediction:

`y_pred = (1*200 + 0.5*250 + 0.25*300) / (1 + 0.5 + 0.25)`

`= (200 + 125 + 75) / 1.75`

`= 228.57`

The closest observation contributes more to the prediction.

## 5.4 Effect of K on Regression

### Small K

- Prediction depends on fewer observations.
- More sensitive to noise.
- Captures local patterns.
- Higher variance.
- Greater risk of overfitting.

### Large K

- Prediction averages more observations.
- Smoother predictions.
- Less sensitive to individual observations.
- Higher bias in many situations.
- Greater risk of underfitting.

## 5.5 Choosing the Optimal K

Evaluate different K values using cross-validation.

For regression, commonly used evaluation metrics include:

- Mean Absolute Error (MAE).
- Mean Squared Error (MSE).
- Root Mean Squared Error (RMSE).
- R-squared.


## 5.6 Important Limitation: KNN Regression and Extrapolation

KNN regression generally predicts values within the range of the selected neighbors' target values.

With uniform or positive distance weights, its prediction is a weighted average of the selected target values.

Therefore:

`min(neighbor targets) <= prediction <= max(neighbor targets)`

This means KNN regression does not naturally extrapolate beyond the observed target range.

For example, if all selected neighbors have house prices between ₹40 lakh and ₹60 lakh, their weighted average cannot produce a prediction of ₹90 lakh.

This can be a limitation when predicting values outside the range represented by the training observations.

## 5.7 Outliers in KNN Regression

An outlier among the selected neighbors can influence the prediction.

For example:

`[200, 220, 240, 2000]`

The mean becomes:

`(200 + 220 + 240 + 2000) / 4 = 665`

The extreme target value significantly increases the prediction.

Distance weighting may reduce the influence of a distant outlier, but it does not automatically eliminate its effect.

---

# 6. KNN Optimization and Hyperparameter Tuning

## 6.1 Choosing the Optimal K

The value of K is one of the most important KNN hyperparameters.

A very small K may overfit, while a very large K may underfit.

The optimal value should be selected using validation data or cross-validation.

There is no universal formula that gives the best K for every dataset.

## 6.2 Bias-Variance Tradeoff

The bias-variance tradeoff explains the relationship between model flexibility and prediction errors.

### Small K

- More flexible model.
- Highly dependent on nearby observations.
- Lower bias generally.
- Higher variance generally.
- Greater overfitting risk.

### Large K

- Less flexible model.
- More observations contribute to predictions.
- Higher bias generally.
- Lower variance generally.
- Greater underfitting risk.

The objective is to select a K that balances bias and variance.
--- 

# 7. Limitations of KNN

## 7.1 Computationally Expensive Prediction

KNN stores training observations and calculates distances during prediction.

For brute-force search, the distance calculation cost for one query is approximately:

`O(N * D)`

Where:

- N = Number of training observations.
- D = Number of features.

This can become expensive for large datasets.

## 7.2 High Memory Requirement

KNN needs access to the stored training observations during prediction.

Therefore, memory consumption can increase significantly as the training dataset grows.

## 7.3 Curse of Dimensionality

As dimensionality increases:

- Data becomes sparse.
- Distance concentration can occur.
- Neighborhoods become less informative.
- Distance-based predictions may become less reliable.

## 7.4 Sensitivity to Feature Scaling

Features with larger numerical ranges can dominate distance calculations.

Appropriate scaling is generally required for numerical features.

## 7.5 Sensitivity to Irrelevant Features

Irrelevant features can distort distance measurements and lead to poor neighbor selection.

## 7.6 Sensitivity to Noise and Outliers

Small K values can be particularly sensitive to noisy observations.

In regression, extreme target values can influence local averages.

## 7.7 No Natural Extrapolation in Regression

KNN regression predicts using the target values of existing neighbors.

It does not naturally learn a global trend that can be extended beyond the observed target range.

## 7.8 Class Imbalance

In classification, a majority class can dominate neighborhood voting.

This may result in poor minority-class predictions.


