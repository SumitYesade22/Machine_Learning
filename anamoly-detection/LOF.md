# Local Outlier Factor (LOF)

## 1. Introduction

Local Outlier Factor (LOF) is an **unsupervised anomaly detection algorithm** based on the concept of local density.

It identifies anomalies by comparing the local density of a data point with the local densities of its nearest neighbors.

Unlike Isolation Forest, which identifies anomalies based on how easily observations can be isolated using random partitions, LOF uses distance and neighborhood density.

### Main Idea

A data point can be considered an anomaly when its local density is significantly lower than the local density of its neighbors.

For example:

- Points inside a densely populated region generally have similar local densities.
- A point located in a sparse region surrounded by densely populated points may have a lower local density.
- Such a point can receive a high LOF score and be identified as a potential anomaly.

**Important:** LOF detects local anomalies. A point is evaluated relative to its neighbors, not simply based on its global distance from the entire dataset.

---

## 2. Why Do We Need LOF?

Consider a dataset containing two clusters:

- Cluster A has high density.
- Cluster B has relatively low density.

Suppose a point belongs to Cluster B.

A global distance-based approach might incorrectly consider this point anomalous because it is far from the majority of observations.

However, if the point has approximately the same density as its neighbors in Cluster B, it may be a perfectly normal observation.

LOF addresses this problem by comparing the density of a point with the density of its local neighborhood.

### Global vs Local Anomaly Detection

| Global Anomaly Detection | Local Anomaly Detection |
|---|---|
| Considers observations relative to the overall dataset. | Considers observations relative to their local neighborhoods. |
| May struggle with clusters having different densities. | Can handle normal clusters with different local densities. |
| Focuses on global deviation. | Focuses on local density deviation. |

---

## 3. Important Terminologies

Before understanding the algorithm, understand these five concepts.

### 3.1 K-Nearest Neighbors (KNN)

For a given observation P, KNN identifies its k nearest observations based on a selected distance metric.

For example, if k = 3, the three nearest observations to P are selected.

Here, k determines the neighborhood size used by LOF.

### 3.2 K-Distance

The k-distance of a point P is the distance between P and its kth nearest neighbor.

For example:

Suppose the distances of P from other observations in ascending order are:

`1, 2, 3, 5, 8`

If k = 3:

`k-distance(P) = 3`

It represents the distance required to reach the kth nearest neighbor.

### 3.3 K-Distance Neighborhood

The k-distance neighborhood of P consists of all observations whose distance from P is less than or equal to its k-distance.

Mathematically:

`N_k(P) = { O | d(P,O) <= k-distance(P) }`

**Important points:**

- The neighborhood contains at least k observations.
- It can contain more than k observations when multiple points have equal distances at the kth-neighbor boundary.
- The neighborhood defines the local comparison group for P.

### 3.4 Reachability Distance

Reachability distance is a modified distance between two observations.

Formula:

`reachability_distance_k(P,O) = max(k-distance(O), d(P,O))`

Where:

- P = observation whose density is being calculated.
- O = one of P's neighbors.
- d(P,O) = actual distance between P and O.
- k-distance(O) = distance from O to its kth nearest neighbor.

**Why do we use reachability distance?**

It prevents the distance contribution from becoming smaller than the neighborhood scale of O.

There are two cases:

**Case 1: Actual distance is greater than O's k-distance**

`d(P,O) > k-distance(O)`

The actual distance is considered.

Example:

`d(P,O) = 5`

`k-distance(O) = 2`

`reachability distance = max(5,2) = 5`

No adjustment occurs.

**Case 2: O's k-distance is greater than actual distance**

`k-distance(O) > d(P,O)`

O's k-distance is considered instead.

Example:

`d(P,O) = 2`

`k-distance(O) = 7`

`reachability distance = max(2,7) = 7`

The reachability distance increases.

**Impact on P's density:**

- Larger reachability distance increases the average reachability distance of P.
- A larger average reachability distance decreases the Local Reachability Density (LRD) of P.
- Smaller reachability distance decreases the average and increases LRD.

**Important distinction:**

O's k-distance is used to adjust the reachability distance from P to O.

O's own LRD is calculated separately using O's own neighborhood.

These are two different concepts.

### 3.5 Local Reachability Density (LRD)

LRD measures the local density of an observation based on the average reachability distance to its neighbors.

Formula:

`LRD_k(P) = 1 / (average reachability distance of P to its neighbors)`

Expanded:

`LRD_k(P) = 1 / [sum(reachability_distance_k(P,O)) / |N_k(P)|]`

Where:

- `N_k(P)` = neighborhood of P.
- `|N_k(P)|` = number of observations in that neighborhood.

**Interpretation:**

| Average Reachability Distance | LRD |
|---|---|
| Low | High density |
| High | Low density |

Remember:

`LRD = 1 / Average Reachability Distance`

A smaller average distance means observations are closer together, resulting in higher local density.

---

## 4. Local Outlier Factor (LOF)

After calculating the LRD of a point and its neighbors, LOF compares their densities.

Formula:

`LOF_k(P) = Average(LRD_k(O) for O in N_k(P)) / LRD_k(P)`

In simple words:

`LOF = Average density of neighbors / Density of current point`

### Interpretation of LOF Score

| LOF Score | Interpretation |
|---|---|
| Approximately 1 | Density is similar to neighbors; generally normal. |
| Greater than 1 | Point has lower density than its neighbors. |
| Significantly greater than 1 | Potential local anomaly. |
| Less than 1 | Point is denser than its neighbors. |

**Important:**

- LOF is not a probability.
- LOF greater than 1 does not automatically mean anomaly.
- A suitable anomaly threshold depends on the dataset and application.
- A higher LOF indicates a greater relative density difference.

---

## 5. Complete LOF Algorithm

The algorithm follows these five steps.

### Step 1: Calculate K-Distance

For every observation P, calculate its distance from all other observations.

Sort the distances and identify the kth nearest neighbor.

Its distance is the k-distance of P.

### Step 2: Find K-Distance Neighborhood

Identify all observations whose distance from P is less than or equal to its k-distance.

These observations form the neighborhood of P.

### Step 3: Calculate Reachability Distance

For every neighbor O of P, calculate:

`reachability_distance_k(P,O) = max(k-distance(O), d(P,O))`

Notice that this formula uses the k-distance of O, not P.

### Step 4: Calculate Local Reachability Density

Calculate the average reachability distance of P to all its neighbors.

Take its reciprocal:

`LRD_k(P) = 1 / Average Reachability Distance`

This gives the local density estimate of P.

### Step 5: Calculate LOF Score

Calculate the average LRD of P's neighbors and divide it by P's own LRD.

`LOF_k(P) = Average Neighbor LRD / LRD(P)`

A significantly high LOF indicates a potential local anomaly.

---

## 6. Complete Numerical Example (1D Dataset)

Consider the following one-dimensional dataset:

`[0, 1, 2, 3, 4, 10]`

Assume:

`k = 2`

We will calculate LOF for:

- Q = 2 (normal point)
- P = 10 (potential anomaly)

### Step 1: Calculate K-Distance

For each point, calculate the distances to all other points.

| Point | Sorted distances | 2nd nearest distance |
|---|---|---:|
| 0 | 1, 2, 3, 4, 10 | 2 |
| 1 | 1, 1, 2, 3, 9 | 1 |
| 2 | 1, 1, 2, 2, 8 | 1 |
| 3 | 1, 1, 2, 3, 7 | 1 |
| 4 | 1, 2, 3, 4, 6 | 2 |
| 10 | 6, 7, 8, 9, 10 | 7 |

Therefore:

- k-distance(0) = 2
- k-distance(1) = 1
- k-distance(2) = 1
- k-distance(3) = 1
- k-distance(4) = 2
- k-distance(10) = 7

### Step 2: Find K-Distance Neighborhoods

Using the condition:

`d(P,O) <= k-distance(P)`

| Point | K-Distance Neighborhood |
|---|---|
| 0 | {1, 2} |
| 1 | {0, 2} |
| 2 | {1, 3} |
| 3 | {2, 4} |
| 4 | {3, 2} |
| 10 | {4, 3} |

### Step 3: Calculate Reachability Distances

Formula:

`reachability_distance(P,O) = max(k-distance(O), d(P,O))`

For point 10:

Its neighbors are 4 and 3.

**For neighbor 4:**

`d(10,4) = 6`

`k-distance(4) = 2`

`reachability_distance(10,4) = max(6,2) = 6`

**For neighbor 3:**

`d(10,3) = 7`

`k-distance(3) = 1`

`reachability_distance(10,3) = max(7,1) = 7`

Therefore:

`Average Reachability Distance(10) = (6+7)/2 = 6.5`

### Step 4: Calculate LRD

First, calculate the LRD of every point needed.

**LRD(0):**

Neighbors = {1, 2}

- Reachability(0,1) = max(1,1) = 1
- Reachability(0,2) = max(1,2) = 2

`LRD(0) = 1 / ((1+2)/2) = 0.667`

**LRD(1):**

Neighbors = {0, 2}

- Reachability(1,0) = max(2,1) = 2
- Reachability(1,2) = max(1,1) = 1

`LRD(1) = 1 / ((2+1)/2) = 0.667`

**LRD(2):**

Neighbors = {1, 3}

- Reachability(2,1) = max(1,1) = 1
- Reachability(2,3) = max(1,1) = 1

`LRD(2) = 1 / ((1+1)/2) = 1`

**LRD(3):**

Neighbors = {2, 4}

- Reachability(3,2) = max(1,1) = 1
- Reachability(3,4) = max(2,1) = 2

`LRD(3) = 1 / ((1+2)/2) = 0.667`

**LRD(4):**

Neighbors = {3, 2}

- Reachability(4,3) = max(1,1) = 1
- Reachability(4,2) = max(1,2) = 2

`LRD(4) = 1 / ((1+2)/2) = 0.667`

**LRD(10):**

Neighbors = {4, 3}

- Reachability(10,4) = 6
- Reachability(10,3) = 7

`LRD(10) = 1 / ((6+7)/2)`

`LRD(10) = 1/6.5 = 0.154`

Summary:

| Point | LRD |
|---|---:|
| 0 | 0.667 |
| 1 | 0.667 |
| 2 | 1.000 |
| 3 | 0.667 |
| 4 | 0.667 |
| 10 | 0.154 |

### Step 5: Calculate LOF for P = 10

Neighbors of 10 = {4, 3}

`LOF(10) = Average(LRD(4), LRD(3)) / LRD(10)`

`= ((0.667 + 0.667)/2) / 0.154`

`LOF(10) ≈ 4.33`

**Conclusion:** Point 10 has significantly lower local density than its neighbors. It is a potential anomaly.

### Step 6: Calculate LOF for Q = 2

Neighbors of 2 = {1, 3}

`LRD(2) = 1`

`Average Neighbor LRD = (0.667 + 0.667)/2 = 0.667`

`LOF(2) = 0.667 / 1`

`LOF(2) ≈ 0.667`

**Conclusion:** Point 2 has higher local density than its neighbors. It is not an anomaly.

### Final Comparison

| Point | LRD | Average Neighbor LRD | LOF | Result |
|---|---:|---:|---:|---|
| 2 | 1.000 | 0.667 | 0.667 | Normal |
| 10 | 0.154 | 0.667 | 4.33 | Potential anomaly |

**Key Learning:** LOF does not simply check whether a point is far away. It checks whether the point is less dense than its local neighborhood.

---

## 7. Distance Metrics Used in LOF

LOF requires a distance metric to identify nearest neighbors.

The choice of metric depends on the nature of the features.

### 7.1 Euclidean Distance

Measures straight-line distance between two points.

`d(P,O) = sqrt(sum((P_i - O_i)^2))`

Commonly used for continuous numerical features.

### 7.2 Manhattan Distance

Measures the sum of absolute differences between coordinates.

`d(P,O) = sum(|P_i - O_i|)`

Can be useful when movement or differences are represented along grid-like dimensions.

### 7.3 Minkowski Distance

A generalized distance metric.

`d(P,O) = (sum(|P_i - O_i|^p))^(1/p)`

Special cases:

- p = 1: Manhattan distance.
- p = 2: Euclidean distance.

### 7.4 Cosine Distance

Measures the difference in direction between two vectors.

`Cosine Distance = 1 - Cosine Similarity`

Can be useful for high-dimensional text vectors and embeddings when direction is more meaningful than magnitude.

### 7.5 Hamming Distance

Counts the number of positions at which two vectors differ.

Useful for binary and categorical representations.

**Practical note:** Metric selection should be based on the meaning and representation of the features, not just on algorithm availability.

---

## 8. Feature Scaling in LOF

Feature scaling is important because LOF relies on distance calculations.

Consider two features:

- Age: 18–60
- Salary: 20,000–200,000

Salary has a much larger numerical range and may dominate Euclidean distance.

Consequently, neighborhood selection and anomaly scores can be heavily influenced by salary.

Common scaling techniques:

### StandardScaler

Transforms features to have approximately:

- Mean = 0
- Standard deviation = 1

Useful when standardization is appropriate for the feature distributions.

### MinMaxScaler

Transforms values into a specified range, commonly [0,1].

Useful when a bounded feature range is desirable.

### RobustScaler

Uses the median and interquartile range.

Can be useful when features contain extreme values.

**Important:** Scaling is not a universal guarantee of better results. Choose it based on feature distributions and domain meaning.

---

## 9. Important Hyperparameters

### 9.1 n_neighbors

Controls the number of neighbors used to estimate local density.

Smaller k:

- Captures very local density differences.
- Can detect small, localized anomalies.
- More sensitive to noise and small neighborhood changes.

Larger k:

- Uses a broader neighborhood.
- Produces a more smoothed density comparison.
- May fail to detect small local anomalies if the neighborhood becomes too large.

The choice of k significantly affects the LOF scores.

### 9.2 contamination

Represents the expected proportion of anomalies in the training data.

In scikit-learn, it is used to determine the threshold for classifying observations as outliers.

Important distinction:

- `contamination` controls the decision threshold.
- It does not directly change the underlying LOF density calculation.
- A wrong contamination assumption can produce too many or too few predicted anomalies.

### 9.3 metric

Specifies the distance metric used to find neighbors.

Common choices include:

- Minkowski
- Euclidean
- Manhattan
- Cosine
- Hamming

### 9.4 p

The power parameter used by the Minkowski metric.

- p = 1: Manhattan distance.
- p = 2: Euclidean distance.

### 9.5 algorithm

Controls the nearest-neighbor search method.

Common options in scikit-learn:

- auto
- ball_tree
- kd_tree
- brute

The appropriate method depends on dataset size, dimensions, metric and implementation constraints.

---

## 10. LOF in Scikit-Learn: Important Concepts

### 10.1 novelty=False

This is the default setting.

Used for detecting outliers in the dataset on which LOF is fitted.

The `fit_predict()` workflow is commonly used in this mode.

### 10.2 novelty=True

Used when the objective is to detect anomalies in unseen observations.

For example:

- Fit the model on historical normal data.
- Receive new observations.
- Detect whether new observations are anomalous.

Important differences:

- Novelty detection and outlier detection are different use cases.
- With novelty=True, use prediction and scoring methods on unseen observations.
- Do not assume the scores and predictions for training observations are interchangeable with those from the default outlier-detection mode.

### 10.3 Negative Outlier Factor

Scikit-learn exposes an attribute called `negative_outlier_factor_`.

It is the negative of the LOF score.

Therefore:

- Values close to -1 generally indicate normal observations.
- More negative values indicate stronger outlier behavior.

Do not confuse this with the original positive LOF score.

---

## 11. Advantages of LOF

1. **Local anomaly detection:** Identifies observations whose density is lower than their local neighbors.

2. **Different-density clusters:** Can recognize normal observations in clusters with different density levels, provided their local neighborhoods are appropriately represented.

3. **Unsupervised learning:** Does not require labeled anomaly data.

4. **Flexible distance metrics:** Supports different metrics depending on the feature representation.

5. **Point-level anomaly scores:** Provides a continuous LOF score instead of only a binary label.

6. **Useful for irregular local structures:** Can identify anomalies that global distance-based approaches may miss.

---

## 12. Disadvantages of LOF

1. **Computational cost:** Requires nearest-neighbor searches, which can become expensive for large datasets.

2. **High-dimensional data:** Distance measures become less informative as dimensionality increases due to the curse of dimensionality.

3. **Sensitive to k:** Different neighborhood sizes can produce different anomaly scores.

4. **Sensitive to feature scaling:** Features with larger numerical ranges can dominate distance calculations.

5. **Sensitive to irrelevant features:** Irrelevant features can distort neighborhood relationships.

6. **Metric selection:** An inappropriate distance metric can result in incorrect neighborhoods.

7. **Threshold selection:** A high LOF score does not automatically establish that an observation is anomalous.

8. **Collective anomalies:** A small, compact group of anomalous observations may have similar local densities to one another and therefore receive LOF scores close to 1.

9. **Memory and computation:** Storing and processing neighbor relationships can become challenging for large datasets. 

---

## 13. Important Limitation: LOF May Miss Compact Anomalous Clusters

Consider a small group of observations located far away from the main dataset.

If the points within this small group are close to each other, they may have similar local densities.

Their LOF scores can therefore be close to 1.

Even though the entire group is globally isolated, LOF may not identify its members as anomalies.

Why?

Because LOF compares each observation with its local neighbors, and those neighbors belong to the same isolated group.

This is an example of a collective or group anomaly.

**Important:** A globally isolated cluster is not automatically anomalous. Domain knowledge is required to decide whether the group represents legitimate behavior or an anomaly.

---

## 14. LOF vs Isolation Forest

| Parameter | LOF | Isolation Forest |
|---|---|---|
| Approach | Local density-based | Random partition-based |
| Main principle | Compares local densities | Measures how easily points are isolated |
| Distance calculation | Required | Not required |
| Feature scaling | Usually important | Generally not required |
| Local anomalies | Particularly useful | Can struggle with some local anomalies |
| Global anomalies | Depends on neighborhood structure | Often useful for globally isolated observations |
| High-dimensional data | Distance concentration can be problematic | Can also struggle with irrelevant features, but does not rely on pairwise distances |
| Main hyperparameters | n_neighbors, metric, contamination | n_estimators, max_samples, max_features, contamination |
| Output | LOF anomaly score | Isolation-based anomaly score |

### When to Use Which?

**Prefer LOF when:**

- Local density differences are important.
- The dataset contains clusters with different densities.
- Anomalies are expected to be locally less dense than their neighbors.

**Consider Isolation Forest when:**

- You need an efficient tree-based anomaly detection approach.
- The dataset has many observations.
- Globally isolated observations are of interest.
- Pairwise distance calculations are undesirable.

Neither algorithm is universally better.

---

## 15. Practical Considerations Before Applying LOF

Before using LOF on a real-world dataset:

1. Understand the meaning of each feature.
2. Handle missing values appropriately.
3. Identify categorical and numerical features.
4. Encode categorical features using meaningful representations.
5. Remove or investigate irrelevant features.
6. Scale numerical features when appropriate.
7. Select a suitable distance metric.
8. Experiment with different values of k.
9. Inspect the distribution of LOF scores.
10. Validate suspicious observations using domain knowledge.

### How to Evaluate LOF?

If labeled anomaly data is available, use:

- Precision
- Recall
- F1-score
- Precision-Recall AUC
- ROC-AUC

For imbalanced anomaly detection problems, Precision-Recall AUC and Precision@K can be particularly informative.

If labels are unavailable:

- Inspect high-scoring observations.
- Check whether suspicious records make business sense.
- Evaluate score stability across different k values.
- Use domain experts to validate detected anomalies.
- Consider synthetic anomalies carefully, since synthetic examples may not represent real anomalies.

---

## 16. Important Interview Questions

### Q1. What is LOF?

LOF is an unsupervised, density-based anomaly detection algorithm that compares the local density of an observation with the average local density of its nearest neighbors.

### Q2. What is the difference between k-distance and reachability distance?

K-distance is the distance to the kth nearest neighbor of a point.

Reachability distance is the maximum of the actual distance between two points and the k-distance of the neighboring point.

### Q3. Why is the k-distance of O used instead of P in reachability distance?

Because LOF defines the reachability distance from P to O using O's local neighborhood scale as a lower bound.

This creates a scale-adjusted distance used in estimating P's local density.

### Q4. Why do we calculate LRD?

LRD estimates the local density of an observation using the inverse of its average reachability distance.

### Q5. What does an LOF score greater than 1 indicate?

It indicates that the observation has lower local density than its neighbors. A sufficiently high score may indicate an anomaly.

### Q6. Can LOF detect anomalies in clusters having different densities?

Yes. Since LOF compares local densities instead of global distances, it can identify anomalies while handling clusters with different local densities.

### Q7. Why is feature scaling important in LOF?

Because LOF uses distance calculations. Features with larger numerical ranges can dominate the distance and affect nearest-neighbor selection.

### Q8. Can LOF detect a small isolated cluster as anomalous?

Not necessarily. If observations within the isolated cluster have similar local densities, their LOF scores may be close to 1.

### Q9. What happens when we increase k?

The neighborhood becomes broader, resulting in a more global density comparison. Small local anomalies may become harder to detect.

### Q10. Is LOF a classification algorithm?

No. It is an unsupervised anomaly detection algorithm. It calculates anomaly scores, which can subsequently be used to classify observations using a chosen threshold.

---

## 17. Key Takeaways

- LOF is an unsupervised, local density-based anomaly detection algorithm.
- It compares an observation's local density with the local densities of its neighbors.
- K-distance determines the neighborhood scale.
- K-distance neighborhood identifies the comparison group.
- Reachability distance adjusts the actual distance using the neighbor's k-distance.
- LRD is the inverse of average reachability distance.
- LOF compares the average neighbor LRD with the current point's LRD.
- LOF approximately equal to 1 indicates similar density.
- A significantly higher LOF indicates a potential local anomaly.
- Feature scaling and distance metric selection are important.
- The choice of k significantly affects results.
- LOF may miss compact isolated anomalous groups.
- LOF is not a probability and is not universally better than Isolation Forest.

## 18. Reference

- Original Research Paper: *LOF: Identifying Density-Based Local Outliers* — Markus M. Breunig, Hans-Peter Kriegel, Raymond T. Ng, Jörg Sander (2000).
- Scikit-learn Documentation: Local Outlier Factor (LOF).
- Scikit-learn User Guide: Novelty and Outlier Detection.

