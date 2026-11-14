# Isolation Forest – Anomaly Detection

## 1. Introduction to Isolation Forest

Isolation Forest is an **unsupervised machine learning algorithm** used for detecting anomalies (unusual observations) in a dataset.

It is based on the principle:

**Anomalies are easier to isolate than normal observations.**

The algorithm identifies anomalies by randomly partitioning the dataset and measuring how easily individual observations can be separated from the remaining observations.

- Normal observations generally require more splits to get isolated.
- Anomalous observations generally require fewer splits to get isolated.

Isolation Forest does not require labeled data during training.

### Example

Consider a dataset containing transaction amounts:

`500, 550, 600, 650, 700, 5000`

Most transactions are between 500 and 700, whereas 5000 is significantly different.

A random split at 1000 separates 5000 from all the other observations in a single split.

The observation 5000 is therefore easier to isolate and becomes a potential anomaly.

**Important:** An unusual observation is not necessarily an error or fraud. Isolation Forest identifies unusual patterns; domain knowledge is required to interpret them.

---

## 2. Why Is It Called Isolation Forest?

The name has two parts:

- **Isolation:** The algorithm attempts to separate individual observations from the remaining observations.
- **Forest:** It constructs multiple Isolation Trees and combines their results.

An Isolation Forest consists of multiple independently constructed Isolation Trees (iTrees).

Each tree uses random sampling and random splitting to isolate observations.

The final anomaly score is calculated using the path lengths obtained from these trees.

---

## 3. Core Concepts Behind Isolation Forest

Before understanding the complete algorithm, understand these fundamental concepts:

1. Random sampling
2. Random feature selection
3. Random splitting
4. Isolation Trees
5. Path length
6. Average path length
7. Normalization using c(n)
8. Anomaly score

---

## 4. Random Sampling

Isolation Forest does not necessarily use the entire training dataset to construct every tree.

Instead, it randomly selects a subset of observations for each Isolation Tree.

This parameter is called `max_samples`.

### Example

Suppose:

- Total dataset observations = 1000
- Number of trees = 100
- `max_samples` = 256

Each tree is constructed using 256 randomly selected observations.

| Tree | Number of selected observations |
|---|---:|
| Tree 1 | 256 |
| Tree 2 | 256 |
| Tree 3 | 256 |
| ... | ... |
| Tree 100 | 256 |

### Are samples different for every tree?

Yes, each tree independently performs random sampling.

However, samples can overlap between different trees.

For example:

- Tree 1 may select observations A, B, C and D.
- Tree 2 may select observations A, C, E and F.
- Tree 3 may select observations B, D, G and H.

The same observation can be selected by multiple trees.

### Sampling with and without replacement

By default, scikit-learn uses:

`bootstrap=False`

This means sampling is performed **without replacement** within each tree.

Therefore:

- An observation cannot be selected more than once in the same tree.
- The same observation can be selected in different trees.

### What happens to observations not selected?

Suppose the original dataset contains 1000 observations, but a particular tree uses only 256.

The remaining 744 observations are not used to construct that particular tree.

They are called **Out-of-Bag (OOB) observations** relative to that tree.

However, these observations are not permanently discarded.

They can be selected by other trees.

Also, they can be passed through the already constructed tree to calculate their path lengths.

**Important:** Standard scikit-learn Isolation Forest does not automatically provide a Random Forest-style OOB error estimate.

---

## 5. Random Feature Selection

Isolation Forest works with both one-dimensional and multidimensional datasets.

Suppose a dataset contains two features:

- Transaction Amount
- Transaction Frequency

An Isolation Tree can use both features.

However, each individual split selects only one feature.

For example:

1. First split: Transaction Amount < 1000?
2. Second split: Transaction Frequency < 10?
3. Third split: Transaction Amount < 600?

Therefore:

- One split uses one feature.
- One tree can use multiple features.
- Different trees can select different features and split values.

The random feature selection helps create diverse Isolation Trees.

### Important distinction

Isolation Forest does not calculate distances between observations like distance-based algorithms.

It identifies unusual observations using random partitioning and path lengths.

---

## 6. Random Splitting

After selecting a feature, the algorithm randomly chooses a split value between the minimum and maximum values of that feature in the current node.

The observations are divided into two branches:

- Left branch: Observations satisfying the split condition.
- Right branch: Remaining observations.

The same process is repeated recursively for the resulting branches.

### Example: One-dimensional dataset

Consider:

`500, 550, 600, 650, 700, 5000`

Suppose the algorithm randomly selects a split value of 1000.

The resulting partitions are:

**Left branch:**

`500, 550, 600, 650, 700`

**Right branch:**

`5000`

The observation 5000 has been isolated using only one split.

The other observations still require additional splits to become individually isolated.

### Important

Every random split will not necessarily isolate an anomaly.

For example, a split at 600 may separate some normal observations while keeping the anomaly together with other observations.

This is why Isolation Forest constructs multiple trees instead of relying on a single tree.

---

## 7. Isolation Tree Construction

An Isolation Tree is a randomly constructed binary tree.

Its construction follows these steps:

1. Select a random subset of observations.
2. Select a feature randomly.
3. Select a random split value between the feature's minimum and maximum values in the current node.
4. Divide observations into two branches.
5. Repeat the process recursively for each branch.
6. Stop when a stopping condition is satisfied.

Unlike conventional Decision Trees, Isolation Trees do not select splits by minimizing Gini Impurity, Entropy or another prediction loss.

Their splits are random.

### When does an Isolation Tree stop growing?

An Isolation Tree stops splitting when one of the following conditions is met:

**Condition 1: Only one observation remains**

If a node contains one observation, that observation is isolated.

No further splitting is required.

**Condition 2: All feature values are identical**

If all available feature values in a node are identical, no valid random split can separate the observations using those features.

**Condition 3: Maximum tree height is reached**

Isolation Forest limits the maximum depth of its trees.

This prevents unnecessarily deep trees and reduces computational cost.

### Maximum tree depth

In scikit-learn, the maximum tree depth is determined internally using:

`max_depth = ceil(log2(max_samples))`

For example:

If `max_samples = 256`:

`max_depth = ceil(log2(256))`

`max_depth = 8`

Therefore, the tree cannot grow beyond the specified maximum height.

**Important:** `max_depth` is internally determined by scikit-learn; it is not a direct user-facing constructor parameter of IsolationForest.

### What happens if multiple observations remain at maximum depth?

The tree stops growing even if a leaf node contains multiple observations.

Their path lengths are adjusted using the expected additional path length, represented by `c(size of leaf)`.

This is important because the remaining observations have not been individually isolated.

---

## 8. Path Length

Path length is one of the most important concepts in Isolation Forest.

**Path length is the number of edges traversed from the root node to the leaf node containing an observation.**

### Example

Suppose an observation follows this path:

Root → Split 1 → Split 2 → Leaf

Its path length is 2 edges.

### Why is path length important?

Isolation Forest uses path length to measure how easily an observation can be isolated.

- Short path length: Observation is isolated quickly.
- Long path length: Observation requires more splits to get isolated.

Therefore:

**Shorter path length generally indicates a stronger anomaly signal.**

### Example

| Observation | Path length |
|---|---:|
| 5000 | 1 |
| 500 | 3 |
| 550 | 4 |
| 600 | 3 |
| 650 | 3 |
| 700 | 3 |

The observation 5000 has the shortest path length.

It is more likely to be an anomaly.

These values are illustrative.

---

## 9. Why Are Multiple Trees Required?

Isolation Trees use random sampling, random feature selection and random splitting.

Consequently, different trees can produce different structures.

An observation may be isolated quickly in one tree but require more splits in another.

Therefore, relying on a single tree may produce an unstable anomaly assessment.

Isolation Forest constructs multiple trees and calculates the average path length of each observation across all trees.

### Example

Suppose five Isolation Trees produce these path lengths:

| Tree | Transaction 5000 | Transaction 600 |
|---|---:|---:|
| Tree 1 | 1 | 3 |
| Tree 2 | 2 | 4 |
| Tree 3 | 1 | 3 |
| Tree 4 | 2 | 5 |
| Tree 5 | 1 | 4 |
| Average | 1.4 | 3.8 |

For transaction 5000:

`Average path length = (1 + 2 + 1 + 2 + 1) / 5`

`Average path length = 1.4`

For transaction 600:

`Average path length = (3 + 4 + 3 + 5 + 4) / 5`

`Average path length = 3.8`

Transaction 5000 has a smaller average path length.

This indicates that it is consistently easier to isolate across the trees.

The average path length is represented by:

`E(h(x))`

Where:

- `h(x)` = Path length of observation x in an individual tree.
- `E(h(x))` = Expected or average path length of observation x across the forest.

---

## 10. Why Is Normalization Required?

Suppose there are two datasets:

- Dataset A contains 10 observations.
- Dataset B contains 1000 observations.

Generally, isolating an observation in a dataset containing more observations can require more splits.

Therefore, the same path length may have different interpretations depending on the number of observations used to construct the tree.

For example:

| Dataset | Number of observations | Actual path length |
|---|---:|---:|
| A | 10 | 3 |
| B | 1000 | 3 |

A path length of 3 in a dataset containing 10 observations does not necessarily represent the same degree of isolation as a path length of 3 in a dataset containing 1000 observations.

We therefore need a reference path length that accounts for the number of observations.

This reference is represented by `c(n)`.

### What is c(n)?

`c(n)` represents the expected path length of an unsuccessful search in a Binary Search Tree (BST) containing n observations.

In simpler terms:

**It provides a reference for the expected path length based on the number of observations used to construct a tree.**

We compare the actual average path length with this reference to calculate the anomaly score.

### Important clarification

- `E(h(x))` = Actual average path length of an observation.
- `c(n)` = Expected path length reference for the sample size.

`c(n)` is not the actual path length of an observation.

It is also not a measure of the number of features.

It depends on the number of observations used to construct each tree.

### Formula for c(n)

`c(n) = 2H(n-1) - 2(n-1)/n`

Where:

- `n` = Number of observations used to construct each tree.
- `H(n-1)` = Harmonic Number.

### What is a Harmonic Number?

`H(k) = 1 + 1/2 + 1/3 + ... + 1/k`

For example:

`H(4) = 1 + 1/2 + 1/3 + 1/4`

`H(4) = 2.0833`

For n = 5:

`c(5) = 2H(4) - 2(4)/5`

`c(5) = 2(2.0833) - 1.6`

`c(5) ≈ 2.5666`

Therefore, the normalization reference for five observations is approximately 2.57.

For large n, the harmonic number can be approximated by:

`H(n) ≈ ln(n) + 0.57721566`

Here, 0.57721566 is the Euler–Mascheroni constant.

### Connection with max_samples

In scikit-learn, when `max_samples="auto"`:

`max_samples = min(256, n_samples)`

Suppose the original dataset contains 1000 observations.

Each tree uses 256 randomly selected observations by default.

Therefore:

`n = 256`

The normalization factor is based on `c(256)`, not `c(1000)`.

If every tree uses the same number of samples, the normalization factor is the same for all trees.

However, normalization is still required to interpret the average path length relative to the sample size.

---

## 11. Why Is c(size of leaf) Added to Path Length?

This is another important concept.

Suppose an Isolation Tree reaches a leaf containing two observations.

The observations have not been individually isolated.

However, the tree cannot grow further because it has reached its maximum depth.

If we use only the actual path length to the leaf, these observations may receive artificially short path lengths.

To account for the additional isolation that would be expected if the tree continued growing, the algorithm adds an estimated path-length adjustment.

The adjusted path length is:

`Adjusted path length = Actual path to leaf + c(size of leaf)`

### Example

Suppose:

- Actual path length = 2
- Number of observations remaining in the leaf = 2

We know:

`c(2) = 1`

Therefore:

`Adjusted path length = 2 + 1 = 3`

### What if only one observation remains?

If only one observation remains:

`c(1) = 0`

Therefore, no additional adjustment is required.

### Why is this adjustment necessary?

It accounts for the fact that multiple observations remain together in a leaf node.

It estimates the additional path length that would be required to separate them further.

**Important:** This adjustment is used for leaf nodes containing multiple observations. It is not added to every internal node.

---

## 12. Anomaly Score

After calculating the average path length, Isolation Forest converts it into an anomaly score.

The original Isolation Forest paper defines the score as:

`s(x,n) = 2^(-E(h(x))/c(n))`

Where:

- `s(x,n)` = Anomaly score of observation x.
- `E(h(x))` = Average path length across all trees.
- `c(n)` = Normalization factor.
- `n` = Number of observations used to construct each tree.

### Understanding the formula

The formula compares the actual average path length with the expected path length reference.

When an observation is isolated quickly:

- Its average path length is small.
- The exponent is closer to zero.
- Its anomaly score becomes closer to 1.

When an observation takes more splits to isolate:

- Its average path length is larger.
- Its anomaly score becomes smaller.

### Example

Suppose:

`n = 256`

`c(256) ≈ 10.24`

Observation A:

`E(h(A)) = 2`

`Score = 2^(-2/10.24)`

`Score ≈ 0.873`

Observation B:

`E(h(B)) = 9`

`Score = 2^(-9/10.24)`

`Score ≈ 0.544`

Observation A has a higher anomaly score.

### Interpretation of the original anomaly score

| Score | Interpretation |
|---|---|
| Close to 1 | Stronger anomaly signal |
| Around 0.5 | No strong anomaly indication |
| Close to 0 | More likely normal |

These are general interpretations of the original paper's anomaly score, not universal probability thresholds.

**Important:** Anomaly scores are not probabilities that an observation is fraudulent or anomalous.

---

## 13. Prediction on New, Unseen Data

Isolation Forest can predict anomalies for observations that were never used during training.

This is possible because the trees have already learned their random splitting rules.

### Training phase

During training:

1. Random subsets are selected.
2. Multiple Isolation Trees are constructed.
3. Each tree learns its random splitting structure.
4. The trees are stored in the trained model.

### Prediction phase

Suppose a completely new transaction arrives:

- Transaction Amount = 5000
- Transaction Frequency = 50

The new observation was not present in the training dataset.

However, it can still be evaluated by the trained trees.

The process is:

1. Pass the new observation to the root node of a trained tree.
2. Check the condition at the root node.
3. Follow the appropriate branch based on the new observation's feature value.
4. Continue until a leaf node is reached.
5. Calculate its path length, including the leaf adjustment when required.
6. Repeat the process for all trees.
7. Calculate the average path length and anomaly score.

### Example

Suppose the root node has the condition:

`Transaction Amount < 1000`

The new transaction has an amount of 5000.

Therefore:

`5000 < 1000 = False`

The observation follows the right branch.

It continues following the existing tree's splitting rules until it reaches a leaf.

Its path length can then be calculated.

**Important:**

- `fit()` constructs the Isolation Trees.
- `predict()` evaluates observations using the existing trained trees.
- New observations do not need to be part of the training samples.
- New observations are not inserted into the trained trees.
- The trees are not retrained during prediction.

This is how Isolation Forest handles unseen data.

---

## 14. Out-of-Bag (OOB) Observations

An observation not selected during the construction of a particular tree is an Out-of-Bag observation for that tree.

For example:

- Original dataset = 1000 observations.
- Tree 1 uses 256 observations.
- Remaining 744 observations are OOB for Tree 1.

These OOB observations can be passed through Tree 1 to calculate their path lengths.

However, there is an important distinction:

**An OOB prediction must only use trees that did not use that observation during training.**

If an observation is passed through all trees, including trees that used it during training, that is an ordinary prediction, not a strict OOB prediction.

Scikit-learn's standard IsolationForest does not automatically provide OOB anomaly estimates.

---

## 15. Important Hyperparameters

| Parameter | Purpose |
|---|---|
| `n_estimators` | Number of Isolation Trees constructed |
| `max_samples` | Number of observations randomly selected for each tree |
| `contamination` | Expected proportion of anomalies; used to determine the decision threshold |
| `max_features` | Number or proportion of features selected for each tree |
| `bootstrap` | Determines whether sampling is performed with replacement |
| `random_state` | Controls randomness for reproducibility |
| `n_jobs` | Number of parallel jobs used during training and prediction |

### n_estimators

Controls how many Isolation Trees are constructed.

More trees generally provide a more stable average path-length estimate, but increase computational cost.

### max_samples

Controls how many observations are selected for constructing each tree.

In scikit-learn:

`max_samples="auto"`

Means:

`max_samples = min(256, n_samples)`

For example:

If the dataset contains 1000 observations, each tree uses 256 observations by default.

### contamination

Represents the expected proportion of anomalies in the dataset.

For example:

`contamination = 0.05`

This tells the algorithm to set a threshold that classifies approximately 5% of the training observations as anomalies.

It does not mean the model has learned that exactly 5% of the observations are truly anomalous.

### max_features

Controls how many features are selected for each tree.

A tree can use multiple features, but each individual split uses only one feature.

### bootstrap

By default, `bootstrap=False`.

This means observations are sampled without replacement within each tree.

---

## 16. Scikit-learn Score Methods

Scikit-learn provides different methods for obtaining predictions and scores.

An important distinction is that its score orientation differs from the original paper's anomaly score.

| Method | Output interpretation |
|---|---|
| `predict(X)` | `1` = Normal, `-1` = Anomaly |
| `score_samples(X)` | Lower / more negative = More anomalous |
| `decision_function(X)` | Negative = Anomaly, Positive = Normal |

### score_samples()

Returns the opposite-oriented version of the original paper's anomaly score.

Lower values indicate more anomalous observations.

### decision_function()

Returns a threshold-adjusted score.

Negative values indicate anomalies, while positive values indicate normal observations.

### predict()

Returns the final binary anomaly classification:

- `1`: Normal observation.
- `-1`: Anomalous observation.

### Important distinction

The original paper's anomaly score and scikit-learn's `score_samples()` should not be interpreted in the same direction.

Do not assume that a higher `score_samples()` value means a stronger anomaly.

---

## 17. Does Isolation Forest Require Feature Scaling?

Isolation Forest is a tree-based algorithm.

Unlike distance-based algorithms such as KNN, it does not calculate Euclidean or Manhattan distances between observations.

Therefore, feature scaling is generally not mandatory.

However, appropriate preprocessing is still important.

Considerations:

- Missing values should be handled.
- Categorical features need appropriate numerical representation.
- Irrelevant features can affect random splitting.
- Constant features provide no useful splitting information.
- Feature transformations may be useful for highly skewed data depending on the application.

---

## 18. Advantages of Isolation Forest

1. Unsupervised algorithm; labeled anomaly data is not required.
2. Works with multidimensional datasets.
3. Efficient for relatively large datasets.
4. Does not require distance calculations.
5. Feature scaling is generally not mandatory.
6. Can detect unusual observations using random partitioning.
7. Can evaluate new, unseen observations using trained trees.

---

## 19. Limitations of Isolation Forest

1. Random splitting introduces variability between trees.
2. Irrelevant features can reduce anomaly detection effectiveness.
3. Local anomalies may not always be identified effectively.
4. The contamination assumption affects the classification threshold.
5. Anomaly scores are not probabilities.
6. The algorithm does not explain the business reason behind an anomaly.
7. Highly correlated or complex feature distributions can make interpretation difficult.

---

## 20. Isolation Forest vs LOF

| Aspect | Isolation Forest | LOF |
|---|---|---|
| Full form | Isolation Forest | Local Outlier Factor |
| Main principle | Random isolation | Local density comparison |
| Approach | Tree-based | Nearest-neighbor-based |
| Main measurement | Average path length | Local Reachability Density |
| Distance calculation | Not required | Required |
| Scaling | Generally not mandatory | Usually important |
| Anomaly identification | Isolation difficulty | Difference in local density |
| Local anomalies | Can be challenging | Designed to identify local outliers |

### Key difference

Isolation Forest identifies observations that can be isolated quickly through random splits.

LOF identifies observations whose local density is significantly lower than the density of their neighboring observations.

---

## 21. Complete Isolation Forest Workflow

1. Start with an unlabeled dataset.
2. Preprocess the features.
3. Select the number of Isolation Trees.
4. Randomly sample observations for each tree.
5. Randomly select features.
6. Randomly select split values.
7. Recursively partition the observations.
8. Stop splitting when a stopping condition is reached.
9. Calculate path lengths for observations in each tree.
10. Apply the leaf adjustment when multiple observations remain in a leaf.
11. Calculate average path length across the forest.
12. Normalize using `c(n)`.
13. Calculate anomaly scores.
14. Apply the decision threshold.
15. Classify observations as normal or anomalous.
16. Use the trained model to evaluate new, unseen observations.

---

## 22. Interview Explanation

Isolation Forest is an unsupervised anomaly detection algorithm based on the principle that anomalies are easier to isolate than normal observations.

It constructs multiple random Isolation Trees by selecting random subsets of observations, random features and random split values.

For each observation, it calculates the path length required to reach a leaf node in every tree.

These path lengths are averaged across the forest and normalized using the expected path length, represented by c(n).

Anomalies generally have shorter average path lengths and consequently higher anomaly scores in the original paper's scoring formulation.

The final anomaly classification is determined using a threshold.

---

## 23. Important Concepts to Remember

| Concept | Meaning |
|---|---|
| Isolation Forest | Collection of random Isolation Trees |
| Isolation Tree | Binary tree constructed using random splits |
| Random sampling | Selecting a subset of observations for each tree |
| Random feature selection | Selecting features for random splitting |
| Random split | Randomly choosing a threshold between minimum and maximum feature values |
| Path length | Number of edges from root to leaf |
| Average path length | Average path length across all trees |
| c(n) | Expected path-length normalization factor |
| Leaf adjustment | Estimated additional path length for observations remaining in a leaf |
| Anomaly score | Score derived from normalized average path length |
| Contamination | Expected anomaly proportion used to determine the threshold |
| OOB observation | Observation not selected to construct a particular tree |

**Central takeaway:**

Isolation Forest does not search for anomalies by calculating how far observations are from one another.

Instead, it uses random partitioning to measure how easily an observation can be separated from the rest of the dataset.

**Fewer splits generally indicate a stronger anomaly signal.**