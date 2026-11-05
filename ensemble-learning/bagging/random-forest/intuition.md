# Random Forest

## 1. Introduction

Random Forest is an ensemble learning technique based on **Bagging (Bootstrap Aggregating)**.

It combines multiple Decision Trees to produce a more stable and generalized prediction.

Random Forest uses:
- Bootstrap sampling to create different training datasets.
- Multiple Decision Trees as base estimators.
- Random feature selection at each node while searching for the best split.
- Aggregation of predictions from all trees.

## 2. Relationship Between Bagging and Random Forest

Random Forest is essentially an extension of Bagging that introduces additional randomness through feature selection at each split.

Both techniques train multiple models on different bootstrap samples and aggregate their predictions.

However, their feature selection strategies differ.

## 3. Bagging vs Random Forest

| Parameter | Bagging | Random Forest |
|---|---|---|
| Base estimator | Any suitable ML model | Decision Trees |
| Row sampling | Bootstrap sampling (with replacement) | Bootstrap sampling (with replacement by default) |
| Feature sampling | Can select a subset of features at the tree level | Random subset of features considered at each node |
| Feature selection level | Tree level (when feature sampling is enabled) | Node/split level |
| Feature subset | Generally remains the same throughout a tree | Can change at every node |
| Tree correlation | Can be relatively higher | Random feature selection helps reduce correlation |
| Prediction aggregation | Voting for classification; averaging for regression | Voting for classification; averaging for regression |

## 4. Feature Selection: Core Difference

### Bagging with Decision Trees

Suppose the dataset contains 10 features.

If we select 5 features for each tree:

- Tree 1 receives features A, B, C, D, E.
- Tree 2 receives features F, G, H, I, J.

The selected feature subset remains the same throughout the respective tree.

### Random Forest

Suppose the dataset contains 10 features and we consider 5 randomly selected features at each node.

Different nodes can consider different subsets:

- Root node: A, B, C, D, E.
- Node 2: B, F, G, H, I.
- Node 3: A, C, E, G, J.

The algorithm selects the best split from the randomly selected candidate features at each node.

## 5. Why Does Random Forest Use Random Feature Selection?

Suppose some features are highly predictive.

Without random feature selection, multiple Decision Trees may repeatedly select the same strong features, making their structures and predictions similar.

This increases correlation between trees.

Random Forest introduces randomness at every split, encouraging trees to explore different features and potentially learn different patterns.

As a result:

- Correlation between trees can decrease.
- Aggregation becomes more effective.
- Ensemble variance can reduce.
- Generalization performance may improve.

**Important:** Random feature selection does not guarantee lower correlation in every dataset, but it is designed to encourage diversity among trees.

## 6. Prediction Aggregation

**Classification:**

The final prediction is determined through majority voting across the Decision Trees.

**Regression:**

The final prediction is calculated by averaging predictions from all Decision Trees.

## 7. Key Takeaway

Random Forest = Bagging of Decision Trees + Random Feature Selection at Each Split.

The main purpose of introducing feature randomness is to reduce correlation between individual trees, making the ensemble more effective at reducing variance.

# Out-of-Bag (OOB) Evaluation in Random Forest

## 1. What is OOB?

Out-of-Bag (OOB) evaluation is a model evaluation technique used in Bagging and Random Forest.

It allows us to estimate a model's performance without explicitly creating a separate validation dataset.

It uses the observations that were **not selected in a particular tree's bootstrap sample** to evaluate that tree.

## 2. How Does OOB Work?

Suppose the original training dataset contains 1,000 observations.

For each Decision Tree:

1. A bootstrap sample of 1,000 observations is created with replacement.
2. Some observations appear multiple times.
3. Some observations are not selected at all.
4. The observations not selected are called Out-of-Bag (OOB) observations.

These OOB observations are not used to train that particular tree.

They can be used to evaluate its predictions.

## 3. How Are OOB Predictions Generated?

Suppose we have 3 Decision Trees and an observation X.

| Tree   | Was X used for training? | Can it predict X for OOB evaluation? |
| ------ | ------------------------ | ------------------------------------ |
| Tree 1 | Yes                      | No                                   |
| Tree 2 | No                       | Yes                                  |
| Tree 3 | No                       | Yes                                  |

For observation X:

- Tree 2 and Tree 3 provide predictions.
- Their predictions are aggregated.
- The aggregated prediction is compared with the actual target value.

The same process is repeated for all training observations.

### Important

An observation is evaluated only using trees for which it was OOB.

It is not necessary for an observation to be OOB in every tree.

## 4. Why Does OOB Work?

In bootstrap sampling, each observation has a probability of not being selected in a particular bootstrap sample.

For a dataset of N observations, the probability that a particular observation is not selected in N draws with replacement is:

P(OOB) = (1 - 1/N)^N

For sufficiently large N:

P(OOB) ≈ 0.368

Therefore, approximately **36.8% of observations are OOB for each tree**, while approximately 63.2% are represented at least once in its bootstrap sample.

This gives each tree its own set of observations that it has not seen during training.

## 5. OOB Score

Random Forest can aggregate OOB predictions for each training observation and calculate an evaluation score.

- Classification: OOB predictions are aggregated using majority voting.
- Regression: OOB predictions are aggregated using averaging.

The aggregated predictions are compared against the actual target values.

For classification, the OOB score generally represents accuracy.

For regression, the default OOB score is R².

## 6. Advantages of OOB Evaluation

1. No separate validation dataset is required for OOB evaluation.
2. Makes efficient use of the available training data.
3. Provides an estimate of generalization performance.
4. Useful for evaluating Random Forest models.
5. Helps compare different model configurations.

## 7. Limitations of OOB Evaluation

1. OOB estimates can be less stable when the number of trees is small.When the number of trees is small, each observation receives OOB predictions from fewer trees. Consequently, the aggregated prediction becomes more sensitive to individual tree predictions, making the OOB estimate less stable.
2. It is not a replacement for a final independent test dataset.
3. It requires bootstrap sampling.
4. Each observation may receive OOB predictions from a different number of trees.

# Extremely Randomized Trees (Extra Trees)

## 1. Introduction

Extremely Randomized Trees (Extra Trees) is an ensemble learning technique that combines multiple Decision Trees to produce a more stable and generalized prediction.

It is similar to Random Forest but introduces additional randomness during the split selection process.

The main objective is to increase diversity among individual trees, reduce their correlation, and potentially improve variance reduction.

## 2. Relationship Between Random Forest and Extra Trees

Both Random Forest and Extra Trees are ensemble techniques based on multiple Decision Trees.

Both techniques introduce randomness through feature selection at each node.

However, their main difference is how they select split thresholds.

- Random Forest searches for the best split threshold among candidate thresholds.
- Extra Trees generates random split thresholds and selects the best one among the generated candidates.

Therefore:

**Extra Trees = Random Feature Selection + Random Threshold Generation + Aggregation**

## 3. How Does Random Forest Work?

Suppose a dataset contains five features:

A, B, C, D, E.

### Step 1: Random Feature Selection

At each node, a random subset of features is selected.

For example:

A, B, C

### Step 2: Best Split Threshold Selection

For each selected feature, the algorithm evaluates candidate split thresholds.

Example:

| Feature | Candidate Thresholds |
|---|---|
| A | 10, 20, 30, 40 |
| B | 5, 15, 25, 35 |
| C | 12, 22, 32, 42 |

### Step 3: Impurity Calculation

The algorithm evaluates candidate splits using an impurity criterion such as Gini impurity or entropy.

### Step 4: Best Split Selection

The feature and threshold that provide the best impurity reduction are selected.

The process continues recursively for the child nodes.

## 4. How Does Extra Trees Work?

Extra Trees introduces additional randomness during threshold selection.

### Step 1: Random Feature Selection

At each node, a random subset of features is selected.

Example:

A, B, C

### Step 2: Random Threshold Generation

Instead of searching all candidate thresholds, Extra Trees generates random thresholds for the selected features.

Example:

| Feature | Randomly Generated Threshold |
|---|---:|
| A | 27 |
| B | 18 |
| C | 35 |

### Step 3: Impurity Calculation

The algorithm calculates impurity reduction for the randomly generated candidate splits.

### Step 4: Best Split Selection

Among the randomly generated splits, the algorithm selects the one that provides the best impurity reduction.

The process continues recursively for the child nodes.

**Important:** Extra Trees does not randomly select the final split without evaluation. It evaluates the randomly generated candidates and selects the best one among them.

## 5. Random Forest vs Extra Trees

| Parameter | Random Forest | Extra Trees |
|---|---|---|
| Base estimator | Decision Trees | Decision Trees |
| Feature selection | Random subset at each node | Random subset at each node |
| Threshold selection | Searches for the best candidate threshold | Generates random candidate thresholds |
| Split evaluation | Evaluates candidate splits using impurity | Evaluates randomly generated splits using impurity |
| Final split | Best among evaluated candidates | Best among randomly generated candidates |
| Bootstrap sampling | True by default | False by default |
| Randomness | Comparatively less | Additional randomness through thresholds |
| Bias | Can be relatively lower | Can be relatively higher |
| Variance | Can be relatively higher | Can be relatively lower |

**Note:** These bias and variance comparisons describe typical tendencies, not guaranteed outcomes. Actual performance depends on the dataset and hyperparameters.

## 6. Why Do We Introduce Additional Randomness?

### Problem: Correlation Between Trees

Suppose a dataset contains some highly predictive features.

When multiple Decision Trees are trained, they may repeatedly select the same strong features at their root nodes and subsequent splits.

This can result in:

- Similar tree structures.
- Similar predictions.
- Similar errors.
- Higher correlation between individual trees.

When trees are highly correlated, averaging their predictions provides less variance reduction.

### How Does Random Forest Address This?

Random Forest randomly selects a subset of features at each node.

This gives different features opportunities to participate in the splitting process.

As a result, different trees may learn different patterns, reducing their correlation.

### How Does Extra Trees Improve This Further?

Extra Trees introduces additional randomness by generating random thresholds for the selected features.

Even when two trees consider the same feature, they may generate different thresholds.

This encourages greater diversity among trees.

Consequently, correlation between trees may decrease further.

## 7. High-Cardinality Feature Bias

High cardinality means that a feature contains a large number of unique values.

Some splitting criteria, particularly Information Gain, can favor high-cardinality features because they provide more possible partitions.

### Example

Suppose we want to predict whether a customer will default on a loan.

Our dataset contains:

| Customer ID | Education Level | Loan Default |
|---|---|---|
| C101 | Graduate | Yes |
| C102 | Graduate | No |
| C103 | Graduate | Yes |
| C104 | Postgraduate | No |
| C105 | Postgraduate | Yes |

Consider two features:

- Education Level: Only a few unique categories.
- Customer ID: Potentially a unique value for every customer.

Customer ID provides many possible partitions.

The Decision Tree may isolate individual customers and obtain high impurity reduction on the training data.

However, Customer ID may have no meaningful relationship with loan default.

This can cause the model to learn accidental patterns instead of meaningful relationships.

### Impact

1. High-cardinality features may receive an unfair advantage during split selection.
2. The model may learn random patterns or noise.
3. Overfitting can occur.
4. Generalization to unseen observations may become poor.

**Important distinction:**

High cardinality does not automatically mean a feature is useless.

For example, PIN code and city may have many unique values but can contain meaningful geographical information.

Also, high-cardinality bias and correlation between trees are two different problems, although they can sometimes be related.

Random feature selection does not guarantee that high-cardinality bias will disappear.

## 8. Bias-Variance Trade-off in Extra Trees

Extra Trees introduces additional randomness compared with Random Forest.

### Advantages from Additional Randomness

- Reduces similarity between individual trees.
- Can reduce correlation between trees.
- Can improve variance reduction through aggregation.
- Can improve model generalization.

### Possible Disadvantages from Additional Randomness

- Random thresholds may not always produce the best possible splits.
- Individual trees may become less accurate.
- Bias may increase.
- Excessive randomness may reduce predictive performance.

**Key takeaway:**

Extra Trees attempts to reduce variance by increasing randomness and diversity among trees, but this may introduce additional bias.

Whether Extra Trees performs better than Random Forest depends on the dataset.

## 9. Row Sampling in Extra Trees

Extra Trees differs from Random Forest in its default row sampling behavior.

- Random Forest uses bootstrap sampling by default.
- Extra Trees does not use bootstrap sampling by default.

When bootstrap sampling is disabled, each tree is trained using the complete training dataset, rather than a bootstrap sample.

However, Extra Trees can also use bootstrap sampling if explicitly enabled.

Therefore, bootstrap sampling is not the core difference between these algorithms.

The main difference is their split threshold selection strategy.

## 10. Advantages of Extra Trees

1. **Reduced Correlation:** Additional randomness can reduce correlation between individual trees.
2. **Variance Reduction:** Aggregating diverse trees can reduce ensemble variance.
3. **Faster Training:** Random threshold generation can reduce the computational cost of searching for optimal thresholds.
4. **Non-linear Relationships:** Decision Trees can capture complex non-linear patterns.
5. **Feature Interactions:** Trees can automatically capture interactions through successive splits.
6. **Minimal Preprocessing:** Feature scaling is generally not required.

## 11. Disadvantages of Extra Trees

1. **Potentially Higher Bias:** Random thresholds may not provide the best possible splits.
2. **No Guaranteed Improvement:** Extra randomness does not guarantee better performance than Random Forest.
3. **Computational Cost:** Training multiple trees requires additional computational resources compared with a single tree.
4. **Reduced Interpretability:** Understanding predictions from multiple trees is more difficult than interpreting a single Decision Tree.
5. **Hyperparameter Sensitivity:** Performance depends on parameters such as the number of trees and the number of features considered at each split.

## Advantages of Random Forest
- Reduces Overfitting: Averaging multiple Decision Trees helps reduce variance compared to a single tree.
- Handles Non-linearity: Captures complex, non-linear relationships in data.
- Captures Feature Interactions: Automatically learns interactions through successive splits.
- Robust to Noise: Aggregation makes predictions less sensitive to individual noisy observations.
- Handles High-Dimensional Data: Can work with many features, depending on dataset size and structure.
- Feature Importance: Provides impurity-based feature importance scores.
- Versatile: Supports both classification and regression tasks.
- Less Preprocessing: Generally does not require feature scaling or normalization.
## Disadvantages of Random Forest
- Computationally Expensive: Training multiple trees requires more time and resources.
- High Memory Usage: Stores multiple Decision Trees.
- Less Interpretable: Difficult to understand compared to a single Decision Tree.
- Poor Extrapolation: Regression predictions are generally limited to the range of target values observed during training.
- Not Ideal for High-Dimensional Sparse Data: Often less effective than linear models for sparse datasets such as TF-IDF text features.
- Slower Predictions: Predicting requires passing data through multiple trees.
- Large Model Size: Ensembles with many deep trees can become large.