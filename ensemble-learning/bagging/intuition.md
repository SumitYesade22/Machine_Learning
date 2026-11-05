# Bagging (Bootstrap Aggregating)

## 1. What is Ensemble Learning?

Ensemble Learning is a machine learning technique in which multiple models (base learners) are combined to produce a better overall prediction than an individual model.

The main idea is that combining multiple models can improve generalization by reducing prediction errors.

Different ensemble techniques include:

- Bagging
- Boosting
- Voting
- Stacking

## 2. What is Bagging?

Bagging stands for **Bootstrap Aggregating**.

It is an ensemble learning technique in which multiple models are trained independently on different bootstrap samples of the original training dataset. Their predictions are then aggregated to produce the final prediction.

Bagging primarily aims to **reduce variance and improve model stability**.

### Why Do We Use Bagging?

Consider a Decision Tree trained on a dataset.

A deep Decision Tree can have:

- Low bias: It can learn complex patterns in the training data.
- High variance: Small changes in the training data can produce significantly different trees and predictions.

This happens because Decision Trees are sensitive to changes in their training data.

For example:

- Tree 1 may select Feature A as its root node.
- Tree 2, trained on slightly different data, may select Feature B.
- Tree 3 may generate a completely different structure.

Consequently, an individual deep tree may fit its training dataset very well but produce unstable predictions on unseen data.

**Bagging addresses this problem by training multiple models on different bootstrap samples and aggregating their predictions.**

The random variations in individual models can partially cancel each other during aggregation.

As a result:
- Overall variance generally decreases.
- Model predictions become more stable.
- Generalization performance may improve.
- Bias generally remains similar, although it can sometimes change.

**Important:** Bagging primarily reduces variance. It does not guarantee a reduction in bias.

## 3. How Does Bagging Work?

Bagging consists of three main steps:

### Step 1: Bootstrap Sampling

Bootstrap sampling means creating multiple samples from the original training dataset by sampling **with replacement**.

Suppose the original dataset contains 1,000 rows.

We create multiple bootstrap samples, each generally containing 1,000 rows.

Because sampling is performed with replacement:

- The same observation can appear multiple times in a bootstrap sample.
- Some original observations may not appear in a particular sample.
- Different bootstrap samples contain different combinations of observations.

For example:

Original dataset:

`[A, B, C, D, E]`

Bootstrap Sample 1:

`[A, B, B, D, E]`

Bootstrap Sample 2:

`[A, A, C, D, D]`

Bootstrap Sample 3:

`[B, C, C, E, E]`

Each sample has five observations, but their compositions differ.

### Step 2: Train Multiple Models

Train a separate base model on each bootstrap sample.

For example:

- Bootstrap Sample 1 → Decision Tree 1
- Bootstrap Sample 2 → Decision Tree 2
- Bootstrap Sample 3 → Decision Tree 3

These models are trained independently and can be trained in parallel.

### Step 3: Aggregate Predictions

Once all models are trained, combine their predictions.

**For Regression:**

Take the average of all model predictions.

Prediction = (T1 + T2 + ... + TM) / M

**For Classification:**

Use majority voting to determine the final predicted class.

Alternatively, average the predicted class probabilities and select the class with the highest average probability.

## 4. Why Does Bagging Reduce Variance?

To understand this, consider an individual Decision Tree.

Its prediction can change significantly when the training dataset changes.

Suppose we train five Decision Trees using different bootstrap samples. They may produce slightly different predictions because each tree learns from a different sample.

Instead of relying on one tree, Bagging averages their predictions.

Individual fluctuations can partially cancel each other during averaging.

This makes the final prediction more stable.

### Important Role of Correlation

Variance reduction depends not only on the number of models but also on how correlated their predictions are.

- Highly correlated models make similar predictions and similar errors.
- Less correlated models have more independent variations, allowing better cancellation during aggregation.

Therefore, lower correlation generally leads to greater variance reduction.

If all models are perfectly correlated, averaging them does not reduce variance.

## 5. Mathematical Understanding of Variance Reduction

Let:

- M = Number of models
- T1, T2, ..., TM = Predictions from individual models
- T_bar = Average prediction of the ensemble
- σ² = Variance of each individual model
- ρ = Pairwise correlation between models

The ensemble prediction for regression is:

T_bar = (T1 + T2 + ... + TM) / M

### General Variance Formula

The general formula is:

Var(T_bar) = (1/M²) [Σ Var(Ti) + 2Σ Cov(Ti,Tj)]

Here, the second summation includes every unique pair of models (i < j).

This formula does not assume that all models have equal variance or equal correlation.

### Simplified Formula

For easier interpretation, assume:

1. All individual models have equal variance σ².
2. Every pair of models has the same correlation ρ.

Under these assumptions:

Var(T_bar) = σ² [ρ + (1 - ρ)/M]

This formula explains how individual model variance, correlation, and number of models affect ensemble variance.

## 6. Interpretation of the Variance Formula

### A. Individual Model Variance (σ²)

- Higher individual variance → Higher ensemble variance.
- Lower individual variance → Lower ensemble variance.

**Why?**

If individual models produce highly variable predictions across different training datasets, averaging cannot eliminate all that variability.

Example:

Assume:
- Number of models (M) = 10
- Correlation (ρ) = 0.2

| Individual Variance (σ²) | Ensemble Variance |
|---:|---:|
| 50 | 14 |
| 100 | 28 |
| 200 | 56 |

As individual model variance increases, ensemble variance also increases.

### B. Correlation Between Models (ρ)

- Higher correlation → Higher ensemble variance.
- Lower correlation → Lower ensemble variance.

**Why?**

When models are highly correlated, they tend to make similar predictions and similar errors.

Averaging their predictions does not cancel out much of their variation.

When models are less correlated, their prediction fluctuations can partially cancel each other.

Example:

Assume:
- Individual variance (σ²) = 100
- Number of models (M) = 10

| Correlation (ρ) | Ensemble Variance |
|---:|---:|
| 0 | 10 |
| 0.2 | 28 |
| 0.5 | 55 |
| 0.8 | 82 |
| 1 | 100 |

Lower correlation results in greater variance reduction.

### C. Number of Models (M)

- Increasing the number of models → Ensemble variance decreases or remains unchanged.
- Decreasing the number of models → Ensemble variance increases or remains unchanged.

**Why?**

Averaging more models helps cancel out individual random fluctuations.

However, the benefit becomes smaller as the number of models increases.

Example:

Assume:
- Individual variance (σ²) = 100
- Correlation (ρ) = 0.2

| Number of Models (M) | Ensemble Variance |
|---:|---:|
| 1 | 100 |
| 2 | 60 |
| 5 | 36 |
| 10 | 28 |
| 20 | 24 |
| 100 | 20.8 |

As the number of models approaches infinity, ensemble variance approaches:

σ² × ρ

Therefore, adding more models cannot eliminate the variance caused by their shared correlation.

### Summary of Variance Trade-offs

| Factor Increases | Effect on Ensemble Variance |
|---|---|
| Individual variance (σ²) | Increases |
| Correlation (ρ) | Increases |
| Number of models (M) | Decreases or remains unchanged |

**Important:** The simplified formula assumes equal individual variances and equal pairwise correlations. Real models do not necessarily satisfy these assumptions.

## 7. Relationship Between Bagging and Bias-Variance Trade-off

Recall the bias-variance decomposition of expected prediction error:

Expected Error = Bias² + Variance + Irreducible Noise

Bagging primarily targets the variance component.

Consider a deep Decision Tree:

- It generally has low bias because it can learn complex patterns.
- It generally has high variance because it is sensitive to changes in training data.

After applying Bagging:

- Variance generally decreases.
- Bias often remains approximately similar, although it can change.
- Irreducible noise cannot be eliminated by Bagging.

Therefore, Bagging can improve generalization by reducing variance without necessarily increasing bias significantly.

**Key takeaway:** Bagging is particularly useful when the base learning algorithm is unstable and has high variance.

## 8. Advantages of Bagging

1. **Reduces Variance:** Aggregating predictions from multiple models helps reduce prediction variability.
2. **Improves Stability:** The final prediction is less dependent on a single training sample.
3. **Reduces Overfitting Risk:** Particularly useful with high-variance base learners such as deep Decision Trees.
4. **Supports Parallel Training:** Base models can be trained independently.
5. **Flexible:** Can be applied to different base learning algorithms.
6. **Improves Generalization:** Can improve performance on unseen data when variance reduction outweighs any bias increase.

## 9. Limitations of Bagging

1. **Computational Cost:** Training and maintaining multiple models requires more computational resources.
2. **Limited Bias Reduction:** Bagging primarily reduces variance and is not designed specifically to reduce high bias.
3. **Correlation Problem:** Highly correlated base models provide less variance reduction.
4. **Interpretability:** Understanding the combined predictions of multiple models is harder than interpreting a single Decision Tree.
5. **Not Always Better:** Bagging may provide little improvement when the base learner is already stable and has low variance.

## 10. Bagging vs Cross-Validation

These techniques serve different purposes.

| Bagging | Cross-Validation |
|---|---|
| Ensemble learning technique | Model evaluation technique |
| Trains multiple models on bootstrap samples | Trains models across different data folds |
| Aggregates predictions | Estimates generalization performance |
| Primarily reduces variance | Helps select models and hyperparameters |
| Can produce a final ensemble model | Usually does not produce an ensemble by itself |

**Important:** Bagging uses bootstrap sampling with replacement, whereas standard K-fold cross-validation divides data into non-overlapping validation folds.

## Sampling Techniques in Bagging

Bagging supports four sampling techniques based on how rows and features are selected.

| Technique | Row Sampling | Feature Sampling | `bootstrap` | `bootstrap_features` |
|---|---|---|---|---|
| **Bagging** | Subset of rows | All features | `True` | `False` |
| **Pasting** | Subset of rows | All features | `False` | `False` |
| **Random Subspaces** | All rows | Subset of features | `False` | `False` |
| **Random Patches** | Subset of rows | Subset of features | `False` | `False` |

**Scikit-learn parameters:**
- `max_samples`: Number or fraction of rows selected.
- `max_features`: Number or fraction of features selected.
- `bootstrap=True`: Row sampling with replacement.
- `bootstrap=False`: Row sampling without replacement.
- `bootstrap_features=True`: Feature sampling with replacement.
- `bootstrap_features=False`: Feature sampling without replacement.

**Note:** The table shows standard configurations. The `max_samples` and `max_features` parameters control the sampling quantity; their values are not fixed. Replacement settings determine the sampling method.