# K-Fold Cross-Validation

## 1. Definition

K-Fold Cross-Validation is a model evaluation technique in which the dataset is divided into K approximately equal-sized folds.

The model is trained K times. During each iteration, K-1 folds are used for training and the remaining fold is used for validation.

Each fold is used exactly once as the validation set. The final performance is typically calculated by averaging the validation scores obtained across all K iterations.

## 2. How It Works

Suppose we have a dataset containing 1,000 samples and choose K = 5.

The dataset is divided into five folds, each containing approximately 200 samples.

| Iteration | Training Data | Validation Data |
|---|---:|---:|
| 1 | 800 | 200 |
| 2 | 800 | 200 |
| 3 | 800 | 200 |
| 4 | 800 | 200 |
| 5 | 800 | 200 |

During each iteration:
- The model is trained using 800 samples.
- The model is evaluated using 200 validation samples.
- The validation fold changes in the next iteration.

After five iterations, each sample has been used once for validation and four times for training.

## 3. Performance Calculation

Suppose the model obtains the following accuracy scores:

| Fold | Accuracy |
|---|---:|
| 1 | 90% |
| 2 | 92% |
| 3 | 88% |
| 4 | 91% |
| 5 | 89% |

The mean cross-validation accuracy is:

`Mean CV Accuracy = (90 + 92 + 88 + 91 + 89) / 5`

`Mean CV Accuracy = 90%`

The standard deviation of the scores can also be calculated to understand how much the model's performance varies across different folds.

## 4. Why Do We Need K-Fold Cross-Validation?

A single train-validation split may not provide a reliable estimate of model performance because the performance can depend on how the dataset is split.

In one split, the training data may contain patterns similar to those in the validation data, resulting in good performance.

In another split, important patterns may be missing from the training data, resulting in poor performance.

K-Fold Cross-Validation evaluates the model across multiple splits, providing a more robust estimate of its performance and helping us understand its sensitivity to the data split.

## 5. When to Use K-Fold Cross-Validation

K-Fold is useful when:

- The dataset contains independent and identically distributed (i.i.d.) samples.
- We want a more reliable performance estimate than a single train-validation split.
- The dataset is not extremely large, and training the model multiple times is computationally feasible.
- We want to compare multiple models using the same validation strategy.
- We want to evaluate model performance while making efficient use of limited data.

### Example

Suppose we have a dataset containing 1,000 independent customer records.

Instead of using only one 80:20 split, we can use 5-Fold Cross-Validation to evaluate the model across five different train-validation splits.

This helps us understand whether the model consistently performs well or whether its performance depends heavily on a particular split.

## 6. Advantages

### 1. More Reliable Performance Estimate

The model is evaluated across multiple train-validation splits instead of relying on a single split.

### 2. Efficient Use of Data

Each sample is used for training in K-1 iterations and for validation in one iteration.

This is useful when the dataset is limited and we want to make efficient use of the available samples.

### 3. Helps Understand Model Stability

The scores across different folds help us understand how much the model's performance varies across different splits.

### 4. Useful for Model Comparison

Different models can be evaluated using the same folds, making their performance comparison more consistent.

## 7. Disadvantages

### 1. Computationally Expensive

The model needs to be trained K times.

For computationally expensive models, such as large neural networks, this can require significant time and resources.

### 2. Does Not Automatically Prevent Data Leakage

If preprocessing, feature selection, or other data-dependent operations are performed using the entire dataset before cross-validation, information from validation folds may leak into training.

### 3. Not Suitable for Every Dataset Structure

Ordinary K-Fold assumes that samples can be divided into folds without violating the evaluation objective.

It may not be appropriate for:
- Imbalanced classification problems.
- Grouped or entity-based datasets.
- Time-series datasets.

Specialized cross-validation strategies may be required for these situations.

### 4. Fold Scores May Vary

Even after using K-Fold, the performance scores across folds may differ.

This variation can occur because the samples in each fold are different.

## 8. Data Leakage in K-Fold Cross-Validation

Data leakage occurs when information that should not be available during model training is used, directly or indirectly, to train the model.

### Example: Same Entity in Training and Validation

Suppose a dataset contains multiple transactions from the same customer.

| Customer ID | Transaction | Target |
|---|---|---|
| C101 | T1 | 0 |
| C101 | T2 | 1 |
| C101 | T3 | 0 |
| C202 | T4 | 1 |
| C202 | T5 | 0 |

If ordinary K-Fold is applied, records belonging to customer C101 may appear in both the training and validation folds.

If the objective is to predict outcomes for completely unseen customers, this can cause data leakage or overly optimistic performance estimates.

The model may learn customer-specific patterns from the training records that also help it predict the validation records belonging to the same customer.

### Important Distinction

The presence of similar patterns in training and validation data is not automatically data leakage.

However, when the same entity appears in both sets, and the evaluation objective requires generalization to unseen entities, the split may not represent the real-world prediction scenario.

### Solution

Use Group K-Fold when samples belonging to the same entity must remain together in a single fold.

For example:
- Customer-based datasets
- Patient-based datasets
- User-based datasets
- Device-based datasets

Group K-Fold ensures that records belonging to the same group are not divided between training and validation within a fold.

## 9. Important Considerations

- K represents the number of folds.
- Each iteration uses K-1 folds for training and one fold for validation.
- Each sample appears in the validation set exactly once during a complete K-Fold cycle.
- Common choices are K = 5 and K = 10.
- Increasing K increases the number of model training iterations and computational cost.
- K-Fold does not automatically prevent data leakage.
- K-Fold is not the same as the final train-test evaluation.
- A separate test set can be kept untouched for final evaluation after model selection.

## 10. Summary

K-Fold Cross-Validation evaluates a model across multiple train-validation splits to obtain a more robust estimate of model performance.

It helps reduce dependence on a single split, makes efficient use of available data, and supports model comparison.

However, the cross-validation strategy must match the structure of the dataset and the real-world prediction objective to avoid misleading performance estimates.