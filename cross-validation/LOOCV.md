# Leave-One-Out Cross-Validation (LOOCV)

## 1. Definition

Leave-One-Out Cross-Validation (LOOCV) is a special case of K-Fold Cross-Validation in which the number of folds K is equal to the total number of samples in the dataset.

During each iteration, one sample is used for validation, and all the remaining samples are used for training.

The model is trained N times, where N is the total number of samples.

## 2. How It Works

Suppose we have a dataset containing 5 samples.

| Iteration | Training Samples | Validation Sample |
|---|---|---|
| 1 | 2, 3, 4, 5 | 1 |
| 2 | 1, 3, 4, 5 | 2 |
| 3 | 1, 2, 4, 5 | 3 |
| 4 | 1, 2, 3, 5 | 4 |
| 5 | 1, 2, 3, 4 | 5 |

During each iteration:
- One sample is held out for validation.
- The remaining N-1 samples are used for training.
- The held-out sample changes in the next iteration.

After N iterations, every sample has been used exactly once for validation.

## 3. Performance Calculation

Suppose we have 5 samples and the model obtains the following validation accuracy scores:

| Iteration | Accuracy |
|---|---:|
| 1 | 100% |
| 2 | 100% |
| 3 | 0% |
| 4 | 100% |
| 5 | 100% |

Mean LOOCV Accuracy:

`Mean Accuracy = (100 + 100 + 0 + 100 + 100) / 5`

`Mean Accuracy = 80%`

For classification, each iteration produces a prediction for one validation sample.

The mean of the individual validation accuracy scores is equivalent to the overall accuracy obtained from all N held-out predictions.

For regression, the validation errors can similarly be combined to calculate metrics such as mean squared error.

## 4. Why Do We Need LOOCV?

In a single train-validation split, a considerable portion of a small dataset may be used for validation, leaving fewer samples for training.

LOOCV allows the model to train on N-1 samples in every iteration, making maximum use of the available training data.

This is particularly useful when the dataset is very small.

## 5. When to Use LOOCV

LOOCV is useful when:

- The dataset is very small.
- We want to use almost all available samples for training in each iteration.
- The model can be trained quickly.
- We want to evaluate a model without reserving a large portion of the dataset for validation.

### Example

Suppose we have only 50 samples for a regression problem.

Using 5-Fold Cross-Validation, the model trains on 40 samples and validates on 10 samples in each iteration.

Using LOOCV, the model trains on 49 samples and validates on 1 sample in each iteration.

This allows the model to use more samples for training in every iteration.

## 6. Advantages

### 1. Maximum Training Data Per Iteration

The model uses N-1 samples for training in every iteration.

This is useful when the dataset is very small.

### 2. Every Sample Is Used for Validation

Each sample is held out exactly once, ensuring that every observation contributes to the validation process.

### 3. No Random Fold Partitioning

LOOCV systematically holds out each sample once rather than creating random groups of validation samples.

### 4. Useful for Small Datasets

It can be useful when reserving a large validation set would leave too little data for training.

## 7. Disadvantages

### 1. Computationally Expensive

The model must be trained N times.

For a dataset containing 10,000 samples, LOOCV requires 10,000 model training iterations.

This can be computationally expensive for complex models.

### 2. High Variance in Individual Validation Scores

Each validation fold contains only one sample.

Therefore, individual validation scores can vary significantly, particularly for classification.

For example, the accuracy of one iteration can only be 0% or 100%.

### 3. Not Always Better Than K-Fold

LOOCV uses more training samples per iteration, but this does not guarantee that it will produce a better performance estimate than 5-Fold or 10-Fold Cross-Validation.

The appropriate validation strategy depends on the dataset, model, and evaluation objective.

### 4. Does Not Automatically Prevent Data Leakage

LOOCV does not prevent data leakage caused by:
- Preprocessing performed on the entire dataset.
- Feature selection performed before cross-validation.
- The same entity appearing in training and validation.
- Information from future observations being used during training.

## 8. Important Considerations

- LOOCV is a special case of K-Fold Cross-Validation.
- K = N, where N is the total number of samples.
- Each iteration uses N-1 samples for training and 1 sample for validation.
- The model is trained N times.
- LOOCV can be computationally expensive for large datasets.
- It is commonly considered for small datasets.
- It does not automatically prevent data leakage.
- A separate test set may still be needed for final model evaluation.

## 9. Summary

Leave-One-Out Cross-Validation evaluates a model by holding out one sample at a time and training on all remaining samples.

It is useful for small datasets because it allows the model to use almost all available data for training in every iteration.

However, it can be computationally expensive and individual validation scores can be highly variable.

LOOCV should be selected based on dataset size, computational cost, and the evaluation objective rather than assuming it is always better than ordinary K-Fold Cross-Validation.

## 10. When to Prefer LOOCV Over K-Fold

LOOCV may be preferred over ordinary K-Fold Cross-Validation when:

- The dataset is very small.
- Maximizing the number of training samples in each iteration is important.
- Each observation is valuable and limited training data is available.
- The model is computationally inexpensive to train.

### When to Prefer K-Fold Over LOOCV

K-Fold is generally preferred when:

- The dataset contains a moderate or large number of samples.
- The model is computationally expensive to train.
- Training the model N times, as required by LOOCV, would be impractical.
- A practical balance between computational cost and performance estimation is required.

### Important Note

LOOCV is not always better than K-Fold simply because it uses more training samples in each iteration.

The choice depends on dataset size, model complexity, computational cost, and the evaluation objective.