# Stratified K-Fold Cross-Validation

## 1. Definition

Stratified K-Fold Cross-Validation is a variation of K-Fold Cross-Validation primarily used for classification problems.

It divides the dataset into K approximately equal-sized folds while maintaining approximately the same proportion of target classes in each fold as in the original dataset.

The model is trained K times. During each iteration, K-1 folds are used for training and the remaining fold is used for validation.

Each fold is used exactly once as the validation set.

## 2. Why Do We Need Stratified K-Fold?

In classification problems, the target classes may not be equally represented.

For example, consider a fraud detection dataset:

| Class | Number of Samples | Percentage |
|---|---:|---:|
| Non-Fraud (0) | 950 | 95% |
| Fraud (1) | 50 | 5% |
| Total | 1000 | 100% |

Suppose we apply ordinary K-Fold with K = 5.

Since K-Fold does not explicitly preserve class proportions, some folds may contain fewer fraud samples than others.

In certain cases, a validation fold may contain very few or even no fraud samples.

This can make the evaluation unreliable because the model is not being evaluated on a representative class distribution.

Stratified K-Fold addresses this issue by maintaining approximately the same class proportions in every fold.

## 3. How It Works

Suppose we have:

- Total samples: 1000
- Non-Fraud samples: 950
- Fraud samples: 50
- Number of folds: K = 5

The original class proportions are:

- Non-Fraud: 95%
- Fraud: 5%

Stratified K-Fold attempts to maintain these proportions in every fold.

| Fold | Non-Fraud | Fraud | Total |
|---|---:|---:|---:|
| Fold 1 | 190 | 10 | 200 |
| Fold 2 | 190 | 10 | 200 |
| Fold 3 | 190 | 10 | 200 |
| Fold 4 | 190 | 10 | 200 |
| Fold 5 | 190 | 10 | 200 |

During each iteration:

- 4 folds are used for training.
- 1 fold is used for validation.
- The validation fold contains approximately 95% non-fraud and 5% fraud samples.

This helps ensure that each validation fold represents the original target class distribution.

## 4. Performance Calculation

Suppose the model obtains the following accuracy scores:

| Fold | Accuracy |
|---|---:|
| 1 | 94% |
| 2 | 96% |
| 3 | 95% |
| 4 | 93% |
| 5 | 97% |

Mean CV Accuracy:

`(94 + 96 + 95 + 93 + 97) / 5`

`Mean CV Accuracy = 95%`

The mean score provides an estimate of the model's performance across the different validation folds.

However, for imbalanced classification problems, accuracy alone may not be sufficient.

Metrics such as precision, recall, F1-score, and ROC-AUC may also be useful depending on the problem.

## 5. When to Use Stratified K-Fold

Stratified K-Fold is useful when:

- We are solving a classification problem.
- The target classes are imbalanced.
- We want to maintain approximately the same target class proportions in each fold.
- We want a more reliable evaluation across multiple train-validation splits.
- The dataset contains independent samples and does not require group-based or time-based splitting.

### Example

Suppose we are building a fraud detection model.

Only 5% of the transactions are fraudulent.

If the validation folds contain different proportions of fraudulent transactions, the evaluation scores may vary significantly depending on the split.

Stratified K-Fold helps maintain the class proportions across folds, making the evaluation more consistent.

## 6. Advantages

### 1. Maintains Class Proportions

Each fold contains approximately the same class distribution as the original dataset.

### 2. More Representative Validation Folds

The validation folds are less likely to have an unusual target class distribution.

### 3. Useful for Imbalanced Classification

It is particularly useful when one class has significantly fewer samples than another.

### 4. Supports Model Comparison

Different classification models can be evaluated using the same stratified folds, making comparison more consistent.

### 5. Efficient Use of Data

Each sample is used for validation once and for training K-1 times.

## 7. Disadvantages

### 1. Computationally Expensive

The model needs to be trained K times.

### 2. Does Not Solve Class Imbalance

Stratification preserves the class proportions in each fold, but it does not increase the number of minority-class samples.

For example, if only 1% of the dataset belongs to the minority class, stratification does not make that class more common in the original dataset.

### 3. Does Not Guarantee Data Leakage Prevention

Stratified K-Fold only maintains the target class proportions.

It does not prevent:
- The same entity appearing in both training and validation.
- Preprocessing leakage.
- Feature selection leakage.
- Time-related leakage.

### 4. Not Suitable for Every Dataset Structure

Stratified K-Fold may not be appropriate when:
- The same entity has multiple related records.
- The data has a time-dependent structure.
- The validation objective requires unseen groups or future observations.

In such cases, other strategies may be required.

## 8. Data Leakage Considerations

Stratified K-Fold maintains target class proportions, but it does not automatically prevent data leakage.

### Example: Same Customer in Training and Validation

Suppose a dataset contains multiple transactions from the same customer.

Ordinary Stratified K-Fold may place some transactions from Customer C101 in the training set and other transactions from C101 in the validation set.

If the goal is to predict outcomes for completely unseen customers, the model may learn customer-specific patterns from training records that also help it predict validation records.

This may result in overly optimistic performance.

### Solution

Use Stratified Group K-Fold when:

- The target is categorical.
- The dataset contains multiple records belonging to the same entity.
- The same entity must not appear in both training and validation.
- We also want to maintain approximately the same target class proportions across folds.

For example:
- Customer-based classification
- Patient-based classification
- User-based classification

## 9. Important Considerations

- Stratified K-Fold is primarily used for classification.
- It preserves approximately the same target class proportions in every fold.
- It does not preserve the distribution of every input feature.
- It does not guarantee exact class proportions in every fold.
- It does not solve class imbalance.
- It does not automatically prevent data leakage.
- The number of splits must be compatible with the number of samples in each class.
- For grouped data, Stratified Group K-Fold may be more appropriate.
- For time-dependent data, time-based validation may be more appropriate.

## 10. Summary

Stratified K-Fold Cross-Validation is a variation of K-Fold Cross-Validation that maintains approximately the same target class proportions in every fold.

It is particularly useful for classification problems where the target classes are imbalanced.

It helps produce more representative validation folds, but it does not solve class imbalance, prevent data leakage, or handle group and time dependencies automatically.