# Cross-Validation

## 1. What is Cross-Validation?

Cross-validation (CV) is a model evaluation technique used to estimate how well a machine learning model generalizes to unseen data.

It involves splitting the available dataset into multiple subsets and repeatedly training and evaluating the model on different subsets.

The main objective is to obtain a more reliable estimate of model performance than relying on a single train-validation split.

---

## 2. Why Do We Need Cross-Validation?

A single train-validation split may not provide a reliable estimate of a model's performance because the model's performance can be sensitive to how the data is split. In some splits, the training data may contain patterns or representations similar to those in the validation data, resulting in good performance. In other splits, important patterns may be missing from the training data, resulting in poor performance.

Cross-validation addresses this issue by evaluating the model across multiple train-validation splits, providing a more robust estimate of its performance and helping us understand how sensitive the model is to the data split.

### 2.1 To Estimate Generalization Performance

A machine learning model learns patterns from training data. However, performing well on training data does not guarantee that the model will perform well on unseen data.

Cross-validation helps estimate how well the model may perform on unseen samples drawn from a similar data-generating process.

This is called **generalization performance**.

### 2.2 To Reduce Dependence on a Single Data Split

Consider a dataset containing 1,000 samples.

Using a single train-test split:

- 800 samples are used for training.
- 200 samples are used for testing.

The model's performance may depend on which 200 samples were selected for testing.

For example:

| Test split | Accuracy |
|---|---:|
| Split 1 | 90% |
| Split 2 | 82% |
| Split 3 | 95% |

Different splits can produce different scores because the samples in each split may differ in difficulty or representativeness.

Cross-validation evaluates the model across multiple validation subsets, reducing reliance on one particular split.

### 2.3 To Compare Different Machine Learning Models

Suppose we want to compare:

- Logistic Regression
- Naive Bayes
- Random Forest

Evaluating each model on a single split may produce misleading results.

Cross-validation allows us to evaluate the models across the same validation folds and compare their average performance.

### 2.4 To Perform Hyperparameter Tuning

Models have hyperparameters that affect their behavior.

Examples:

- Logistic Regression: `C`
- KNN: `n_neighbors`
- Random Forest: `max_depth`
- DBSCAN: `eps`, `min_samples`

Cross-validation can evaluate different hyperparameter configurations and help select the configuration with the best validation performance.

### 2.5 To Use Limited Data More Efficiently

When a dataset is small, reserving a large portion of the data for a single validation set reduces the amount available for training.

Cross-validation allows each sample to be used for validation once and for training in the remaining iterations.

This is particularly useful when the dataset is small.

---

## 3. Cross-Validation vs Train-Test Split

### Train-Test Split

The dataset is divided into two subsets:

- Training set: Used to train the model.
- Test set: Used to evaluate the trained model.

A single split can produce a performance estimate that depends on the specific samples selected.

### Cross-Validation

The dataset is divided into multiple folds.

The model is trained and evaluated multiple times, using a different fold for validation in each iteration.

### Comparison

| Train-Test Split | Cross-Validation |
|---|---|
| Usually one training and one test split | Multiple training and validation splits |
| Model is evaluated once | Model is evaluated multiple times |
| Faster computationally | More computationally expensive |
| Score may depend on a particular split | Provides performance across multiple splits |
| Commonly used for final evaluation | Commonly used for model selection and hyperparameter tuning |

**Important:** Cross-validation does not replace the final test set when an independent final evaluation is required.

---

## 4. Generalization, Underfitting, and Overfitting

Cross-validation helps evaluate a model's performance across different validation folds.

### Underfitting

The model fails to learn the important patterns in the data.

Typical behavior:

- Training performance is poor.
- Validation performance is poor.

### Overfitting

The model learns training-specific patterns that do not generalize well.

Typical behavior:

- Training performance is very high.
- Validation performance is substantially lower.

### Good Generalization

The model learns useful patterns that perform well on unseen data.

Typical behavior:

- Training performance is good.
- Validation performance is also good.
- The difference between them is reasonably small.

Cross-validation can help identify these patterns, but it does not automatically solve underfitting or overfitting.

---

# Type 1: K-Fold Cross-Validation

## 5. What is K-Fold Cross-Validation?

K-Fold Cross-Validation is a model evaluation technique that divides a dataset into K approximately equal-sized subsets called folds.

In each iteration:

- K - 1 folds are used for training.
- The remaining fold is used for validation.

The process is repeated K times so that every fold is used once for validation.

The final CV score is commonly calculated as the average of the K validation scores.

---

## 6. How K-Fold Works

Suppose we have a dataset containing 1,000 samples.

We select:

`K = 5`

The dataset is divided into five folds:

- Fold 1: 200 samples
- Fold 2: 200 samples
- Fold 3: 200 samples
- Fold 4: 200 samples
- Fold 5: 200 samples

### Iteration 1

Training: Folds 2, 3, 4, 5

Validation: Fold 1

### Iteration 2

Training: Folds 1, 3, 4, 5

Validation: Fold 2

### Iteration 3

Training: Folds 1, 2, 4, 5

Validation: Fold 3

### Iteration 4

Training: Folds 1, 2, 3, 5

Validation: Fold 4

### Iteration 5

Training: Folds 1, 2, 3, 4

Validation: Fold 5

Every sample is used:

- Once for validation.
- In the training set of the remaining iterations.

---

## 7. Example of K-Fold Evaluation

Suppose a classification model produces the following validation accuracies:

| Fold | Accuracy |
|---|---:|
| 1 | 85% |
| 2 | 90% |
| 3 | 80% |
| 4 | 95% |
| 5 | 85% |

Average CV accuracy:

`(85 + 90 + 80 + 95 + 85) / 5 = 87%`

The average score summarizes the model's performance across the five validation folds.

It is an estimate of generalization performance, not a guarantee of the exact accuracy on future real-world data.

---

## 8. Assumptions and Important Considerations

### 8.1 Independent Observations

The observations should generally be independent of one another for ordinary K-Fold evaluation to provide a meaningful estimate.

Independence means that one observation does not contain information that directly links it to another observation in a way that causes leakage across training and validation sets.

#### Example: Student Records

Suppose a dataset contains multiple records for the same student.

If some records of Student A appear in training and other records of Student A appear in validation, the model may learn student-specific patterns during training.

This can make validation performance appear better than the performance on genuinely unseen students.

In such situations, Group K-Fold may be more appropriate.

### 8.2 Similar Data-Generating Process

Ordinary K-Fold is generally suitable when observations come from a similar data-generating process.

If the data has a special structure, such as time dependency or groups, an appropriate specialized cross-validation method should be selected.

Examples:

- Time-dependent observations: Time Series Split
- Multiple observations per customer: Group K-Fold
- Imbalanced classification: Stratified K-Fold

### 8.3 Class Distribution Is Not Guaranteed to Be Preserved

Normal K-Fold does not explicitly preserve class proportions.

For example, suppose the complete classification dataset contains:

- 80% Pass
- 20% Fail

The folds may contain different class proportions.

Stratified K-Fold is designed to preserve approximately the same class proportions in each fold.

---

## 9. Advantages of K-Fold Cross-Validation

1. Evaluates the model across multiple validation subsets.
2. Reduces dependence on a single train-validation split.
3. Supports model comparison.
4. Supports hyperparameter tuning.
5. Uses available data efficiently for training and validation.
6. Can be useful for evaluating models on relatively small datasets.

## 10. Disadvantages of K-Fold Cross-Validation

1. Computationally expensive because the model is trained multiple times.
2. Does not guarantee class balance in classification folds.
3. May produce misleading evaluation when dependent observations are split across training and validation.
4. Not appropriate for ordinary random splitting of time-dependent data.
5. The CV score is still an estimate and may differ from actual future performance.

---

## 11. When to Use K-Fold Cross-Validation?

K-Fold is commonly used for:

- Standard regression problems.
- Classification problems where preserving class proportions is not essential.
- Model comparison.
- Hyperparameter tuning.
- Estimating model performance across multiple data splits.

For imbalanced classification, consider Stratified K-Fold.

For time series data, consider Time Series Split.

For grouped observations, consider Group K-Fold.

---

