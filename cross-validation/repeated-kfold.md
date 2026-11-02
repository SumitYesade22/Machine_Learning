# Repeated K-Fold Cross-Validation

## 1. What is Repeated K-Fold Cross-Validation?

Repeated K-Fold Cross-Validation is a cross-validation technique in which K-Fold Cross-Validation is performed multiple times using different train-validation splits.

In standard K-Fold Cross-Validation:
- The dataset is divided into K folds.
- The model is trained K times.
- Each fold is used once as the validation set.

In Repeated K-Fold Cross-Validation:
- The dataset is divided into K folds.
- K-Fold Cross-Validation is performed multiple times.
- Each repetition uses a different random partition of the dataset when shuffling is enabled.
- The scores from all folds and repetitions are aggregated.

The main purpose is to evaluate model performance across multiple different data partitions and reduce dependence on a single K-Fold partition.

---

## 2. How Does Repeated K-Fold Work?

Suppose we have a dataset containing 10 samples.

```text
Dataset = [A, B, C, D, E, F, G, H, I, J]
```

Let:

```text
K = 5
Number of repetitions = 3
```

In each repetition:

- The dataset is divided into 5 folds.
- The model is trained 5 times.
- Each fold is used once for validation.

### Repetition 1

The dataset is divided into 5 folds.

```text
Fold 1: [A, B]
Fold 2: [C, D]
Fold 3: [E, F]
Fold 4: [G, H]
Fold 5: [I, J]
```

The model is trained 5 times.

### Repetition 2

The dataset is shuffled and divided into 5 different folds.

```text
Fold 1: [A, D]
Fold 2: [B, F]
Fold 3: [C, H]
Fold 4: [E, J]
Fold 5: [G, I]
```

Again, the model is trained 5 times.

### Repetition 3

The dataset is shuffled and divided into 5 different folds.

```text
Fold 1: [B, C]
Fold 2: [A, G]
Fold 3: [D, I]
Fold 4: [E, H]
Fold 5: [F, J]
```

The model is trained 5 more times.

Therefore:

```text
Total number of model trainings = K × R
```

Where:

- `K` = Number of folds
- `R` = Number of repetitions

For this example:

```text
K = 5
R = 3

Total model trainings = 5 × 3 = 15
```

---

## 3. Why Do We Need Repeated K-Fold?

In standard K-Fold Cross-Validation, the model is evaluated using one particular partition of the dataset.

Even if we use K-Fold, the performance estimate can depend on how the samples are divided into folds.

For example:

```text
K-Fold partition 1 → Mean Accuracy = 91%
K-Fold partition 2 → Mean Accuracy = 89%
K-Fold partition 3 → Mean Accuracy = 93%
```

Different partitions may produce different performance estimates.

Repeated K-Fold allows us to evaluate the model using multiple different partitions.

This helps us understand how much the evaluation score changes across different train-validation splits.

### Main purpose

Repeated K-Fold is useful when we want to:

- Evaluate the model across multiple random partitions.
- Reduce dependence on one particular K-Fold partition.
- Obtain multiple performance scores.
- Analyze the variability of model performance.

---

## 4. Number of Iterations

The total number of iterations is:

```text
Total iterations = K × R
```

Where:

- `K` = Number of folds
- `R` = Number of repetitions

### Example

Suppose:

```text
K = 5
R = 10
```

Then:

```text
Total iterations = 5 × 10 = 50
```

The model will be trained 50 times.

### Different configurations

| K | Repetitions (R) | Total iterations |
|---:|---:|---:|
| 5 | 1 | 5 |
| 5 | 3 | 15 |
| 5 | 10 | 50 |
| 10 | 5 | 50 |
| 10 | 10 | 100 |

Increasing K or R increases computational cost.

---

## 5. Advantages of Repeated K-Fold

### 1. More Robust Evaluation

The model is evaluated across multiple train-validation partitions instead of relying on a single K-Fold partition.

### 2. Reduced Dependence on a Single Split

The performance estimate is less dependent on one particular random partition.

### 3. Performance Variability Analysis

We obtain multiple scores, which help us understand how much the model's performance varies across different splits.

### 4. Useful for Model Comparison

Repeated K-Fold can be used to compare different models across multiple partitions.

For example:

```text
Model A → Repeated K-Fold Mean Accuracy = 90%
Model B → Repeated K-Fold Mean Accuracy = 93%
```

We can compare their average performance, while also examining the variability of their scores.

---

## 6. Disadvantages of Repeated K-Fold

### 1. Computationally Expensive

The model is trained K × R times.

For example:

```text
K = 10
R = 10

Total trainings = 100
```

This can become expensive for large datasets or complex models.

### 2. Increased Execution Time

More repetitions require more model training and evaluation.

### 3. Not a Guarantee of Better Performance

Repeated K-Fold does not improve the model itself.

It improves the evaluation process by allowing us to examine performance across multiple partitions.

### 4. Does Not Automatically Prevent Data Leakage

Preprocessing must be performed correctly within each training fold.

Using a scikit-learn Pipeline helps prevent leakage during cross-validation.

---

## 7. When Should We Use Repeated K-Fold?

Repeated K-Fold can be useful when:

- We have a small or medium-sized dataset.
- Model evaluation changes noticeably across different data splits.
- We want to compare models across multiple partitions.
- We want to analyze performance variability.

### When Should We Avoid It?

Avoid or limit repetitions when:

- The dataset is very large.
- Model training is computationally expensive.
- Standard K-Fold already provides a sufficient evaluation for the task.
- The data has group or temporal dependencies that ordinary K-Fold does not respect.

For classification tasks with imbalanced target classes, Repeated Stratified K-Fold may be more appropriate.

For grouped data, use an appropriate group-based cross-validation method.

For time-dependent data, use a time-aware validation strategy.

---

## 8. Repeated K-Fold vs K-Fold

| Feature | K-Fold | Repeated K-Fold |
|---|---|---|
| Number of partitions | One | Multiple |
| Model trainings | K | K × R |
| Different random partitions | Not necessarily | Yes, when shuffling is enabled |
| Performance scores | K | K × R |
| Computational cost | Lower | Higher |
| Main purpose | Evaluate model using K folds | Evaluate across multiple K-Fold partitions |

---

## 9. Important Points to Remember

- Repeated K-Fold performs K-Fold Cross-Validation multiple times.
- Each repetition uses a different random partition when shuffling is enabled.
- Total iterations = K × R.
- It evaluates the model across multiple train-validation partitions.
- It helps analyze the variability of model performance.
- It does not directly improve model training or guarantee better accuracy.
- For imbalanced classification, Repeated Stratified K-Fold is often more appropriate.
- For grouped or time-dependent data, use a suitable group-aware or time-aware CV strategy.

---

