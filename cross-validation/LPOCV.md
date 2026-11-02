# Leave-P-Out Cross-Validation (LPOCV)

## 1. What is Leave-P-Out Cross-Validation?

Leave-P-Out Cross-Validation (LPOCV) is a cross-validation technique in which P samples are selected as the validation set, and the remaining N - P samples are used for training.

Where:
- `N` = Total number of samples in the dataset
- `P` = Number of samples used for validation in each iteration
- `N - P` = Number of samples used for training

The process is repeated for every possible combination of P validation samples.

The final performance is calculated by aggregating the scores obtained across all iterations.

---

## 2. How Does LPOCV Work?

Suppose we have a dataset containing 5 samples:

```text
Dataset = [A, B, C, D, E]
```

Let:

```text
N = 5
P = 2
```

In every iteration:
- 2 samples are used for validation.
- The remaining 3 samples are used for training.

### Iteration 1

```text
Training:   [C, D, E]
Validation: [A, B]
```

### Iteration 2

```text
Training:   [B, D, E]
Validation: [A, C]
```

### Iteration 3

```text
Training:   [B, C, E]
Validation: [A, D]
```

The process continues until every possible combination of 2 validation samples has been used.

---

## 3. Number of Iterations

The number of iterations is calculated using the combination formula:

Number of iterations = C(N, P)

Where:

```text
C(N, P) = N! / (P! * (N - P)!)
```

### Example

Suppose:

```text
N = 5
P = 2
```

Then:

```text
C(5, 2) = 5! / (2! * 3!)
          = 10
```

Therefore, LPOCV will train the model 10 times.

### Number of Iterations for Different Values

| N | P | Number of Iterations |
|---:|---:|---:|
| 5 | 1 | 5 |
| 5 | 2 | 10 |
| 5 | 3 | 10 |
| 10 | 2 | 45 |
| 10 | 3 | 120 |
| 20 | 2 | 190 |
| 20 | 5 | 15,504 |

The number of iterations can increase rapidly as N or P increases.

---

## 4. Relationship Between LPOCV and LOOCV

Leave-One-Out Cross-Validation (LOOCV) is a special case of LPOCV.

In LOOCV:

```text
P = 1
```

Only one sample is used for validation in each iteration.

For example:

```text
N = 5
P = 1
```

Number of iterations:

```text
C(5, 1) = 5
```

In LPOCV, we can choose:

```text
P = 1
P = 2
P = 3
...
```

Therefore:

**LOOCV is LPOCV with P = 1.**

---

## 5. Advantages of LPOCV

### 1. Multiple Validation Samples

Unlike LOOCV, where only one sample is used for validation per iteration, LPOCV allows multiple samples to be used for validation.

### 2. Extensive Evaluation

LPOCV evaluates the model on every possible combination of P validation samples.

### 3. Useful for Small Datasets

It can be useful when the dataset is small and we want to make extensive use of the available samples for model evaluation.

### 4. Flexible Validation Set Size

We can choose P according to the validation set size we want.

For example:

```text
P = 1 → 1 validation sample
P = 2 → 2 validation samples
P = 3 → 3 validation samples
```

---

## 6. Disadvantages of LPOCV

### 1. Computationally Expensive

The number of iterations is:

```text
C(N, P)
```

As the dataset size increases, the number of possible combinations can become very large.

### 2. Not Suitable for Large Datasets

For large datasets, LPOCV can require an impractical amount of computation.

### 3. Increased Execution Time

The model is trained repeatedly for every possible validation combination.

### 4. Does Not Automatically Prevent Data Leakage

LPOCV does not prevent data leakage by itself.

Preprocessing steps such as scaling, imputation, and feature selection must be performed correctly within each training fold.

Using a scikit-learn Pipeline helps prevent leakage during cross-validation.

---

## 7. When Should We Use LPOCV?

LPOCV can be considered when:

- The dataset is small.
- We want multiple validation samples per iteration.
- We can afford the computational cost.
- We want to explore different train-validation combinations.

### When Should We Avoid It?

Avoid LPOCV when:

- The dataset is large.
- The model is computationally expensive to train.
- The number of possible combinations becomes too high.

In such situations, K-Fold or Stratified K-Fold Cross-Validation is generally more practical.

---
## 8. LPOCV vs LOOCV vs K-Fold

| Feature | LPOCV | LOOCV | K-Fold |
|---|---|---|---|
| Validation samples per iteration | P | 1 | Approximately N/K |
| Number of iterations | C(N, P) | N | K |
| Training samples per iteration | N - P | N - 1 | Approximately N × (K - 1) / K |
| Computational cost | Can be very high | High for large N | Usually more manageable |
| Main advantage | Exhaustive combinations with multiple validation samples | Uses N - 1 samples for training | Good balance of computation and data usage |

---

## 9. Key Points to Remember

- LPOCV stands for Leave-P-Out Cross-Validation.
- P samples are used for validation in each iteration.
- N - P samples are used for training.
- Number of iterations = `comb(N, P)`.
- LOOCV is a special case of LPOCV where P = 1.
- LPOCV can be computationally expensive because of the large number of combinations.
- K-Fold is generally more practical for larger datasets.


