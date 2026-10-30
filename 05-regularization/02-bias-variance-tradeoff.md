# Bias-Variance Tradeoff


## 1. Introduction

The Bias-Variance Tradeoff is an important concept in Machine Learning that explains how model complexity affects prediction error and generalization.

A machine learning model should:

- Learn the underlying relationship between input features and the target variable.
- Avoid learning random noise present in the training data.
- Generalize well to unseen data.

Two important sources of prediction error are:

- Bias
- Variance

The goal is to find a model that balances these two sources of error to achieve good generalization performance.

> **Important:** Bias and variance are theoretical properties of a learning procedure. They are not the same as training error and test error.

---

## 2. Bias

### Definition

Bias measures the systematic difference between the model's average prediction and the true underlying relationship.

A model with high bias makes overly strong or simplistic assumptions about the relationship between input features and the target variable.

### Formula

```text
Bias(f_hat(x)) = E_D[f_hat_D(x)] - f(x)
```

Where:

- `x` = a particular input.
- `f(x)` = true underlying relationship between input and target.
- `D` = a training dataset.
- `f_hat_D(x)` = prediction made by the model trained on dataset D.
- `E_D` = expectation over possible training datasets.

### Explanation

Imagine training a model using different training datasets collected from the same underlying population.

For a particular input `x`, the model may produce different predictions depending on the training dataset.

Bias measures how far the average of those predictions is from the true underlying value.

### Example

Suppose the true house price for a particular house is:

```text
True price = 50 lakh
```

After training models on different datasets, their predictions are:

```text
Prediction 1 = 30 lakh
Prediction 2 = 35 lakh
Prediction 3 = 40 lakh
```

Average prediction:

```text
Average prediction = (30 + 35 + 40) / 3
                   = 35 lakh
```

Bias:

```text
Bias = Average prediction - True value
     = 35 - 50
     = -15 lakh
```

The model's average prediction is 15 lakh below the true value.

This indicates systematic underestimation at that input.

### High Bias

A model has high bias when its predictions systematically fail to capture the underlying relationship.

Common causes include:

- Model is too simple.
- Important patterns are not captured.
- Model assumptions are too restrictive.
- Relevant features or transformations are missing.

High bias is commonly associated with underfitting.

---

## 3. Variance

### Definition

Variance measures how much the model's predictions change when it is trained on different datasets.

A model with high variance is highly sensitive to changes in the training data.

### Formula

```text
Variance(f_hat(x)) =
E_D[(f_hat_D(x) - E_D[f_hat_D(x)])^2]
```

Where:

- `f_hat_D(x)` = prediction from a model trained on dataset D.
- `E_D[f_hat_D(x)]` = average prediction across possible training datasets.
- `E_D` = expectation over possible training datasets.

### Explanation

Suppose we train multiple models using different training datasets.

If their predictions are very different for the same input, the learning procedure has high variance.

If their predictions remain similar, the learning procedure has low variance.

### Example

Suppose the true house price is 50 lakh.

Predictions from different trained models:

```text
Prediction 1 = 30 lakh
Prediction 2 = 50 lakh
Prediction 3 = 70 lakh
```

Average prediction:

```text
Average prediction = (30 + 50 + 70) / 3
                   = 50 lakh
```

The average prediction is equal to the true value, so the bias is zero.

However, the predictions vary considerably across models.

This indicates high variance.

### High Variance

A model has high variance when its predictions are highly sensitive to the training dataset.

Common causes include:

- Excessive model complexity.
- Training on insufficient data.
- Learning noise in the training data.
- Using a flexible model without sufficient constraints.

High variance is commonly associated with overfitting.

---

## 4. Bias vs Variance

| Bias | Variance |
|---|---|
| Measures systematic prediction error. | Measures prediction variability across training datasets. |
| Related to overly simplistic assumptions. | Related to sensitivity to training data. |
| High bias is commonly associated with underfitting. | High variance is commonly associated with overfitting. |
| Model may fail to capture important patterns. | Model may learn patterns specific to the training data. |

### Important Difference

Bias and variance are not calculated using training error directly.

Training error measures the model's error on the dataset used for training.

Bias and variance describe theoretical behavior across possible training datasets.

---

## 5. Bias-Variance Decomposition

For a regression problem with squared-error loss, the expected prediction error can be decomposed into three components:

```text
Expected Prediction Error =
Bias^2 + Variance + Irreducible Error
```

Mathematically:

```text
E_D[(Y - f_hat_D(x))^2] =
Bias(f_hat(x))^2 +
Variance(f_hat(x)) +
sigma^2
```

Where:

- `Bias^2` = squared bias.
- `Variance` = prediction variance.
- `sigma^2` = irreducible error.

### Irreducible Error

Irreducible error is the uncertainty in the target variable that cannot be eliminated by the model using the available information.

For example:

House prices may depend on factors that are not present in the dataset, such as:

- Negotiation between buyer and seller.
- Unrecorded property conditions.
- Unexpected changes in market demand.

Even an ideal model cannot eliminate all uncertainty caused by such factors.

### Important Assumption

This decomposition applies to squared-error regression under the standard assumptions:

```text
Y = f(x) + epsilon
```

Where:

```text
E[epsilon | x] = 0
Var(epsilon | x) = sigma^2
```

This is not a universal decomposition for every classification loss or evaluation metric.

---

## 6. Numerical Example

Suppose the true underlying house price is:

```text
True price = 50 lakh
```

Different models trained on different datasets produce:

```text
Model 1 prediction = 30 lakh
Model 2 prediction = 35 lakh
Model 3 prediction = 40 lakh
```

### Step 1: Calculate Average Prediction

```text
Average prediction = (30 + 35 + 40) / 3

Average prediction = 35 lakh
```

### Step 2: Calculate Bias

```text
Bias = Average prediction - True value

Bias = 35 - 50

Bias = -15 lakh
```

### Step 3: Calculate Squared Bias

```text
Bias^2 = (-15)^2

Bias^2 = 225
```

### Step 4: Calculate Variance

```text
Variance =
[(30 - 35)^2 + (35 - 35)^2 + (40 - 35)^2] / 3
```

```text
Variance = [25 + 0 + 25] / 3

Variance = 16.67
```

### Interpretation

The models have:

- High bias because their average prediction is far from the true value.
- Low variance relative to the prediction errors shown because their predictions are close to each other.

This example demonstrates that bias and variance measure different things.

---

## 7. Connection with Underfitting and Overfitting

### Underfitting

Underfitting occurs when a model fails to learn the important patterns in the training data.

Typical characteristics:

- High training error.
- High validation or test error.
- Model is too simple or has overly restrictive assumptions.

Underfitting is commonly associated with high bias.

### Overfitting

Overfitting occurs when a model learns training-specific patterns, including noise, that do not generalize well to unseen data.

Typical characteristics:

- Low training error.
- High validation or test error.
- Model is overly sensitive to training data.

Overfitting is commonly associated with high variance.

### Comparison

| Underfitting | Overfitting |
|---|---|
| Model fails to learn important patterns. | Model learns training-specific patterns and noise. |
| High training error. | Low training error. |
| High validation error. | High validation error. |
| Commonly associated with high bias. | Commonly associated with high variance. |

---

## 8. Bias-Variance Tradeoff

As model complexity increases, the model becomes more flexible and can learn more complex relationships.

However, increasing model complexity can also make the model more sensitive to the training data.

In general:

- A very simple model may have high bias and low variance.
- A very complex model may have low bias and high variance.
- An appropriate model balances bias and variance.

### Conceptual Relationship

```text
Low Model Complexity
        |
        v
High Bias
Low Variance
        |
        v
Underfitting
```

```text
High Model Complexity
        |
        v
Low Bias
High Variance
        |
        v
Overfitting
```

The objective is not to minimize bias or variance independently.

The objective is to minimize expected prediction error and achieve good generalization.

---

## 9. Practical Interpretation in Machine Learning

In real-world machine learning, we generally do not calculate theoretical bias and variance directly.

The true underlying function is usually unknown, and we do not have access to every possible training dataset.

Instead, we use training and validation performance to diagnose model behavior.

### Case 1: High Training Error and High Validation Error

Possible interpretation:

- Model is underfitting.
- Model may be too simple.
- Features may not represent the underlying relationship adequately.

Possible actions:

- Increase model complexity.
- Add meaningful features.
- Use appropriate feature transformations.
- Reduce excessive regularization.

### Case 2: Low Training Error and High Validation Error

Possible interpretation:

- Model is overfitting.
- Model may be too complex.
- Training data may be insufficient or unrepresentative.

Possible actions:

- Collect more representative training data.
- Reduce model complexity.
- Apply regularization.
- Use cross-validation for model selection.
- Remove data leakage and evaluate using an appropriate split.

### Case 3: Low Training Error and Low Validation Error

Possible interpretation:

- Model is learning useful patterns.
- Model generalizes reasonably well to unseen data.

However, the final assessment should also consider:

- Appropriate test-set performance.
- Data leakage.
- Distribution shifts.
- The metric relevant to the actual business problem.

### Cross-Validation

Cross-validation helps estimate how model performance changes across different data splits.

It can provide useful evidence about model stability, but the variability of cross-validation scores is not exactly the same as the theoretical prediction variance defined in the bias-variance decomposition.

---

## 10. Key Takeaways

- Bias measures the systematic difference between average model predictions and the true underlying function.
- Variance measures how much model predictions change across different training datasets.
- High bias is commonly associated with underfitting.
- High variance is commonly associated with overfitting.
- Bias and variance are not the same as training error and validation error.
- The bias-variance decomposition explains expected squared prediction error as the sum of squared bias, variance, and irreducible error.
- Model complexity affects the balance between bias and variance.
- The practical objective is to achieve good generalization performance on unseen data.


