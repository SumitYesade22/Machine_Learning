

## 1. Introduction

In Machine Learning, a model learns patterns and relationships from training data.

The objective is not simply to achieve low error on the training dataset. The model should also perform well on unseen data.

The ability of a model to perform well on unseen data is known as **generalization**.

Two common problems that affect a model's generalization are:

- Underfitting
- Overfitting

These problems occur when the model's complexity is not appropriate for the underlying patterns in the data.

## 2. Underfitting

Underfitting occurs when a model is too simple to capture the underlying patterns and relationships present in the data.

The model fails to learn important information from the training dataset.

### Characteristics

- High training error
- High testing error
- Important patterns remain unexplained
- Generally associated with high bias

### Example

Suppose the actual relationship between a feature and the target is nonlinear:

`y = x²`

However, we use a simple Linear Regression model:

`y = β₀ + β₁x`

Since Linear Regression can only learn a straight-line relationship, it cannot properly capture the nonlinear pattern.

Consequently, the model performs poorly on both training and testing data.

This is an example of underfitting.

## 3. Overfitting

Overfitting occurs when a model learns the training data too closely, including noise, random fluctuations, and patterns that do not generalize to the population.

The model performs very well on the training dataset but poorly on unseen data.

### Characteristics

- Very low training error
- High testing error
- Model learns noise or random patterns
- Generally associated with high variance

### Example

Suppose the actual relationship is:

`y = x²`

But the training dataset contains some random noise.

If we fit a very high-degree polynomial, the model may create a complex curve that passes close to every training observation.

The model learns both:
- The actual underlying relationship
- Random noise in the training data

When new observations are provided, the learned curve may produce inaccurate predictions.

This is an example of overfitting.

## 4. Causes of Underfitting

### 4.1 Model is too simple

The model does not have sufficient complexity to learn the actual relationship.

Example:

- Actual relationship: nonlinear
- Model used: Linear Regression

The model cannot capture the nonlinear pattern.

### 4.2 Important features are missing

If important predictors are not provided to the model, it may fail to learn the relationship accurately.

Example:

Predicting house prices using only Area while ignoring important factors such as Location and Age.

### 4.3 Excessive constraints

If the model is constrained too much, it may not be able to learn the underlying patterns.

For example, excessive regularization can shrink coefficients too strongly, causing the model to underfit.

### 4.4 Insufficient training

Some learning algorithms may underfit if they have not been trained sufficiently.

The model has not yet learned the important patterns present in the data.

## 5. Causes of Overfitting

### 5.1 Model is too complex

A highly complex model can fit the training data very closely.

It may learn patterns that are specific to the training dataset rather than the underlying population.

Example:

Using a very high-degree polynomial for a relatively simple relationship.

### 5.2 Too many features compared to available data

When a dataset contains many predictors but relatively few observations, the model may have many possible ways to explain the training data.

For example:

- Training observations: 100
- Features: 90

A model with many parameters may fit random variations in the training data.

Additional features provide more possible patterns for the model to learn. Some of these patterns may be accidental rather than meaningful.

This increases the risk of overfitting, especially when the model is highly flexible.

### 5.3 Insufficient training data

A small training dataset may not adequately represent the patterns present in the population.

The model may:

- Learn from limited examples
- Become overly influenced by unusual observations
- Fail to distinguish genuine patterns from random fluctuations

A complex model may fit these limited examples very closely, but its learned patterns may not generalize to new observations.

### 5.4 Learning noise

Training data may contain noise due to random variation, measurement errors, or other factors.

A sufficiently complex model may learn this noise as if it were a genuine relationship.

This can result in low training error but poor testing performance.

## 6. Features vs Training Data

The number of features and the amount of training data both influence the risk of overfitting.

### Many features

More features can provide useful information, but they also increase the number of relationships the model can learn.

If many features are irrelevant or noisy, a complex model may learn accidental patterns.

### Less training data

With fewer observations, the model has less evidence about the actual underlying relationships.

Random variations in the training dataset can have a stronger influence on the model.

### Combined effect

A dataset with many features and few observations can be particularly challenging.

A complex model may fit the available data very closely without learning patterns that generalize well.

**Important:** Having many features or limited data does not automatically cause overfitting. The risk depends on the model's complexity, feature relevance, data quality, and the underlying problem.

## 7. Bias-Variance Connection

Underfitting and overfitting are commonly associated with the bias-variance tradeoff.

### High Bias

A model with high bias makes overly simple assumptions about the relationship between features and target.

It fails to capture important patterns.

This is generally associated with underfitting.

### High Variance

A model with high variance is highly sensitive to the training dataset.

Small changes in the training data can cause substantial changes in the learned model.

This is generally associated with overfitting.

## 8. Comparison

| Property | Underfitting | Good Fit | Overfitting |
|---|---|---|---|
| Model complexity | Too low | Appropriate | Too high |
| Training error | High | Low | Very low |
| Testing error | High | Low | High |
| Bias | Generally high | Balanced | Generally low |
| Variance | Generally low | Balanced | Generally high |
| Generalization | Poor | Good | Poor |

## 9. Connection with Regularization

Regularization is a technique used to control model complexity.

It adds a penalty to the model's objective function, discouraging unnecessarily large coefficients.

This can help reduce overfitting.

### Ridge Regression

Ridge adds an L2 penalty:

`Cost = RSS + α × Σβⱼ²`

It penalizes large coefficients and shrinks them toward zero.

### Lasso Regression

Lasso adds an L1 penalty:

`Cost = RSS + α × Σ|βⱼ|`

It can shrink some coefficients exactly to zero.

### Important

Regularization must be balanced.

- Too little regularization may allow overfitting.
- Too much regularization may cause underfitting.

The objective is to find an appropriate level of model complexity that generalizes well.

## 10. Key Takeaways

- Underfitting occurs when the model fails to learn important patterns.
- Overfitting occurs when the model learns training-specific patterns, including noise.
- A complex model is not automatically an overfit model.
- More features and less training data can increase the risk of overfitting.
- Training error alone is not sufficient to evaluate generalization.
- Regularization helps control model complexity and can reduce overfitting.
- The ultimate objective is to achieve good performance on unseen data.

## 11. How to Reduce Underfitting and Overfitting

### 11.1 How to Reduce Underfitting

Underfitting occurs when a model fails to learn important patterns from the training data.

It is commonly associated with high bias.

#### Techniques to Reduce Underfitting

**1. Increase Model Complexity**

Use a model that can capture more complex relationships.

Example:

- Linear Regression → Polynomial Regression
- Shallow Decision Tree → Deeper Decision Tree

**2. Add Relevant Features**

Include meaningful features that help the model learn the relationship between input and target variables.

Example:

Instead of using only `Area` to predict house prices, include:

- `Area`
- `Bedrooms`
- `Location`
- `Age_of_House`

**3. Perform Feature Engineering**

Create new features or transformations that help the model represent the underlying relationship.

Example:

```text
Area_squared = Area^2
```

This can help a regression model capture nonlinear patterns.

**4. Reduce Regularization**

Excessive regularization restricts model complexity and can prevent the model from learning important patterns.

Reducing regularization allows the model to fit the training data more closely.

Example:

```text
Ridge Regression:

Large alpha → Strong regularization
Small alpha → Weak regularization
```

**5. Train the Model Properly**

Ensure the model has sufficient training time or appropriate optimization settings.

For iterative models, insufficient training can prevent the model from learning the patterns present in the data.

---

### 11.2 How to Reduce Overfitting

Overfitting occurs when a model learns training-specific patterns, including noise, that do not generalize well to unseen data.

It is commonly associated with high variance.

#### Techniques to Reduce Overfitting

**1. Reduce Model Complexity**

Use a less complex model that is less likely to memorize training-specific patterns.

Examples:

- Reduce Decision Tree depth.
- Reduce the number of parameters.
- Use a simpler model when a complex model is unnecessary.

**2. Apply Regularization**

Regularization penalizes excessive model complexity to reduce overfitting.

Examples:

- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net

**3. Collect More Training Data**

More representative training data can help the model learn general patterns instead of memorizing a small training dataset.

However, the additional data must be relevant and representative of the problem.

**4. Perform Feature Selection**

Remove irrelevant or redundant features when they contribute to noise or unnecessary model complexity.

Feature selection should be performed carefully because removing useful features can cause underfitting.

**5. Use Cross-Validation**

Cross-validation helps evaluate how well the model performs across different training and validation splits.

It can help identify models that perform well only on a particular split.

**6. Use Early Stopping**

For models trained iteratively, stop training when validation performance stops improving.

This can prevent the model from continuing to fit training-specific patterns.

**7. Use Dropout**

In neural networks, dropout randomly disables some neurons during training.

This can reduce reliance on specific neurons and help improve generalization.

---

### 8.3 Summary

| Underfitting | Overfitting |
|---|---|
| Increase model complexity. | Reduce model complexity. |
| Add relevant features. | Perform feature selection when appropriate. |
| Perform feature engineering. | Apply regularization. |
| Reduce excessive regularization. | Collect more representative data. |
| Train the model properly. | Use cross-validation and early stopping. |

### Important Note

The techniques above are not interchangeable in every situation.

For example:

- Increasing model complexity may reduce underfitting but increase overfitting.
- Increasing regularization may reduce overfitting but cause underfitting if it is too strong.
- Removing features may reduce overfitting but can also remove important information.

The objective is to select a model that generalizes well to unseen data, not simply to minimize training error.
