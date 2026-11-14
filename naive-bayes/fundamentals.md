# Naive Bayes – Fundamentals

## 1. Introduction to Naive Bayes

Naive Bayes is a supervised machine learning algorithm primarily used for classification tasks. It is based on **Bayes' Theorem** and makes an assumption that features are conditionally independent given the target class.

It is called:
- **Naive:** Because it assumes conditional independence between features.
- **Bayes:** Because it uses Bayes' Theorem to calculate probabilities.

### Common Applications

- Email Spam Detection
- Sentiment Analysis
- Document Classification
- Medical Diagnosis
- News Categorization

### Key Characteristics

- Probabilistic classification algorithm.
- Based on Bayes' Theorem.
- Assumes conditional independence between features.
- Computationally efficient.
- Works well with high-dimensional datasets, particularly text data.
- Requires relatively little training data compared with many other models.

---

## 2. Bayes' Theorem

Bayes' Theorem calculates the probability of an event based on previously available information.

### Mathematical Formula

`P(A|B) = [P(B|A) * P(A)] / P(B)`

Where:

- `P(A|B)` = Posterior Probability
- `P(B|A)` = Likelihood
- `P(A)` = Prior Probability
- `P(B)` = Evidence or Marginal Probability

### Intuition

Suppose we want to determine whether a student will pass an examination based on their study habits.

We already know:
- The probability of a student passing before observing their study habits.
- How likely regular studying is among students who pass.
- The probability of observing regular studying overall.

Using this information, we can update our belief about whether the student will pass.

This is the fundamental idea behind Bayes' Theorem.

---

## 3. The Naive Assumption – Conditional Independence

The most important assumption in Naive Bayes is that all features are conditionally independent given the target class.

Consider an email classification problem with three features:

- `X1`: Email contains "Free".
- `X2`: Email contains "Offer".
- `X3`: Email contains "Win".

Naive Bayes assumes that once the class (Spam or Ham) is known, the presence of one feature does not provide additional information about another feature.

### Mathematical Representation

`P(X1, X2, X3 | Y) = P(X1|Y) * P(X2|Y) * P(X3|Y)`

For `n` features:

`P(X1, X2, ..., Xn | Y) = Product(P(Xi|Y))`

This assumption simplifies the joint probability calculation significantly.

### Why Is It Called Naive?

In real-world datasets, features are often correlated.

For example:
- "Free" and "Offer" may frequently appear together in spam emails.
- Age and income may be correlated.
- Height and weight may be correlated.

Naive Bayes ignores these dependencies while estimating probabilities.

**Important:** Conditional independence is an assumption of the model, not a requirement that the original features must actually be independent.

---

## 4. Prior Probability

Prior probability represents the probability of a class before considering the features of a new observation.

For example, suppose a dataset contains:

- 100 Spam emails.
- 400 Ham emails.
- Total = 500 emails.

Then:

`P(Spam) = 100/500 = 0.20`

`P(Ham) = 400/500 = 0.80`

These are the prior probabilities.

### Key Points

- Calculated from class frequencies in the training dataset.
- Represents the initial probability of each class.
- Does not depend on the features of the new observation.

---

## 5. Likelihood

Likelihood measures how plausible the observed data is under a particular hypothesis or class.

In Naive Bayes, likelihood represents the probability of observing a feature given a particular class.

### Mathematical Representation

`P(X|Y)`

Where:
- `X` = Observed feature.
- `Y` = Target class.

### Example

Suppose we are classifying emails as Spam or Ham.

We want to calculate:

`P(Free = 1 | Spam)`

This represents the probability that the word "Free" is present in an email, given that the email belongs to the Spam class.

If 80 out of 100 Spam emails contain the word "Free":

`P(Free = 1 | Spam) = 80/100 = 0.80`

### Likelihood vs Probability

Likelihood and conditional probability have closely related mathematical expressions, but their interpretations differ.

- Conditional probability: We fix the class and calculate the probability of observing the feature.
- Likelihood: We use the observed data to evaluate how well a particular class or parameter value explains that data.

Likelihood is not the probability that a hypothesis is true.

---

## 6. Posterior Probability

Posterior probability represents the updated probability of a class after observing the features.

### Mathematical Representation

`P(Y|X)`

Where:
- `Y` = Target class.
- `X` = Observed features.

### Example

Suppose we receive a new email containing the word "Free".

We want to calculate:

`P(Spam | Free = 1)`

This represents the probability that the email belongs to the Spam class, given that the word "Free" is present.

The posterior probability combines:
- Prior probability.
- Likelihood.
- Evidence.

### Likelihood vs Posterior

| Likelihood | Posterior |
|---|---|
| P(X\|Y) | P(Y\|X) |
| Feature given class | Class given feature |
| Evaluates how well a class explains observed data | Represents updated class probability |
| Used in calculating posterior | Final probability of interest |

Both are conditional probabilities, but the conditioned event is different.

---

## 7. Evidence or Marginal Probability

Evidence represents the overall probability of observing the given features across all possible classes.

For a binary classification problem:

`P(X) = P(X|Spam) * P(Spam) + P(X|Ham) * P(Ham)`

It acts as the normalizing denominator in Bayes' Theorem.

### Why Is Evidence Important?

The numerator gives an unnormalized score for a class.

Evidence ensures that posterior probabilities are normalized and sum to 1 across all classes.

For multiple classes:

`P(X) = Sum(P(X|Y) * P(Y))`

The sum is calculated over all possible classes.

---

## 8. How Naive Bayes Makes Predictions

Suppose we want to classify a new observation using features X1, X2, and X3.

Using Bayes' Theorem and the conditional independence assumption:

`P(Y|X1,X2,X3) = P(Y) * P(X1|Y) * P(X2|Y) * P(X3|Y) / P(X1,X2,X3)`

Since the denominator is the same for every class, we can compare the unnormalized scores.

### Classification Score

`Score(Y) = P(Y) * Product(P(Xi|Y))`

The predicted class is:

`Predicted Class = argmax(Score(Y))`

In simple terms:

1. Calculate prior probabilities for every class.
2. Calculate feature likelihoods for every class.
3. Multiply the prior and feature likelihoods.
4. Compare the resulting class scores.
5. Select the class with the highest score.

For final posterior probabilities, normalize the class scores by their sum.

---

## 9. Numerical Instability – Underflow Problem

Naive Bayes multiplies several conditional probabilities to calculate class scores.

Since probabilities lie between 0 and 1, multiplying many small probabilities can produce extremely small numbers.

### Example

Suppose we have:

`0.1 * 0.01 * 0.001 * 0.0001`

As the number of features increases, the resulting value becomes increasingly small.

With hundreds or thousands of features, the result may become smaller than the minimum positive value that the computer can represent accurately.

The computer may then represent the result as zero.

This problem is known as **Numerical Underflow**.

### Why Is It a Problem?

If the calculated class scores become zero due to underflow, the model may fail to distinguish between classes correctly.

### Solution: Log Probability

Instead of multiplying probabilities directly, we take their logarithms.

Using the logarithm property:

`log(a * b) = log(a) + log(b)`

The Naive Bayes classification score becomes:

`Log Score(Y) = log(P(Y)) + Sum(log(P(Xi|Y)))`

### Advantages of Log Probability

- Converts multiplication into addition.
- Reduces numerical underflow.
- Makes calculations more numerically stable.
- Preserves class ranking because logarithm is a strictly increasing function.

---

## 10. Zero Probability Problem

Another problem in Naive Bayes occurs when a feature has never appeared in a particular class during training.

### Example

Suppose a Spam dataset contains 100 emails.

None of the emails contains the word "Lottery".

Then:

`P(Lottery = 1 | Spam) = 0/100 = 0`

When calculating the Naive Bayes score, all other feature probabilities get multiplied by this zero value.

Therefore:

`Score(Spam) = Prior * ... * 0 = 0`

Even if other features strongly indicate Spam, the entire class score becomes zero.

This is known as the **Zero Probability Problem**.

### Solution: Laplace Smoothing

Laplace Smoothing prevents zero probabilities by adding a small pseudo-count to feature counts.

It is also called Add-One Smoothing when alpha equals 1.

---

## 11. Laplace Smoothing (Additive Smoothing)

Laplace Smoothing adds a small constant to observed feature counts so that unseen features do not receive zero probability.

### General Formula

For a categorical feature:

`P(Xi = category | Y) = (Count(category, Y) + alpha) / (Count(Y) + alpha * K)`

Where:

- `Count(category, Y)` = Number of occurrences of the category in class Y.
- `Count(Y)` = Total observations in class Y.
- `K` = Number of possible categories.
- `alpha` = Smoothing parameter.

The denominator is also adjusted to ensure that the probabilities sum to 1.

### Types of Additive Smoothing

- `alpha = 0`: No smoothing.
- `alpha = 0.5`: Add-Half Smoothing.
- `alpha = 1`: Laplace or Add-One Smoothing.
- `alpha > 1`: Stronger smoothing.

**Note:** The exact smoothing formula depends on the Naive Bayes variant. For example, MultinomialNB uses vocabulary size in its denominator, while BernoulliNB uses two possible outcomes per feature.

---

## 12. Alpha Parameter and Bias-Variance Tradeoff

Alpha controls the strength of additive smoothing.

It determines how much pseudo-count is added to feature counts.

### Low Alpha

When alpha is small:

- Less smoothing is applied.
- Probability estimates remain closer to observed training frequencies.
- Estimates may be more sensitive to rare observations.
- Variance may be higher.

### High Alpha

When alpha is large:

- More smoothing is applied.
- Probability estimates become more uniform.
- Differences between observed feature frequencies become less influential.
- Bias may increase.

### Bias-Variance Tradeoff

| Alpha | Smoothing | General Effect |
|---|---|---|
| Very Low | Weak | Lower bias, potentially higher variance |
| Moderate | Moderate | Balance between bias and variance |
| Very High | Strong | Higher bias, potentially lower variance |

### Important Consideration

Increasing alpha does not guarantee better performance.

- Too little smoothing can make estimates sensitive to sparse observations.
- Too much smoothing can hide useful differences between features.

Therefore, alpha should be selected using validation or cross-validation.

---

## 13. Advantages of Naive Bayes

1. Simple and easy to understand.
2. Computationally efficient during training and prediction.
3. Works well with high-dimensional data.
4. Requires relatively little training data.
5. Supports multiclass classification.
6. Particularly effective as a baseline for text classification.
7. Naturally produces probabilistic predictions.

## 14. Limitations of Naive Bayes

1. Conditional independence assumption is often unrealistic.
2. Strongly correlated features can affect probability estimates.
3. Zero probability can occur without smoothing.
4. Probability estimates may be poorly calibrated.
5. Performance depends on whether the chosen distribution assumption matches the data.
6. Continuous and categorical features require appropriate Naive Bayes variants.

---

## 15. Important Terminology – Quick Revision

| Term | Meaning |
|---|---|
| Prior | Probability of a class before observing features |
| Likelihood | Probability of observed features given a class |
| Posterior | Probability of a class after observing features |
| Evidence | Overall probability of observed features |
| Conditional Independence | Features are assumed independent given the class |
| Underflow | Numerical problem caused by extremely small values |
| Log Probability | Converts probability multiplication into addition |
| Zero Probability | Feature receives zero estimated probability |
| Laplace Smoothing | Adds pseudo-counts to avoid zero probabilities |
| Alpha | Controls additive smoothing strength |

