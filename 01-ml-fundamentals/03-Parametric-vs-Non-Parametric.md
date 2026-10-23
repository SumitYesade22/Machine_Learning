# Parametric vs Non-Parametric Models

 A parametric model has a fixed form of representation, with a fixed number of parameters when the model structure and number of features are fixed and non parametric model doesnt 
 
### Example: Linear Regression

Linear Regression models the relationship between input and output using:

\[
y = wx + b
\]

Where:
- `x` = input feature
- `y` = predicted output
- `w` = weight
- `b` = bias (intercept)

Suppose we use battery life to predict headphone prices.

After training, the model learns:

\[
\text{Price} = 50 \times \text{Battery Hours} + 100
\]

If battery life is 30 hours:

\[
\text{Price} = 50(30) + 100 = 1600
\]

If we train the same model on more data, the learned values of `w` and `b` may change. However, the model still has only two learned parameters.

**Key point:** The parameter values can change, but the number of parameters generally remains fixed when the model form and input features remain unchanged.

Other examples include Logistic Regression and Naive Bayes.

---

## 2. What Are Non-Parametric Models?

A **non-parametric model** does not restrict its learned representation to a fixed-size set of parameters. Its effective structure or stored information can grow with the training data.

### Example: K-Nearest Neighbors (KNN)

Suppose we want to predict the price of a new headphone.

KNN:
1. Stores the training examples.
2. Finds the `k` nearest examples to the new headphone.
3. Uses their target values to make a prediction.

Suppose the three nearest headphones have prices of ₹1,500, ₹2,000, and ₹2,500.

For KNN regression, their average price is:

\[
\frac{1500 + 2000 + 2500}{3} = ₹2000
\]

If the training dataset grows from 100 to 10,000 examples, KNN can store and search a larger set of examples. Its stored representation grows with the dataset.

**Key point:** KNN is non-parametric because its learned representation is not restricted to a fixed-size parameter vector. However, the number of neighbors `k` is a hyperparameter and can remain fixed.

Other examples include Decision Trees and many kernel-based methods.

---

## 3. Core Difference

| Feature | Parametric Models | Non-Parametric Models |
|---|---|---|
| Model representation | Specified functional form with a generally fixed number of learned parameters | Flexible learned structure or stored information |
| Effect of more training data | Parameter values may change; parameter count generally remains fixed | Learned structure or stored information may grow |
| Mathematical formulas | Yes | Formulas can also be used |
| Can predictions change? | Yes | Yes |
| Example | Linear Regression | KNN |

## 4. Important Clarifications

### A. Parameter values vs. model complexity

These are different concepts.

- **Parameter values:** The values learned during training, such as `w` and `b`.
- **Model complexity:** How complicated the learned representation is and what patterns the model can represent.

A parametric model's parameter values can change after retraining without increasing its parameter count.

A non-parametric model's structure or stored information can grow as more training examples become available.

### B. Non-parametric does not mean "no parameters"

A non-parametric model can have hyperparameters. For example, KNN uses `k` to specify the number of neighbors.

