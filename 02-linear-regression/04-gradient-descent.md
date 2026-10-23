# Gradient Descent (GD)

## 1. What Is Gradient Descent?

Gradient Descent is an **iterative optimization algorithm** used to minimize a loss function by repeatedly updating model parameters in the direction opposite to the gradient.

In Machine Learning, a model makes predictions, and a loss function measures how far those predictions are from the actual values. Gradient Descent tries to find parameter values that minimize this loss.

### Why Do We Need Gradient Descent?

Consider Linear Regression:

$$
\hat{y} = mx + c
$$

Here, $m$ is the slope and $c$ is the intercept. Different values of $m$ and $c$ produce different predictions and, consequently, different loss values.

Gradient Descent helps us find parameter values that minimize the loss by iteratively updating $m$ and $c$.

## 2. How Does Gradient Descent Work?

1. **Initialize parameters:** Start with initial values for the model parameters, such as $m$ and $c$.
2. **Make predictions:** Calculate predictions using the current parameter values.
3. **Calculate loss:** Measure the prediction errors using a loss function, such as Mean Squared Error (MSE).
4. **Calculate gradients:** Find the partial derivatives of the loss with respect to each parameter.
5. **Update parameters:** Move in the direction opposite to the gradient.
6. **Repeat:** Continue until a stopping criterion is satisfied, such as a sufficiently small gradient or minimal improvement in loss.

The objective is to find parameter values that minimize the loss function.

## 3. Gradient Descent Formula

The general parameter update rule is:

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial J}{\partial w}
$$

Where:

- $w$ = model parameter or weight
- $J$ = loss function
- $\frac{\partial J}{\partial w}$ = gradient of the loss with respect to the parameter
- $\eta$ = learning rate

### What Does the Gradient Tell Us?

The gradient indicates the direction of the steepest local increase in the loss function.

- **Positive gradient:** The parameter is updated downward.
- **Negative gradient:** The parameter is updated upward.
- **Zero gradient:** The local slope is zero; this alone does not guarantee a minimum.

We subtract the gradient because our objective is to minimize the loss, not increase it.

## 4. Why Do We Need a Learning Rate?

The learning rate controls **how large a step the algorithm takes during each parameter update**.

$$
w_{\text{new}} = w_{\text{old}} - \eta \frac{\partial J}{\partial w}
$$

| Learning rate | Effect |
|---|---|
| Too small | Updates are small, so convergence can be slow. |
| Suitable | The algorithm can approach the minimum efficiently. |
| Too large | It may overshoot the minimum, oscillate, or diverge. |

A suitable learning rate depends on the loss function and the data. Even with a convex loss function, an unsuitable learning rate can prevent convergence.

## 5. Gradient Descent for Linear Regression

For simple Linear Regression, the prediction equation is:

$$
\hat{y}_i = mx_i + c
$$

The Mean Squared Error (MSE) loss function is:

$$
J(m,c) = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

The gradients are:

$$
\frac{\partial J}{\partial m}
=
-\frac{2}{n}\sum_{i=1}^{n}x_i(y_i-\hat{y}_i)
$$

$$
\frac{\partial J}{\partial c}
=
-\frac{2}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
$$

The parameter updates are:

$$
m_{\text{new}} = m_{\text{old}} - \eta\frac{\partial J}{\partial m}
$$

$$
c_{\text{new}} = c_{\text{old}} - \eta\frac{\partial J}{\partial c}
$$

The algorithm repeats these updates to reduce the MSE.

## 6. When Does Gradient Descent Stop?

Gradient Descent can stop when a predefined stopping criterion is satisfied, for example:

- The gradient magnitude becomes sufficiently small.
- The loss improvement between iterations becomes sufficiently small.
- The maximum number of iterations is reached.

A small parameter update alone does not always prove that the algorithm has reached a minimum.

## 7. Important Points

- Gradient Descent is an **optimization algorithm**, not a prediction model.
- It minimizes an objective or loss function by updating model parameters.
- The gradient indicates the direction of steepest local increase in loss.
- The learning rate controls the size of each update.
- Gradient Descent is iterative, unlike directly computing a closed-form least-squares solution.
- Different variants of Gradient Descent differ mainly in how much training data they use to calculate each update.

