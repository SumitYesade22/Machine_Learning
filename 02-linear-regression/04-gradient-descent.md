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
\frac{\partial J}{\partial m} = -\frac{2}{n}\sum_{i=1}^{n}x_i(y_i-\hat{y}_i)
$$

$$
\frac{\partial J}{\partial c} = -\frac{2}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
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

There are 3 types of Gradient descent , they are as follow 

1. Batch Gradient Descent (BGD)

Definition

Batch Gradient Descent is an optimization algorithm that calculates the gradient using the entire training dataset and updates the model parameters once per iteration.

Working

Initialize the model parameters, such as slope $m$ and intercept $c$.

Calculate predictions for all training examples.

Calculate the loss using all training examples.

Calculate the gradients using the entire dataset.

Update the parameters using the learning rate.

Repeat until a stopping criterion is satisfied.

Mathematical Formulation

For simple Linear Regression, the prediction equation is:

$$
\hat{y}_i = mx_i + c
$$

The Mean Squared Error (MSE) loss function is:

$$
J(m,c) = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Substituting the prediction equation:

$$
J(m,c) = \frac{1}{n}\sum_{i=1}^{n}(y_i-mx_i-c)^2
$$

Here, $n$ represents the total number of training examples.

Gradient with Respect to the Slope

Differentiate the loss function with respect to $m$:

-\frac{2}{n}\sum_{i=1}^{n}x_i(y_i-\hat{y}_i)
$$

Gradient with Respect to the Intercept

Differentiate the loss function with respect to $c$:

-\frac{2}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)
$$

Parameter Updates

Update both parameters using their respective gradients:

m_{\text{old}}

\eta\frac{\partial J}{\partial m}
$$

c_{\text{old}}

\eta\frac{\partial J}{\partial c}
$$

Key point: Both gradients use all $n$ training examples. The parameters are updated only after the gradients have been calculated.

3. Advantages of Batch Gradient Descent

Stable gradient estimates: The gradient is calculated using the entire dataset, so individual examples do not cause the same level of update noise as in SGD.

Predictable optimization: For a fixed dataset and parameter values, the gradient is deterministic.

Suitable for convex loss functions: With an appropriate learning rate and suitable conditions, it can converge to a global minimum of the convex Linear Regression loss.

Straightforward implementation: The algorithm is relatively easy to understand and implement.

4. Disadvantages of Batch Gradient Descent

Computationally expensive for large datasets: Every iteration requires processing the entire training dataset.

Slower parameter updates: The model cannot update its parameters until the full-dataset gradient has been calculated.

Memory requirements: Processing the full dataset at once can require substantial memory, depending on the implementation.

Not naturally incremental: When new observations arrive, the algorithm does not automatically update its parameters using only those observations; additional implementation is required.

5. Important Points to Remember

Gradient: Indicates the direction of steepest local increase in the loss.

Learning rate: Controls the size of each parameter update.

Batch GD: Uses the entire dataset to calculate the gradient for each update.

SGD: Uses one training example to calculate the gradient for each update.

Convex loss: Has no suboptimal local minima; every local minimum is also a global minimum.

Convergence: Depends on factors such as the learning rate, loss function, and optimization conditions.

For standard Linear Regression with MSE, the loss is convex. Batch Gradient Descent can converge to a global minimum when the learning rate and other optimization conditions are appropriate.

# 2.Stochastic Gradient Descent (SGD)

## 1. What Is Stochastic Gradient Descent?

Stochastic Gradient Descent (SGD) is an iterative optimization algorithm that updates model parameters using the gradient calculated from **one randomly selected training example at a time**.

Unlike Batch Gradient Descent, which uses the entire training dataset to calculate each gradient, SGD calculates the gradient using a single training example for each update.

## 2. How Does SGD Work?

1. Initialize the model parameters, such as slope $m$ and intercept $c.
2. Shuffle the training dataset, if appropriate.
3. Select one training example.
4. Calculate the prediction and loss for that example.
5. Calculate the gradient using that single example.
6. Update the model parameters using the learning rate.
7. Repeat for the remaining training examples.
8. Process the dataset again for additional epochs until a stopping criterion is satisfied.

An **epoch** means one complete pass through the training dataset. In standard SGD, the parameters are updated after processing each training example.

## 3. Mathematical Formulation

For simple Linear Regression, the prediction equation is:

$$
\hat{y}_i = mx_i + c
$$

The squared error for a single training example is:

$$
J_i(m,c) = (y_i-\hat{y}_i)^2
$$

### Gradient with Respect to the Slope

$$
\frac{\partial J_i}{\partial m}
= -2x_i(y_i-\hat{y}_i)
$$

### Gradient with Respect to the Intercept

$$
\frac{\partial J_i}{\partial c}
= -2(y_i-\hat{y}_i)
$$

### Parameter Updates

$$
m_{\text{new}}
=
m_{\text{old}}
-
\eta\frac{\partial J_i}{\partial m}
$$

$$
c_{\text{new}}
=
c_{\text{old}}
-
\eta\frac{\partial J_i}{\partial c}
$$

Here, $i$ represents the selected training example and $\eta$ is the learning rate.

**Important:** These gradients are calculated using one training example, not the entire dataset. The equations above use the squared-error loss for a single example, without the MSE averaging factor $1/n$.

## 4. Advantages of SGD

1. **Faster individual updates:** The model can update its parameters without waiting for the entire dataset to be processed.
2. **Computationally efficient for large datasets:** Each update requires processing only one training example.
3. **Lower memory requirement per update:** The algorithm does not need to calculate the gradient from the entire dataset at once.
4. **Useful for online learning:** The model can be updated as new examples arrive, provided the implementation supports incremental updates.
5. **Can escape some saddle-point regions when non convex cost function is used:** Noisy updates can sometimes help optimization move through flat or saddle-point regions.

## 5. Disadvantages of SGD

1. **Noisy updates:** A single example may give a gradient that differs from the gradient calculated using the entire dataset.
2. **Fluctuating loss:** The loss may increase during some updates, even when the overall optimization is progressing.
3. **May fluctuate around the minimum:** With a constant learning rate, SGD may continue making noisy updates near the minimum instead of settling exactly at it.
4. **Learning-rate sensitivity:** A learning rate that is too large can cause unstable updates, while one that is too small can make learning slow.
5. **Less predictable individual updates:** The selected example affects each gradient, so the optimization path can be less smooth than Batch Gradient Descent.

## 6. Why Does SGD Fluctuate Around the Minimum?

Each training example contributes differently to the loss function. Consequently, the gradient calculated from one example may not match the gradient of the overall dataset.

As a result, SGD can move toward the minimum in one update and away from it in another.

Reducing the learning rate over time can decrease these fluctuations and help SGD converge under suitable conditions.

## 7. Batch GD vs SGD

| Feature | Batch Gradient Descent | Stochastic Gradient Descent |
|---|---|---|
| Data used per update | Entire dataset | One training example |
| Parameter updates | Once per full-dataset gradient calculation | Once per training example |
| Gradient behavior | Deterministic for fixed data and parameters | Depends on the selected example |
| Optimization path | Generally smoother | Generally noisier |
| Computation per update | Higher | Lower |
| Loss behavior | Generally smoother | Can fluctuate |
| Memory per update | Can be high | Generally lower |
| Online learning | Requires additional implementation | Naturally supports incremental updates |

## 8. Important Points

- SGD is an optimization algorithm, not a prediction model.
- It uses one training example to calculate each gradient update.
- An epoch is one complete pass through the training dataset.
- SGD can make faster individual updates, but it may require many updates to reach a good solution.
- A decreasing learning rate can reduce fluctuations near the minimum.
- For convex Linear Regression loss, SGD can converge toward a global minimum under appropriate learning-rate and sampling conditions.

**Key takeaway:** Batch GD calculates each gradient using the entire dataset, whereas SGD calculates each gradient using one training example at a time.

# 3. Mini-Batch Gradient Descent (MBGD)

## 1. What Is Mini-Batch Gradient Descent?

Mini-Batch Gradient Descent is an optimization algorithm that divides the training dataset into smaller groups called **mini-batches** and calculates the gradient using one mini-batch at a time.

It combines characteristics of Batch Gradient Descent and Stochastic Gradient Descent.

- **Batch GD:** Uses the entire dataset for each parameter update.
- **SGD:** Uses one training example for each parameter update.
- **Mini-Batch GD:** Uses a small group of training examples for each parameter update.

## 2. How Does Mini-Batch Gradient Descent Work?

1. Initialize the model parameters.
2. Shuffle the training dataset, if appropriate.
3. Divide the dataset into mini-batches of a fixed size.
4. Calculate predictions for the examples in the current mini-batch.
5. Calculate the average loss and gradients using that mini-batch.
6. Update the model parameters.
7. Repeat for the remaining mini-batches.
8. Start another epoch and repeat until a stopping criterion is satisfied.

An **epoch** is one complete pass through the training dataset.

Unlike Batch GD, Mini-Batch GD updates the parameters multiple times during one epoch.

## 3. Mathematical Formulation

For Linear Regression, the prediction equation is:

$$
\hat{y}_i = mx_i + c
$$

Suppose a mini-batch contains $B$ training examples. Its Mean Squared Error (MSE) is:

$$
J_B(m,c)
=
\frac{1}{B}\sum_{i \in \mathcal{B}}(y_i-\hat{y}_i)^2
$$

Here:

- $B$ = number of examples in the current mini-batch.
- $\mathcal{B}$ = set of training-example indices in that mini-batch.

### Gradient With Respect to the Slope

$$
\frac{\partial J_B}{\partial m}
=
-\frac{2}{B}\sum_{i \in \mathcal{B}}x_i(y_i-\hat{y}_i)
$$

### Gradient With Respect to the Intercept

$$
\frac{\partial J_B}{\partial c}
=
-\frac{2}{B}\sum_{i \in \mathcal{B}}(y_i-\hat{y}_i)
$$

### Parameter Updates

$$
m_{\text{new}}
=
m_{\text{old}}
-
\eta\frac{\partial J_B}{\partial m}
$$

$$
c_{\text{new}}
=
c_{\text{old}}
-
\eta\frac{\partial J_B}{\partial c}
$$

Here, $\eta$ is the learning rate.

**Important:** The gradients are calculated using only the examples in the current mini-batch, not the entire dataset.

## 4. Advantages of Mini-Batch Gradient Descent

1. **Efficient parameter updates:** The model does not need to process the entire dataset before every update.
2. **Lower memory requirements per update:** Only the current mini-batch needs to be processed at a time.
3. **Vectorization:** Mini-batches allow efficient matrix operations on modern CPUs and GPUs.
4. **Less noisy than SGD:** Averaging gradients over several examples generally reduces update noise compared with using a single example.
5. **Suitable for large datasets:** It provides a practical balance between computational efficiency and gradient stability.

## 5. Disadvantages of Mini-Batch Gradient Descent

1. **Noisy gradients:** The gradient is calculated from a subset of the dataset, so it can differ from the full-dataset gradient.
2. **Batch-size selection:** Very small batches can produce noisy updates, while very large batches require more memory and computation per update.
3. **Learning-rate sensitivity:** A poorly chosen learning rate can lead to slow convergence or unstable updates.
4. **Additional tuning:** Batch size, learning rate, and other training settings may need to be selected experimentally.
5. **Loss fluctuations:** The loss calculated on individual mini-batches can fluctuate during training.

## 6. Batch GD vs SGD vs Mini-Batch GD

| Feature | Batch GD | SGD | Mini-Batch GD |
|---|---|---|---|
| Examples per update | Entire dataset | One | Small group |
| Updates per epoch | Usually one | Approximately one per example | One per mini-batch |
| Gradient noise | Lowest for the fixed dataset | Generally highest | Generally intermediate |
| Computation per update | High | Low | Moderate |
| Vectorization | Yes | Limited per example | Highly efficient |
| Memory per update | Can be high | Low | Depends on batch size |
| Typical use | Smaller datasets | Online or incremental learning | Common choice for large-scale model training |

## 7. Important Points

- Mini-Batch GD calculates gradients using a subset of training examples.
- One mini-batch produces one parameter update.
- An epoch consists of processing all mini-batches in the training dataset.
- Smaller mini-batches generally produce noisier gradients; larger mini-batches generally provide more stable estimates of the full-dataset gradient.
- Mini-Batch GD is widely used to train neural networks because it supports efficient vectorized computation.
- For convex Linear Regression loss, Mini-Batch GD can converge toward a global minimum under suitable optimization conditions.

**Key takeaway:** Mini-Batch Gradient Descent balances the efficient updates of SGD with the more stable gradient estimates of Batch GD.