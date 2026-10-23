# Linear Regression — OLS, Closed-Form Solution & Its Limitations

## 1. Closed-Form Solution vs Iterative Optimization

Optimization means finding the parameter values that minimize a loss or cost function.

### Closed-Form Solution

A closed-form solution calculates the optimal parameters directly using a mathematical formula.

For Linear Regression, Ordinary Least Squares (OLS) gives the closed-form solution when the required inverse exists.

**Advantages:**
- Directly calculates the optimal coefficients.
- No learning rate or iterations required.
- Provides the exact least-squares solution, up to numerical precision.

### Iterative Optimization

Iterative optimization starts with initial parameter values and repeatedly updates them to reduce the loss function.

Examples include:
- Gradient Descent
- Stochastic Gradient Descent (SGD)
- Mini-batch Gradient Descent

These methods are useful when datasets or feature matrices are too large for a direct solution to be practical.

---

## 2. What Is OLS?

**OLS stands for Ordinary Least Squares.**

It finds the regression coefficients that minimize the sum of squared differences between actual and predicted values.

For observation \(i\):

$$
\text{Residual}_i = y_i - \hat{y}_i
$$

The Sum of Squared Errors (SSE) is:

$$
SSE = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

The Mean Squared Error (MSE) is:

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Since \(n\) is constant for a given dataset, minimizing SSE or MSE gives the same optimal coefficients.

OLS minimizes the squared residuals. It does not mean that every prediction will be exactly equal to its actual value.

---

## 3. Derivation of the OLS Solution

### A. Simple Linear Regression

The model is:

$$
\hat{y}_i = mx_i+c
$$

Where:
- \(m\) = slope
- \(c\) = intercept
- \(x_i\) = input value
- \(y_i\) = actual output
- \(\hat{y}_i\) = predicted output

The objective function is:

$$
J(m,c)=\sum_{i=1}^{n}(y_i-mx_i-c)^2
$$

We want to find the values of \(m\) and \(c\) that minimize this function.

**Step 1: Differentiate with respect to \(c\).**

$$
\frac{\partial J}{\partial c}
=-2\sum_{i=1}^{n}(y_i-mx_i-c)
$$

Set the derivative equal to zero:

$$
\sum_{i=1}^{n}(y_i-mx_i-c)=0
$$

Rearranging:

$$
\sum_{i=1}^{n}y_i
-m\sum_{i=1}^{n}x_i-nc=0
$$

Divide by \(n\):

$$
\bar{y}-m\bar{x}-c=0
$$

Therefore:

$$
\boxed{c=\bar{y}-m\bar{x}}
$$

**Step 2: Differentiate with respect to \(m\).**

$$
\frac{\partial J}{\partial m}
=-2\sum_{i=1}^{n}x_i(y_i-mx_i-c)
$$

Set the derivative equal to zero:

$$
\sum_{i=1}^{n}x_i(y_i-mx_i-c)=0
$$

Substitute \(c=\bar{y}-m\bar{x}\) and simplify. The resulting slope is:

$$
\boxed{
m=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
}
$$

Thus, the simple linear regression coefficients are:

$$
\boxed{m=\frac{\sum (x_i-\bar{x})(y_i-\bar{y})}
{\sum (x_i-\bar{x})^2}}
$$

$$
\boxed{c=\bar{y}-m\bar{x}}
$$

The slope formula requires the input values to have nonzero variance.

### B. Multiple Linear Regression — Matrix Form

For multiple features, the model can be written as:

$$
\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}
$$

Where:
- \(\mathbf{y}\) = vector of actual target values
- \(X\) = design matrix containing the features
- \(\boldsymbol{\beta}\) = vector of regression coefficients
- \(\boldsymbol{\varepsilon}\) = vector of errors

The predictions are:

$$
\hat{\mathbf{y}}=X\boldsymbol{\beta}
$$

The SSE objective function is:

$$
J(\boldsymbol{\beta})
=(\mathbf{y}-X\boldsymbol{\beta})^T
(\mathbf{y}-X\boldsymbol{\beta})
$$

**Step 1: Expand the objective function.**

$$
J(\boldsymbol{\beta})
=\mathbf{y}^T\mathbf{y}
-2\boldsymbol{\beta}^TX^T\mathbf{y}
+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}
$$

**Step 2: Calculate the gradient.**

$$
\nabla J(\boldsymbol{\beta})
=-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}
$$

**Step 3: Set the gradient equal to zero.**

$$
-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}=0
$$

Rearranging gives the **normal equation**:

$$
\boxed{X^TX\boldsymbol{\beta}=X^T\mathbf{y}}
$$

If \(X^TX\) is invertible, multiply both sides by its inverse:

$$
\boxed{
\hat{\boldsymbol{\beta}}
=(X^TX)^{-1}X^T\mathbf{y}
}
$$

This is the closed-form OLS solution when the design matrix has full column rank.

**Important:** In practice, numerical linear algebra methods such as QR decomposition or SVD are often preferred over explicitly calculating the matrix inverse.

---

## 4. Why Linear Regression Has No Bad Local Minima

Consider the SSE objective:

$$
J(\boldsymbol{\beta})
=(\mathbf{y}-X\boldsymbol{\beta})^T
(\mathbf{y}-X\boldsymbol{\beta})
$$

Its Hessian (matrix of second derivatives) is:

$$
\boxed{\nabla^2 J(\boldsymbol{\beta})=2X^TX}
$$

For any vector \(\mathbf{v}\):

$$
\begin{aligned}
\mathbf{v}^TX^TX\mathbf{v}
&=(X\mathbf{v})^T(X\mathbf{v})\\
&=\|X\mathbf{v}\|_2^2\\
&\geq 0
\end{aligned}
$$

Therefore, \(X^TX\) is **positive semidefinite**, and so is the Hessian.

This proves that the SSE objective function is **convex**.

### What does this mean?

- There are no bad local minima in standard Linear Regression SSE/MSE optimization.
- Every local minimum is also a global minimum.
- The global minimum is not necessarily unique.
- Gradient Descent can still have convergence problems if the learning rate or other settings are unsuitable.

**Important distinction:** Convexity guarantees the absence of bad local minima, but it does not guarantee a unique solution.

---

## 5. When Is the Solution Unique?

The uniqueness of the coefficient vector depends on the rank of the design matrix \(X\).

### Case 1: Full Column Rank

If all columns of \(X\) are linearly independent, then:

$$
\operatorname{rank}(X)=p
$$

Where \(p\) is the number of columns in \(X\), including the intercept column if one is used.

In this case, \(X^TX\) is invertible, and the coefficient vector is unique:

$$
\boxed{
\hat{\boldsymbol{\beta}}
=(X^TX)^{-1}X^T\mathbf{y}
}
$$

### Case 2: Rank-Deficient Matrix

If one or more columns are linearly dependent, then:

$$
\operatorname{rank}(X)<p
$$

Consequently, \(X^TX\) is singular and cannot be inverted.

However, **the least-squares problem still has at least one solution**. The coefficient vector may not be unique.

### Example

Suppose two features satisfy:

$$
x_2=2x_1
$$

The model is:

$$
\hat{y}=b_0+b_1x_1+b_2x_2
$$

Substituting \(x_2=2x_1\):

$$
\hat{y}=b_0+(b_1+2b_2)x_1
$$

Suppose the best-fitting model requires:

$$
b_1+2b_2=5
$$

All these coefficient pairs satisfy the same condition:

- \(b_1=5,\ b_2=0\)
- \(b_1=3,\ b_2=1\)
- \(b_1=1,\ b_2=2\)

They produce the same predictions because only the combination \(b_1+2b_2\) matters.

Therefore:
- The coefficients are not unique.
- The fitted predictions can still be unique.
- The minimum SSE remains the same.

### How Can We Obtain a Solution If the Inverse Fails?

The normal equation is:

$$
X^TX\boldsymbol{\beta}=X^T\mathbf{y}
$$

Even if \(X^TX\) is singular, this equation still has at least one solution. It simply may not have a unique solution.

Common approaches include:
- **Pseudoinverse:** Provides a least-squares solution, typically the minimum-norm solution.
- **QR decomposition:** Solves least-squares problems without explicitly forming the inverse.
- **SVD (Singular Value Decomposition):** Helps solve rank-deficient or ill-conditioned least-squares problems.

The failure of the inverse formula does not mean that a least-squares solution does not exist.

---

## 6. Limitations of the Closed-Form Solution

### 1. Computational Cost

Calculating the inverse of a \(p \times p\) matrix can require approximately \(O(p^3)\) operations using standard dense matrix methods.

For a very large number of features, this can become computationally expensive.

### 2. Memory Requirements

The design matrix \(X\) and intermediate matrices such as \(X^TX\) can consume substantial memory for large datasets or high-dimensional feature spaces.

### 3. Singular Matrices

If features are linearly dependent, \(X^TX\) is singular, so its inverse does not exist.

### 4. Numerical Instability

If features are highly correlated, \(X^TX\) can be ill-conditioned. Small numerical errors may lead to large changes in the estimated coefficients.

QR decomposition or SVD is generally safer than explicitly computing the inverse.

### 5. Large or Streaming Datasets

When data arrives continuously or the dataset is too large to process in one batch, iterative or online optimization methods can be more practical.

**Important:** Closed-form OLS is not always a bad choice. For small and moderate problems, it can be efficient and convenient. The best method depends on the dataset's size, structure, and numerical properties.

---

## 7. Why Use Iterative Optimization?

Iterative optimization updates the model parameters repeatedly to reduce the objective function.

For example, Gradient Descent updates the parameters using the gradient:

$$
\boldsymbol{\beta}_{t+1}
=\boldsymbol{\beta}_t-\eta\nabla J(\boldsymbol{\beta}_t)
$$

Where:
- \(\boldsymbol{\beta}_t\) = parameters at iteration \(t\)
- \(\eta\) = learning rate
- \(\nabla J(\boldsymbol{\beta}_t)\) = gradient at the current parameters

For Linear Regression with the SSE objective:

$$
\nabla J(\boldsymbol{\beta})
=2X^T(X\boldsymbol{\beta}-\mathbf{y})
$$

Therefore, the update becomes:

$$
\boldsymbol{\beta}_{t+1}
=\boldsymbol{\beta}_t
-2\eta X^T(X\boldsymbol{\beta}_t-\mathbf{y})
$$

The factor of \(2\) can be absorbed into the learning rate if the objective is defined with a factor of \(1/2\).

### Why Is It Useful?

- It can avoid explicitly computing a matrix inverse.
- It can be practical for very large datasets.
- Stochastic and mini-batch variants can update parameters using subsets of the data.
- Similar iterative methods also apply to more complex models that do not have a simple closed-form solution.

### Does Gradient Descent Get Stuck in Local Minima?

For standard Linear Regression with the SSE/MSE objective, the function is convex, so there are no bad local minima.

However, Gradient Descent may fail to converge if the learning rate is too high, or converge very slowly if the learning rate is too low or the problem is poorly conditioned.

The fact that Gradient Descent is iterative does not mean that the objective has local minima.

---

## 8. Quick Revision

| Concept | Key point |
|---|---|
| OLS | Minimizes the sum of squared residuals |
| Closed-form solution | Calculates the least-squares coefficients directly |
| Normal equation | \(X^TX\boldsymbol{\beta}=X^T\mathbf{y}\) |
| OLS formula | \(\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}\), when the inverse exists |
| Convexity | The SSE/MSE objective has no bad local minima |
| Full column rank | Gives a unique coefficient vector |
| Rank deficiency | May lead to multiple optimal coefficient vectors |
| Pseudoinverse | Can provide a solution when the ordinary inverse does not exist |
| QR/SVD | Useful numerical methods for least-squares problems |
| Gradient Descent | Iteratively updates parameters to reduce the loss |
| Learning rate | Controls the size of each parameter update |

**Remember:** A closed-form solution and Gradient Descent are two different ways to solve an optimization problem. For standard Linear Regression, both target the same least-squares minimum when used appropriately, but their computational behavior differs.