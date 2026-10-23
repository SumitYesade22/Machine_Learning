# Linear Regression — OLS, Closed-Form Solution & Its Limitations

## 1. Two Ways to Find Model Parameters

Linear Regression learns coefficients that minimize prediction error. There are two broad approaches to finding these coefficients.

### A. Closed-Form Solution

A closed-form solution calculates the optimal coefficients directly using a mathematical formula.

- Example: Ordinary Least Squares (OLS) for Linear Regression.
- No repeated parameter updates are required.
- For standard least-squares regression, an exact solution can be calculated mathematically.

### B. Iterative Optimization

An iterative method starts with initial coefficients and repeatedly updates them to reduce the error.

- Example: Gradient Descent.
- It moves toward a minimum through repeated updates.
- It is useful when directly solving the optimization problem is computationally expensive.

**Remember:** Closed-form and iterative optimization are two approaches to solving an optimization problem. Gradient Descent is covered in detail in the next file.

---

## 2. What Is OLS?

**OLS stands for Ordinary Least Squares.**

It is a method used to estimate Linear Regression coefficients by minimizing the sum of squared residuals.

Consider a simple linear regression model:

\[
\hat{y}_i=mx_i+c
\]

Where:

- \(x_i\) = input value
- \(y_i\) = actual target value
- \(\hat{y}_i\) = predicted target value
- \(m\) = slope
- \(c\) = intercept

The residual for observation \(i\) is:

\[
e_i=y_i-\hat{y}_i
\]

OLS chooses the values of \(m\) and \(c\) that minimize the total squared residuals.

### Sum of Squared Errors (SSE)

\[
\boxed{SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
\]

Substituting the linear regression equation:

\[
\boxed{J(m,c)=\sum_{i=1}^{n}(y_i-mx_i-c)^2}
\]

Here, \(J(m,c)\) is the objective function we want to minimize.

### Why square the errors?

- Positive and negative errors do not cancel each other out.
- Larger errors receive a greater penalty.
- Squared errors produce a differentiable quadratic objective that can be minimized mathematically.

**Goal:** Find \(m\) and \(c\) such that \(J(m,c)\) is minimum.

---

## 3. Deriving the OLS Formula for Simple Linear Regression

We want to minimize:

\[
J(m,c)=\sum_{i=1}^{n}(y_i-mx_i-c)^2
\]

At the minimum, the first derivatives with respect to both parameters are zero.

### Step 1: Differentiate with respect to the intercept \(c\)

\[
\frac{\partial J}{\partial c}
=-2\sum_{i=1}^{n}(y_i-mx_i-c)
\]

Set the derivative equal to zero:

\[
\sum_{i=1}^{n}(y_i-mx_i-c)=0
\]

Expanding:

\[
\sum y_i-m\sum x_i-nc=0
\]

Rearranging:

\[
c=\frac{\sum y_i}{n}-m\frac{\sum x_i}{n}
\]

Since \(\bar{x}=\frac{\sum x_i}{n}\) and \(\bar{y}=\frac{\sum y_i}{n}\),

\[
\boxed{c=\bar{y}-m\bar{x}}
\]

### Step 2: Differentiate with respect to the slope \(m\)

\[
\frac{\partial J}{\partial m}
=-2\sum_{i=1}^{n}x_i(y_i-mx_i-c)
\]

Set it equal to zero:

\[
\sum x_i(y_i-mx_i-c)=0
\]

Substitute \(c=\bar{y}-m\bar{x}\), then simplify:

\[
\sum (x_i-\bar{x})(y_i-\bar{y})
=m\sum (x_i-\bar{x})^2
\]

Therefore:

\[
\boxed{
m=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
}
\]

Finally:

\[
\boxed{c=\bar{y}-m\bar{x}}
\]

These formulas give the least-squares slope and intercept when the input values are not all identical.

---

## 4. OLS for Multiple Linear Regression

When there are multiple input features, the model becomes:

\[
\hat{y}_i=b_0+b_1x_{i1}+\cdots+b_px_{ip}
\]

Instead of writing every observation separately, we can represent the entire dataset using matrices.

### Matrix notation

\[
\mathbf{y}=X\boldsymbol{\beta}+\boldsymbol{\varepsilon}
\]

Where:

- \(\mathbf{y}\) = vector of actual target values
- \(X\) = design matrix containing the input features
- \(\boldsymbol{\beta}\) = vector of model coefficients, including the intercept if a column of ones is included in \(X\)
- \(\boldsymbol{\varepsilon}\) = vector of errors

Predicted values:

\[
\hat{\mathbf{y}}=X\boldsymbol{\beta}
\]

Residuals:

\[
\mathbf{e}=\mathbf{y}-X\boldsymbol{\beta}
\]

The SSE objective becomes:

\[
J(\boldsymbol{\beta})
=(\mathbf{y}-X\boldsymbol{\beta})^T
(\mathbf{y}-X\boldsymbol{\beta})
\]

The superscript \(T\) means transpose.

### Deriving the matrix OLS solution

Expand the objective:

\[
J(\boldsymbol{\beta})
=\mathbf{y}^T\mathbf{y}
-2\boldsymbol{\beta}^TX^T\mathbf{y}
+\boldsymbol{\beta}^TX^TX\boldsymbol{\beta}
\]

Differentiate with respect to the coefficient vector:

\[
\nabla_{\boldsymbol{\beta}}J
=-2X^T\mathbf{y}+2X^TX\boldsymbol{\beta}
\]

Set the gradient equal to zero:

\[
X^TX\boldsymbol{\beta}=X^T\mathbf{y}
\]

This is called the **normal equation**.

If \(X^TX\) is invertible, multiply both sides by its inverse:

\[
\boxed{
\hat{\boldsymbol{\beta}}
=(X^TX)^{-1}X^T\mathbf{y}
}
\]

This is the closed-form OLS solution for multiple linear regression.

**Important:** The inverse formula requires \(X\) to have full column rank. If that condition fails, a least-squares solution still exists, but the coefficients may not be unique. A pseudoinverse or another least-squares solver can be used.

---

## 5. Why Does OLS Not Get Stuck in Local Minima?

The SSE objective for Linear Regression is a **convex quadratic function** of its coefficients.

\[
J(\boldsymbol{\beta})
=(\mathbf{y}-X\boldsymbol{\beta})^T
(\mathbf{y}-X\boldsymbol{\beta})
\]

Its Hessian (matrix of second derivatives) is:

\[
\boxed{\nabla^2 J=2X^TX}
\]

For any vector \(\mathbf{v}\):

\[
\mathbf{v}^TX^TX\mathbf{v}
=(X\mathbf{v})^T(X\mathbf{v})
=\|X\mathbf{v}\|^2\geq 0
\]

Therefore, \(X^TX\) is positive semidefinite, which means the Hessian is positive semidefinite and the SSE objective is convex.

### What does this mean practically?

- A convex function has no bad local minimum that is worse than its global minimum.
- Any global minimizer gives the lowest possible SSE.
- OLS solves the least-squares problem directly rather than searching through parameter updates, so it does not get trapped in a local minimum.

**Important distinction:** Batch Gradient Descent on this same convex objective also has no bad local minima. However, it can fail to converge if its learning rate or update behavior is unsuitable. The absence of bad local minima does not automatically guarantee convergence of every iterative algorithm.

---

## 6. When Is the OLS Solution Unique?

A least-squares minimizer always exists for a finite dataset, but the coefficient vector is not always unique.

### Case A: Full column rank — unique coefficients

If the columns of \(X\) are linearly independent, then:

\[
\operatorname{rank}(X)=p
\]

Here, \(p\) is the number of columns in \(X\), including the intercept column if present.

In this case, \(X^TX\) is positive definite and invertible, so:

\[
\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}
\]

has a unique solution.

### Case B: Linearly dependent columns — coefficients may not be unique

Suppose:

\[
x_2=2x_1
\]

Our model is:

\[
\hat{y}=b_0+b_1x_1+b_2x_2
\]

Substitute \(x_2=2x_1\):

\[
\hat{y}=b_0+(b_1+2b_2)x_1
\]

Predictions depend on the combination \(b_1+2b_2\), not on the individual values of \(b_1\) and \(b_2\).

Different coefficient combinations can therefore produce the same predictions and the same minimum SSE.

This means the coefficients may not be unique, even though the minimum error and fitted predictions are unique.

### Quick revision

| Property | Full column rank | Rank-deficient design matrix |
|---|---|---|
| Least-squares minimizer exists | Yes | Yes |
| Coefficients unique | Yes | Not necessarily |
| Minimum SSE unique | Yes | Yes |
| Bad local minima in SSE | No | No |

---

## 7. Limitations of the Closed-Form OLS Approach

The OLS formula is mathematically convenient, but calculating the solution directly can become expensive for large problems.

### 1. Computational cost

The formula involves operations on \(X^TX\), which is a \(p\times p\) matrix for \(p\) features.

Forming \(X^TX\) costs approximately \(O(np^2)\), and solving the resulting system typically costs \(O(p^3)\) using a dense factorization.

For very large feature counts, these costs can become significant.

### 2. Memory requirements

Storing and manipulating a dense \(p\times p\) matrix requires \(O(p^2)\) memory. This can be expensive when the number of features is very large.

### 3. Singular or poorly conditioned matrices

If features are linearly dependent, \(X^TX\) is singular and cannot be inverted.

If features are highly correlated, the problem may be poorly conditioned, meaning small changes in the data can lead to substantial changes in estimated coefficients.

In practice, QR decomposition or singular value decomposition (SVD) is often preferred over explicitly computing a matrix inverse. These methods can also handle rank-deficient cases more robustly.

### 4. Large or streaming datasets

If data is too large to process comfortably at once, repeatedly solving the full least-squares problem may be less practical than using iterative or incremental optimization methods.

**Important:** These are limitations of the direct computational approach, not proof that OLS itself is inaccurate or incapable of finding a least-squares solution.

---

## 8. Why Use Iterative Optimization Instead?

For large datasets or models, it may be more practical to update parameters iteratively rather than solve the full system directly.

Gradient-based methods can:

- Process data in batches rather than forming the full normal equation.
- Update coefficients incrementally.
- Work with very large datasets and many modern model architectures.
- Avoid explicitly forming or inverting \(X^TX\).

For standard Linear Regression, iterative optimization is an alternative to the closed-form least-squares solution; it is not required because OLS gets trapped in local minima.

For more complex models, a convenient closed-form solution may not exist, and iterative optimization becomes especially important.

**The next file will cover Gradient Descent itself:** gradients, learning rates, Batch Gradient Descent, Stochastic Gradient Descent, Mini-batch Gradient Descent, convergence, and learning-rate effects.

---

## 9. Final Revision Summary

| Concept | Remember this |
|---|---|
| OLS | Minimizes the sum of squared residuals. |
| SSE | Sum of squared prediction errors. |
| Closed-form solution | Calculates coefficients directly using a formula. |
| Normal equation | \(X^TX\hat{\boldsymbol{\beta}}=X^T\mathbf{y}\) |
| OLS coefficient formula | \(\hat{\boldsymbol{\beta}}=(X^TX)^{-1}X^T\mathbf{y}\), when \(X\) has full column rank. |
| Convexity | The SSE objective has no bad local minima. |
| Uniqueness | Full column rank guarantees unique least-squares coefficients. |
| Rank deficiency | Multiple coefficient vectors may give the same minimum SSE. |
| Direct-solution limitations | Computational cost, memory, and numerical conditioning. |
| Iterative optimization | Useful when direct computation is expensive or a closed-form solution is unavailable. |

**One-line takeaway:** OLS directly solves the least-squares problem for Linear Regression. Its SSE objective is convex, so there are no bad local minima; unique coefficients require full column rank. Iterative methods such as Gradient Descent are often used when direct computation is impractical or no convenient closed-form solution exists.