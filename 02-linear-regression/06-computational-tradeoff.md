# Linear Regression: OLS vs Gradient Descent — Computational Trade-offs

## 1. Overview

Ordinary Least Squares (OLS) and Gradient Descent (GD) are two approaches for finding the coefficients of a Linear Regression model by minimizing the sum of squared errors.

- **OLS / Direct Solver:** Computes the least-squares solution directly using a numerical linear algebra solver.
- **Gradient Descent (GD):** Iteratively updates the coefficients in the direction that reduces the loss function.

Both can produce the same optimal coefficients when they optimize the same objective and the iterative method converges to the optimum.

> **Terminology:** OLS is the optimization objective (minimize squared errors). The Normal Equation, QR decomposition, and SVD are ways to solve the OLS problem. Gradient Descent is an iterative optimization algorithm that can also minimize the OLS objective.

## 2. Computational Trade-off: OLS vs Gradient Descent

| Factor / Scenario | OLS — Direct Solver (QR/SVD) | Gradient Descent (GD) | Which Is Better? |
|---|---|---|---|
| **Small dataset, few features** | Computes the solution directly with relatively little computational overhead. | Requires multiple iterations, even for a small problem. | **OLS** is usually simpler and more convenient. |
| **Large number of rows (n)** | Dense direct solvers must process the full design matrix and can become expensive in time and memory. | Full-batch GD processes all rows per iteration; SGD and mini-batch GD can process subsets. | **GD variants** can be more practical at very large scale. |
| **Large number of features (p)** | QR/SVD factorization can be computationally expensive, particularly when p is large. | Each full-batch iteration is roughly O(np); total cost depends on the number of iterations. | **Depends:** GD may be preferable when direct factorization becomes too expensive. |
| **Very large n and p** | Direct factorization may require substantial memory and computation. | Mini-batch GD can update coefficients without loading the entire dataset into memory at once. | **Mini-batch GD / SGD** is often a practical choice. |
| **Need an accurate least-squares solution** | A stable direct solver provides a high-accuracy solution up to numerical precision. | Accuracy depends on convergence, learning rate, stopping criteria, and numerical precision. | **Direct solver** is often preferable when the problem is manageable. |
| **Need a fast solution on a small or moderate problem** | Avoids choosing a learning rate or tuning convergence settings. | May need many updates before reaching a sufficiently accurate solution. | **OLS direct solver** is often preferable. |
| **Streaming or continuously arriving data** | A standard batch solve may require recomputing the solution when data changes. | SGD can update coefficients incrementally using new observations. | **SGD / online methods** are more suitable. |
| **Out-of-core learning (data does not fit in RAM)** | Standard dense direct solvers may require too much memory for the full matrix. Specialized out-of-core solvers are possible. | Mini-batch GD can read chunks of data and update coefficients incrementally. | **Mini-batch GD** is often more practical. |
| **Features are highly correlated** | SVD is useful for detecting near-linear dependencies and computing a numerically stable least-squares solution. | GD can also optimize the objective, but ill-conditioning may cause slow convergence. | **SVD** is often a strong choice when the problem size permits it. |
| **Rank-deficient design matrix** | SVD-based least-squares solvers can compute a minimum-norm solution. | A converged GD solution depends on initialization and optimization details; zero initialization for standard least-squares GD can yield a minimum-norm solution under suitable conditions. | **SVD** is a reliable direct option. |
| **No learning-rate tuning desired** | Does not require a learning rate. | Requires a suitable learning rate or adaptive optimization strategy. | **OLS direct solver**. |
| **Want to use stochastic or mini-batch updates** | Standard direct solvers are batch methods. | Supports SGD and mini-batch variants naturally. | **GD variants**. |
| **Need reproducible, deterministic results** | A fixed numerical solver is generally deterministic for the same inputs and environment. | Full-batch GD is generally deterministic with fixed initialization and settings; SGD may vary with shuffling and randomness. | **Both**, depending on implementation. |
| **Memory is limited** | Dense matrix factorization can require significant memory. | Mini-batch GD can limit the amount of data processed at a time, although coefficients and optimizer state still require memory. | **Mini-batch GD** can have an advantage. |
| **Want a simple baseline for Linear Regression** | Provides a strong least-squares baseline without iterative hyperparameter tuning. | Requires selecting and monitoring optimization settings. | **OLS direct solver** is usually a good starting point. |

