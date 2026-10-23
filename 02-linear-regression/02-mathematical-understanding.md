# Linear Regression — Mathematical Understanding

## 1. Simple Linear Regression

Simple Linear Regression is a supervised machine learning algorithm used to predict a continuous target variable using one input feature. It learns a linear relationship between the input and output.

The equation is:

$$
\hat{y}=mx+c
$$

Where:

- \(\hat{y}\): Predicted output.
- \(x\): Input feature.
- \(m\): Slope (coefficient).
- \(c\): Intercept.

### Slope

The slope represents the change in the predicted output for a one-unit increase in the input feature.

$$
m=\frac{\Delta \hat{y}}{\Delta x}
$$

For a differentiable function, the slope at a point is represented by its derivative:

$$
m=\frac{d\hat{y}}{dx}
$$

For example, if:

$$
\hat{y}=3x+5
$$

Then:

$$
\frac{d\hat{y}}{dx}=3
$$

Therefore, every one-unit increase in \(x\) increases the predicted output by 3 units.

### Intercept

The intercept represents the predicted output when \(x=0\).

$$
\hat{y}(0)=m(0)+c=c
$$

## 2. Multiple Linear Regression

Multiple Linear Regression is an extension of simple linear regression used when the target depends on multiple input features.

Its equation is:

$$
\hat{y}=b_0+b_1x_1+b_2x_2+\cdots+b_px_p
$$

Where:

- \(\hat{y}\): Predicted output.
- \(b_0\): Intercept.
- \(b_1,b_2,\ldots,b_p\): Feature coefficients.
- \(x_1,x_2,\ldots,x_p\): Input features.
- \(p\): Number of input features.

Each coefficient represents the expected change in the predicted output for a one-unit increase in its corresponding feature, while keeping the other features constant.

For example:

$$
\widehat{\mathrm{Price}}=b_0+b_1(\mathrm{Area})+b_2(\mathrm{Bedrooms})
$$

Here, \(b_1\) represents the change in predicted price per unit increase in area, holding the number of bedrooms constant.

### Matrix Representation

Multiple Linear Regression can be represented as:

$$
\hat{\mathbf{y}}=X\boldsymbol{\beta}
$$

Where \(X\) is the feature matrix, including a column of ones for the intercept, and \(\boldsymbol{\beta}\) is the vector of coefficients.

For example:

$$
\begin{bmatrix}
\hat{y}_1\\
\hat{y}_2\\
\hat{y}_3
\end{bmatrix}
=
\begin{bmatrix}
1 & x_{11}\\
1 & x_{21}\\
1 & x_{31}
\end{bmatrix}
\begin{bmatrix}
b_0\\
b_1
\end{bmatrix}
$$

## 3. Actual Values, Predictions, and Residuals

The actual value is the observed target value, whereas the predicted value is the output produced by the model.

The residual is the difference between the actual and predicted values:

$$
e_i=y_i-\hat{y}_i
$$

For simple linear regression:

$$
e_i=y_i-(mx_i+c)
$$

A positive residual means the model underpredicted the actual value. A negative residual means the model overpredicted it.

## 4. SSE — Sum of Squared Errors

SSE measures the total squared difference between actual and predicted values.

$$
SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

Substituting the linear regression equation:

$$
SSE(m,c)=\sum_{i=1}^{n}(y_i-mx_i-c)^2
$$

Squaring the residuals prevents positive and negative errors from canceling each other out and penalizes larger errors more heavily.

The objective is to find the parameters \(m\) and \(c\) that minimize SSE.

$$
(m^*,c^*)=\arg\min_{m,c}SSE(m,c)
$$

Here, \(m^*\) and \(c^*\) are the parameter values that produce the minimum SSE.

## 5. MSE — Mean Squared Error

MSE is the average of the squared residuals.

$$
MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

For simple linear regression:

$$
MSE(m,c)=\frac{1}{n}\sum_{i=1}^{n}(y_i-mx_i-c)^2
$$

Since \(MSE=SSE/n\), minimizing MSE produces the same optimal parameters as minimizing SSE for a fixed dataset.

## 6. RMSE — Root Mean Squared Error

RMSE is the square root of MSE.

$$
RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

It expresses the root mean squared prediction error in the same units as the target variable.

## 7. Finding the Best-Fit Line

The best-fit line is the line that minimizes the total squared residual error.

For simple linear regression, the objective is:

$$
J(m,c)=\sum_{i=1}^{n}(y_i-mx_i-c)^2
$$

To derive the optimal parameters, differentiate the objective with respect to \(m\) and \(c\), and set both derivatives to zero.

### Derivative with respect to the intercept

$$
\frac{\partial J}{\partial c}
=-2\sum_{i=1}^{n}(y_i-mx_i-c)
$$

Setting the derivative equal to zero:

$$
\sum_{i=1}^{n}(y_i-mx_i-c)=0
$$

Rearranging:

$$
\sum y_i-m\sum x_i-nc=0
$$

Therefore:

$$
c=\bar{y}-m\bar{x}
$$

### Derivative with respect to the slope

$$
\frac{\partial J}{\partial m}
=-2\sum_{i=1}^{n}x_i(y_i-mx_i-c)
$$

Setting the derivative equal to zero:

$$
\sum_{i=1}^{n}x_i(y_i-mx_i-c)=0
$$

Substituting \(c=\bar{y}-m\bar{x}\) and simplifying gives:

$$
\boxed{
m=
\frac{\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y})}
{\sum_{i=1}^{n}(x_i-\bar{x})^2}
}
$$

Once the slope is calculated, the intercept is:

$$
\boxed{c=\bar{y}-m\bar{x}}
$$

These are the closed-form OLS formulas for simple linear regression, provided the input feature has nonzero variance.

## 8. Important Mathematical Properties

### Convexity

The SSE objective in linear regression is a convex quadratic function of the model parameters.

For multiple linear regression:

$$
J(\boldsymbol{\beta})
=(\mathbf{y}-X\boldsymbol{\beta})^T
(\mathbf{y}-X\boldsymbol{\beta})
$$

Its Hessian is:

$$
\nabla^2 J=2X^TX
$$

Because \(X^TX\) is positive semidefinite, the objective is convex. Therefore, any global minimizer is a solution to the least-squares problem.

### Uniqueness

If the design matrix \(X\) has full column rank, the least-squares solution is unique. If its columns are linearly dependent, multiple coefficient vectors may produce the same minimum error.

### Key Takeaway

Linear Regression learns coefficients that define a linear relationship between input features and a continuous target. The best-fit parameters can be derived by minimizing squared residuals. SSE, MSE, and RMSE quantify prediction error, while the slope and intercept define the fitted line.
