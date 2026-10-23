# Simple Linear Regression

## 1. Introduction

**Simple Linear Regression (SLR)** is a supervised machine learning algorithm used to predict a continuous numerical value by establishing a linear relationship between one independent variable and one dependent variable.

It assumes that the relationship between the input feature and the target variable can be represented approximately by a straight line.

### Example

Suppose we want to predict a house's price based on its area.

- **Independent variable (\(X\)):** House area in square feet
- **Dependent variable (\(Y\)):** House price
- **Goal:** Learn the relationship between house area and price to predict the price of a new house.

As the area increases, the price may also increase. Simple Linear Regression attempts to capture this relationship using a straight line.

## 2. Mathematical Equation

The equation of Simple Linear Regression is:

\[
\hat{y} = b_0 + b_1x
\]

Where:

- \(x\): Input feature or independent variable.
- \(\hat{y}\): Predicted value of the target variable.
- \(b_0\): Intercept — the predicted value of \(Y\) when \(X=0\).
- \(b_1\): Slope or regression coefficient — the expected change in the predicted value of \(Y\) for a one-unit increase in \(X\).

**Example:**

Suppose the learned equation is:

\[
\hat{y} = 50 + 0.05x
\]

If the house area is 1,000 square feet:

\[
\hat{y} = 50 + 0.05(1000) = 100
\]

If the price is measured in lakhs of rupees, the predicted house price is ₹100 lakh.

## 3. Why Is It Called Simple Linear Regression?

- **Simple:** It uses only one independent variable.
- **Linear:** It models the relationship using a linear function.
- **Regression:** It predicts a continuous numerical target.

When two or more independent variables are used, the algorithm is called **Multiple Linear Regression**.

## 4. Objective

The objective is to find the best-fitting line that minimizes the prediction errors across the training dataset.

The difference between the actual value and predicted value is called the **residual**:

\[
e_i = y_i - \hat{y}_i
\]

A common approach is to minimize the Sum of Squared Errors (SSE):

\[
SSE = \sum_{i=1}^{n}(y_i-\hat{y}_i)^2
\]

This approach is known as **Ordinary Least Squares (OLS)**.

## 5. Key Characteristics

- It is a supervised learning algorithm.
- It predicts continuous numerical values.
- It uses one independent variable to predict one dependent variable.
- It is a parametric, model-based algorithm.
- Its standard form models a linear relationship between the input and the expected target value.
- Its parameters are the intercept and slope.

## 6. Applications

- Predicting house prices from area.
- Predicting sales from advertising expenditure.
- Estimating salary from years of experience.
- Predicting product demand using price.
- Studying the relationship between two numerical variables.

## 7. Limitations

- It may perform poorly when the relationship is strongly nonlinear.
- Outliers can significantly influence the fitted line.
- Important assumptions and residual patterns should be checked before interpreting the model.
- A relationship between two variables does not necessarily imply causation.
