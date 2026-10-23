# Equation of a Line, Plane, and Hyperplane

## 1. Equation of a Line (2D)

The slope-intercept form of a straight line is:

\[
y = wx + b
\]

Where:

- **\(x\)**: Independent variable; represents the input on the X-axis.
- **\(y\)**: Dependent variable; represents the output on the Y-axis.
- **\(w\)**: Slope of the line.
- **\(b\)**: Y-intercept; the point where the line crosses the Y-axis.

### A. Slope (\(w\))

Slope represents how much \(y\) changes when \(x\) changes by one unit.

\[
w = \frac{\Delta y}{\Delta x}
\]

Example:

\[
y = 2x + 3
\]

Here, \(w=2\). Therefore, when \(x\) increases by 1, \(y\) increases by 2.

- \(w>0\): The line rises from left to right.
- \(w<0\): The line falls from left to right.
- \(w=0\): The line is horizontal.

### B. Y-intercept (\(b\))

The Y-intercept is the point where the line crosses the Y-axis.

On the Y-axis, \(x=0\).

For example:

\[
y = 2x + 3
\]

Substituting \(x=0\):

\[
y = 2(0)+3=3
\]

Therefore, the Y-intercept is \((0,3)\).

**Effect of \(b\):** Keeping the slope fixed, changing \(b\) shifts the line vertically.

- \(b>0\): The line crosses the Y-axis above the origin.
- \(b<0\): The line crosses the Y-axis below the origin.
- \(b=0\): The line passes through the origin.

---

## 2. General Equation of a Line

A line in 2D can also be represented as:

\[
w_1x_1+w_2x_2+b=0
\]

Here, \(x_1\) and \(x_2\) are the two coordinates, and \(w_1,w_2\) are their coefficients.

For example:

\[
y=2x+3
\]

Rearranging:

\[
2x-y+3=0
\]

Both equations represent the same line.

This general form is useful because it can be extended to higher dimensions.

---

## 3. Equation of a Plane (3D)

In three-dimensional space, a plane is represented by:

\[
w_1x_1+w_2x_2+w_3x_3+b=0
\]

Where:

- \(x_1,x_2,x_3\): Coordinates along the X, Y, and Z axes.
- \(w_1,w_2,w_3\): Coefficients that determine the plane's orientation.
- \(b\): Constant or bias term that influences the plane's position.

A plane is a flat, two-dimensional surface in three-dimensional space, like an ideal flat tabletop.

### Understanding \(b\) in a Plane

In a 2D line, \(b\) is the Y-intercept because setting \(x=0\) gives \(y=b\).

However, in a 3D plane, **\(b\) is not directly the intercept of any single axis**. It is a constant term in the equation.

Consider:

\[
2x+3y+4z+6=0
\]

To find each axis-intercept, set the other two coordinates to zero.

**X-intercept:** Set \(y=0,z=0\).

\[
2x+6=0 \Rightarrow x=-3
\]

Point: \((-3,0,0)\)

**Y-intercept:** Set \(x=0,z=0\).

\[
3y+6=0 \Rightarrow y=-2
\]

Point: \((0,-2,0)\)

**Z-intercept:** Set \(x=0,y=0\).

\[
4z+6=0 \Rightarrow z=-1.5
\]

Point: \((0,0,-1.5)\)

Therefore, the plane intersects the three axes at different points.

**Important:** The values of the axis-intercepts depend on both \(b\) and the corresponding coefficients. The constant \(b\) alone is not an axis-intercept.

---

## 4. Equation of a Hyperplane

A hyperplane generalizes the equations of a line and a plane to higher-dimensional spaces.

\[
w_1x_1+w_2x_2+\cdots+w_nx_n+b=0
\]

Where:

- \(n\): Number of input features.
- \(x_1,x_2,\ldots,x_n\): Input features.
- \(w_1,w_2,\ldots,w_n\): Corresponding weights.
- \(b\): Bias or constant term.

### Relationship Between Dimensions and Boundaries

| Dimensions | Geometric boundary |
|---|---|
| 2D | Line |
| 3D | Plane |
| 4D or higher | Hyperplane |

A hyperplane is the higher-dimensional generalization of a line and a plane.

---

## 5. Connection to Machine Learning

### Linear Regression

Linear Regression predicts a numerical value using a weighted combination of input features.

\[
\hat y=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

- Weights determine how the prediction changes with each feature.
- Bias provides a constant offset.

### Logistic Regression

Logistic Regression calculates a linear score:

\[
z=w_1x_1+w_2x_2+\cdots+w_nx_n+b
\]

It converts this score into a probability using the sigmoid function in binary classification.

\[
P(y=1\mid x)=\frac{1}{1+e^{-z}}
\]

Its decision boundary occurs at \(z=0\), corresponding to a line in 2D, a plane in 3D, or a hyperplane in higher dimensions.

### Why Is This Relevant to Parametric Models?

For a fixed number of input features, Linear Regression and Logistic Regression have a fixed number of learned weights and a bias.

For \(n\) features:

\[
\text{Number of parameters}=n+1
\]

This fixed-size parameterization is one reason these models are classified as parametric.

---

## 6. Quick Summary

- **Slope (\(w\))**: Determines the slope of a line.
- **Y-intercept (\(b\))**: Where a 2D line crosses the Y-axis.
- **Plane**: A flat surface in 3D space.
- **Axis-intercepts of a plane**: Found by setting the other two coordinates to zero.
- **Bias (\(b\)) in a plane or hyperplane**: A constant term that influences the boundary's position; it is not itself a specific axis-intercept.
- **Hyperplane**: Generalization of a line and a plane to higher dimensions.
- **Machine Learning**: Linear models use weights and a bias to make predictions or define decision boundaries.