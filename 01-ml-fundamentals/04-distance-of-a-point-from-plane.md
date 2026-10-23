# Distance of a Point from a Plane

## 1. Plane Equation

The general equation of a plane in 3D is:

\[
Ax+By+Cz+D=0
\]

Where:
- \(A,B,C\): Coefficients of \(x,y,z\).
- \(D\): Constant term.
- \((A,B,C)\): Normal vector of the plane, perpendicular to the plane.

Suppose a point is given as:

\[
P(x_1,y_1,z_1)
\]

We want to calculate the shortest perpendicular distance between this point and the plane.

## 2. Distance Formula

The perpendicular distance from a point \(P(x_1,y_1,z_1)\) to the plane \(Ax+By+Cz+D=0\) is:

\[
\boxed{
d=\frac{|Ax_1+By_1+Cz_1+D|}
{\sqrt{A^2+B^2+C^2}}
}
\]

### Understanding the Formula

**Numerator:**

\[
|Ax_1+By_1+Cz_1+D|
\]

Substitute the point's coordinates into the plane equation and take the absolute value of the result.

**Denominator:**

\[
\sqrt{A^2+B^2+C^2}
\]

This is the magnitude (length) of the plane's normal vector \((A,B,C)\).

**Why absolute value?**

Distance cannot be negative. The expression inside the absolute value may be positive or negative depending on which side of the plane the point lies.

## 3. Worked Example

Find the perpendicular distance of the point \(P(1,2,3)\) from the plane:

\[
2x+3y+4z+6=0
\]

**Step 1: Identify the coefficients and coordinates.**

\[
A=2,\quad B=3,\quad C=4,\quad D=6
\]

\[
(x_1,y_1,z_1)=(1,2,3)
\]

**Step 2: Substitute into the formula.**

\[
d=\frac{|2(1)+3(2)+4(3)+6|}
{\sqrt{2^2+3^2+4^2}}
\]

**Step 3: Simplify the numerator.**

\[
|2+6+12+6|=26
\]

**Step 4: Simplify the denominator.**

\[
\sqrt{4+9+16}=\sqrt{29}
\]

**Final answer:**

\[
\boxed{d=\frac{26}{\sqrt{29}}\approx4.83}
\]

The shortest perpendicular distance is approximately **4.83 units**.

## 4. Why Perpendicular Distance?

A point can be connected to many different points on a plane. However, the perpendicular distance is the shortest distance from the point to the plane.

The plane's normal vector \((A,B,C)\) points in a direction perpendicular to the plane.


## 5. Connection to Machine Learning

This concept is useful in **Support Vector Machines (SVMs)**.

An SVM learns a decision boundary represented by a hyperplane:

\[
\mathbf{w}^{T}\mathbf{x}+b=0
\]

The distance of a point from this hyperplane is:

\[
d=\frac{|\mathbf{w}^{T}\mathbf{x}+b|}
{\|\mathbf{w}\|}
\]

SVMs use distances from the decision boundary to construct a margin. In a linearly separable classification problem, the goal of a hard-margin SVM is to maximize the margin between the boundary and the closest training points.

## 6. Quick Revision

- **Plane:** \(Ax+By+Cz+D=0\)
- **Normal vector:** \((A,B,C)\)
- **Normal vector magnitude:** \(\sqrt{A^2+B^2+C^2}\)
- **Point-to-plane distance:** Absolute value of the substituted expression divided by the normal vector's magnitude.
- **Distance is always non-negative.**
- **ML application:** Measuring the distance of a data point from a linear decision boundary, especially in SVMs.