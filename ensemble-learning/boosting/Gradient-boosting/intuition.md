# Gradient Boosting

Gradient Boosting is an ensemble learning technique that builds multiple weak learners, typically Decision Trees, sequentially. Each new learner focuses on improving the predictions of the existing ensemble by reducing its errors. It uses gradient-based optimization to minimize a chosen loss function by adding new learners in the direction of the negative gradient.

## Additive Modeling

Additive modeling is an approach in which the final model prediction is constructed by adding the contributions of multiple individual models or functions.

Gradient Boosting follows this approach by starting with an initial prediction and progressively adding new Decision Trees to improve the existing prediction.

The general additive model is:

`F_m(x) = F_(m-1)(x) + η * h_m(x)`

Where:
- `F_m(x)` = Updated ensemble prediction.
- `F_(m-1)(x)` = Previous ensemble prediction.
- `h_m(x)` = Contribution of the newly added Decision Tree.
- `η` = Learning rate, controlling the contribution of the new tree.

The final prediction is the sum of the initial prediction and the contributions of all subsequently added trees.