# 6. Training Strategy and Hyperparameter Tuning

## 6.1 Training is an iterative optimization process

XGBoost does not build all trees independently. It adds one tree per
boosting round, with each new tree responding to the current ensemble's
loss information.

A conceptual training cycle:

1.  Initialize raw predictions.
2.  Calculate gradients and Hessians.
3.  Construct a tree using candidate splits and gain.
4.  Calculate leaf weights.
5.  Scale the tree output using the learning rate.
6.  Update the ensemble predictions.
7.  Repeat for the next boosting round.

The number of boosting rounds is commonly controlled by `n_estimators`
in the scikit-learn interface or `num_boost_round` in the native
training interface.

## 6.2 Learning rate / eta

The learning rate scales the contribution of each new tree:

`new_prediction = old_prediction + eta × tree_output`

A smaller learning rate makes each individual update more conservative.

Consequences:

-   Smaller eta often requires more boosting rounds.
-   It can improve generalization by reducing the size of each step.
-   It increases training time if many more rounds are needed.
-   It is not a guarantee against overfitting.

A large learning rate can make the ensemble change rapidly and may
overshoot useful predictions.

Learning rate and number of estimators should be considered together. A
very small learning rate with too few trees may underfit.

## 6.3 Number of estimators / boosting rounds

This controls how many trees are added.

-   Too few trees: the ensemble may not have enough capacity to fit the
    underlying pattern.
-   Too many trees: training can continue improving while validation
    performance worsens.

The right number depends on learning rate, tree complexity, dataset
size, noise, and other regularization settings.

Do not select the number of trees solely from training performance.

## 6.4 Tree complexity controls

Important controls include:

  -----------------------------------------------------------------------
  Parameter               Main role               Typical reasoning when
                                                  increased
  ----------------------- ----------------------- -----------------------
  `max_depth`             Maximum tree depth      More complex partitions
                                                  and interactions

  `min_child_weight`      Minimum Hessian sum in  More conservative child
                          a child                 creation

  `gamma`                 Minimum split gain      Fewer weak splits
                          threshold               

  `max_leaves`            Maximum leaves, where   More possible terminal
                          supported               regions

  `lambda`                L2 leaf-weight          Smaller leaf outputs
                          regularization          

  `alpha`                 L1 leaf-weight          More shrinkage and
                          regularization          potentially zero leaf
                                                  outputs
  -----------------------------------------------------------------------

These parameters should not be tuned independently without considering
their interactions.

## 6.5 Row subsampling

`subsample` controls the fraction of training observations sampled for a
boosting iteration.

Using a fraction below 1 can introduce randomness and reduce the
correlation between trees. This can help generalization and reduce
overfitting.

However, sampling too aggressively can remove useful information and
increase variance in the training process.

The sampling behavior and exact details depend on the configured booster
and version.

## 6.6 Column subsampling

Column-sampling controls restrict the features considered by trees.

Common controls include:

-   `colsample_bytree`: fraction of features sampled for each tree.
-   `colsample_bylevel`: fraction sampled at each tree level.
-   `colsample_bynode`: fraction sampled at each split node.

Column subsampling can reduce dependence on a small set of dominant
features and lower computational cost. Excessively low values can
prevent useful features from being considered.

The three controls operate at different granularities and can interact.

## 6.7 Early stopping

Early stopping monitors a metric on a validation dataset while training.

If the monitored metric fails to improve for a specified number of
rounds, training can stop before reaching the maximum number of rounds.

Why it matters:

-   Training loss may keep decreasing after generalization stops
    improving.
-   Validation performance provides evidence about whether more trees
    are useful.
-   Early stopping can select a practical number of boosting rounds.

Use a validation set for early stopping. Do not use the final test set
repeatedly to make training decisions, because that leaks information
from the test set into model selection.

## 6.8 Objective functions and evaluation metrics

The objective determines what the model optimizes. The evaluation metric
measures performance for monitoring or comparison.

They are related but not identical.

Examples of objective families:

-   Squared-error regression objectives for continuous targets.
-   Logistic objectives for binary classification.
-   Multi-class classification objectives for multiple classes.
-   Other objectives for ranking, count data, survival analysis, and
    specialized tasks, depending on the XGBoost version.

Examples of evaluation metrics:

-   Regression: RMSE, MAE, and related metrics.
-   Classification: log loss, error rate, AUC, precision, recall, and F1
    when computed through an appropriate evaluation workflow.

A metric should reflect the business or application objective. Accuracy
alone can be misleading on imbalanced classification data.

## 6.9 Validation strategy

A reliable model-development workflow separates data into:

-   Training data: fit model parameters.
-   Validation data: compare configurations and make model-selection
    decisions.
-   Test data: final unbiased evaluation after decisions are complete.

For small datasets, cross-validation can provide a more stable estimate
than relying on one arbitrary validation split. For time-dependent data,
use a time-aware split rather than random shuffling.

Prevent leakage during preprocessing, feature selection, resampling, and
hyperparameter tuning. Any transformation that learns from data should
be fitted using training folds only.

## 6.10 Hyperparameter tuning: structured approach

Do not tune every parameter randomly at once. Use a staged strategy.

### Stage 1: Establish a baseline

-   Choose a suitable objective.
-   Choose a metric aligned with the problem.
-   Use a reasonable tree depth and learning rate.
-   Establish validation performance.

The baseline gives a reference point. Without it, tuning changes have no
interpretable comparison.

### Stage 2: Control tree complexity

Tune parameters such as:

-   `max_depth` or `max_leaves`
-   `min_child_weight`
-   `gamma`

These control how finely the model partitions feature space and how much
evidence is required for a split.

### Stage 3: Control update magnitude and regularization

Tune:

-   `learning_rate`
-   `n_estimators` / boosting rounds
-   `lambda`
-   `alpha`

Consider learning rate and number of trees jointly. Use validation and
early stopping where appropriate.

### Stage 4: Add sampling

Tune:

-   `subsample`
-   `colsample_bytree`
-   `colsample_bylevel`
-   `colsample_bynode`

These can improve generalization and reduce computational burden, but
overly aggressive sampling may underfit.

### Stage 5: Re-evaluate and inspect errors

Do not stop at one metric. Inspect:

-   Train versus validation performance.
-   Confusion matrix and class-specific metrics for classification.
-   Residuals and error distribution for regression.
-   Performance across meaningful segments.
-   Stability across folds or random seeds where relevant.

## 6.11 Diagnosing underfitting and overfitting

### Underfitting indicators

-   Training performance is poor.
-   Validation performance is also poor.
-   The model may be too constrained, too shallow, or trained for too
    few rounds.

Possible responses include increasing model capacity, allowing more
rounds, or reducing excessive regularization. Verify that the features
contain useful signal before increasing complexity.

### Overfitting indicators

-   Training performance is strong.
-   Validation performance is materially worse.
-   Validation performance deteriorates as more trees are added.

Possible responses include reducing tree complexity, increasing
regularization, using sampling, lowering learning rate with an
appropriate number of rounds, or applying early stopping.

Do not automatically increase regularization whenever a model overfits.
First inspect leakage, data quality, split strategy, and feature
construction.

## 6.12 Feature importance

XGBoost can report different feature-importance measures. Common
tree-based measures include:

-   **Weight:** how often a feature is used for splits.
-   **Gain:** the improvement attributed to splits using the feature.
-   **Cover:** the amount of data / Hessian coverage associated with
    those splits, depending on the definition used by the interface.

These measures answer different questions.

A feature used frequently is not necessarily the feature responsible for
the greatest loss reduction. Gain-based importance is often more
informative about split improvement, but it is still not a causal
explanation.

Importance can be biased by feature cardinality, correlated predictors,
and the model's split choices. Use permutation importance or SHAP-style
analysis when a more robust or observation-level explanation is needed,
while understanding their assumptions and computational cost.

## 6.13 XGBoost versus Random Forest

  -----------------------------------------------------------------------
  Aspect                  Random Forest           XGBoost
  ----------------------- ----------------------- -----------------------
  Ensemble strategy       Usually bagging         Boosting

  Tree training           Trees are largely       Trees are added
                          independent             sequentially

  Main goal               Reduce variance through Correct errors / reduce
                          aggregation             loss iteratively

  Tree relationship       Parallelizable          Dependent on previous
                                                  ensemble

  Typical control focus   Tree depth, feature     Learning rate, rounds,
                          sampling, number of     depth/leaves,
                          trees                   regularization,
                                                  sampling
  -----------------------------------------------------------------------

Neither model is universally superior. Performance depends on dataset
structure, preprocessing, tuning, and evaluation design.

## 6.14 XGBoost versus LightGBM and CatBoost

These are all gradient-boosted tree systems, but they differ in
implementation and design choices.

-   **XGBoost:** known for a regularized objective, multiple
    tree-construction methods, and mature training controls.
-   **LightGBM:** emphasizes efficient histogram-based learning and
    leaf-wise growth options; leaf-wise growth can be fast but needs
    complexity controls.
-   **CatBoost:** emphasizes handling categorical features and uses
    ordered boosting techniques designed to address prediction shift and
    target-statistics leakage concerns.

Do not choose a library based only on popularity or benchmark claims.
Consider categorical-feature handling, data size, latency, memory,
deployment constraints, and validation performance.

## 6.15 Practical model-development checklist

Before training:

-   Define the prediction target and business metric.
-   Confirm the train/validation/test split matches the data-generating
    process.
-   Check leakage and duplicate entities.
-   Understand class imbalance or target distribution.
-   Establish a baseline.

During training:

-   Use an objective appropriate to the target.
-   Monitor a validation metric.
-   Keep the test set out of tuning decisions.
-   Record parameters, random seeds, data version, and metrics.
-   Use early stopping when appropriate.

After training:

-   Compare train and validation performance.
-   Evaluate once on the held-out test set.
-   Inspect errors and segment-level performance.
-   Check feature importance with its limitations understood.
-   Save the full experiment configuration and preprocessing decisions.
-   Document limitations and assumptions.

## 6.16 Interview-level explanation

XGBoost is a gradient-boosted decision-tree method that adds trees
sequentially to improve the current ensemble. Its standard tree
objective combines the training loss with a regularization penalty on
the number of leaves and the magnitude of leaf weights. It uses a
second-order Taylor approximation of the loss, so each observation
contributes a gradient and a Hessian. These are aggregated within
candidate leaves to derive optimal leaf weights and calculate split
gain. Tree growth is controlled by parameters such as gamma, lambda,
alpha, min_child_weight, max_depth, and growth policy. Learning rate,
number of boosting rounds, row and column subsampling, validation, and
early stopping help control generalization.
