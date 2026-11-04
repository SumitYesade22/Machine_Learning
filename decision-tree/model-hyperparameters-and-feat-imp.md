# Feature Importance in Decision Trees

## 1. What is Feature Importance?

Feature Importance measures how much each feature contributes to reducing impurity while building a Decision Tree.

In simple terms, it tells us how useful a feature was in making splits that improved the purity of the resulting child nodes.

It is calculated as the **normalized total reduction in impurity contributed by each feature**.

It is also commonly known as **Mean Decrease in Impurity (MDI)** or Gini Importance when Gini is used as the splitting criterion.

---

## 2. Impurity Reduction at a Node

Whenever a Decision Tree splits a node, it calculates how much impurity is reduced.

The impurity reduction contributed by a node is:

`Node Importance = (N_t / N) × [I_t - (N_left / N_t × I_left) - (N_right / N_t × I_right)]`

Where:

- `N` = Total number of samples in the training dataset.
- `N_t` = Number of samples reaching the current node.
- `N_left` = Number of samples in the left child.
- `N_right` = Number of samples in the right child.
- `I_t` = Impurity of the current node.
- `I_left` = Impurity of the left child.
- `I_right` = Impurity of the right child.

The formula consists of two parts:

1. **Weighted Impurity Reduction:** Measures the reduction in impurity after splitting the node.
2. **Node Weight (N_t / N):** Gives more importance to nodes containing a larger proportion of the training samples.

---

## 3. How is Feature Importance Calculated?

A feature may be used at multiple nodes in a Decision Tree.

Therefore, its importance is calculated by adding the impurity reductions contributed by all nodes where that feature was used.

### Formula

`FI_k = Σ Node Importance_j`

Where the summation includes all nodes that split using feature `k`.

To obtain normalized feature importance:

`Normalized FI_k = FI_k / Σ FI_j`

Where the denominator represents the total importance contributed by all features.

The normalized importance values sum to 1 (or 100%).

---

## 4. Example: Feature Used at Multiple Nodes

Consider a Decision Tree with two features:

- SL
- SW

Suppose the tree uses:

- SL at the root node.
- SW at the left child node.
- SL again at the right child node.

Assume the impurity reduction contributions are:

| Feature | Node | Importance Contribution |
|---|---|---:|
| SL | Root Node | 0.30 |
| SW | Left Child | 0.20 |
| SL | Right Child | 0.50 |

### Step 1: Calculate Total Importance of Each Feature

For SL:

`FI_SL = 0.30 + 0.50 = 0.80`

For SW:

`FI_SW = 0.20`

### Step 2: Calculate Total Importance

`Total Importance = 0.80 + 0.20 = 1.00`

### Step 3: Normalize Feature Importance

For SL:

`Normalized FI_SL = 0.80 / 1.00 = 0.80`

For SW:

`Normalized FI_SW = 0.20 / 1.00 = 0.20`

### Final Result

| Feature | Feature Importance | Percentage |
|---|---:|---:|
| SL | 0.80 | 80% |
| SW | 0.20 | 20% |

Therefore, SL contributed 80% of the total impurity reduction, while SW contributed 20%.

---

## 5. Interpretation of Feature Importance

Suppose the calculated feature importance is:

| Feature | Importance |
|---|---:|
| Age | 0.45 |
| Income | 0.35 |
| Credit Score | 0.20 |

Interpretation:

- Age contributed 45% of the total impurity reduction.
- Income contributed 35%.
- Credit Score contributed 20%.

A higher feature importance indicates that the feature contributed more to reducing impurity across the tree.

**Important:** Feature importance does not indicate whether a feature has a positive or negative relationship with the target variable. It only measures its contribution to impurity reduction.

---

## 6. Limitations of Impurity-Based Feature Importance

1. **Bias Toward High-Cardinality Features:** Features with many unique values or possible split points can receive higher importance under certain splitting criteria.

2. **Importance Distribution Among Correlated Features:** When features are highly correlated, the model may distribute importance between them or favor one over another.

3. **Does Not Show Direction:** Feature importance does not tell whether a feature increases or decreases the predicted target value.

4. **Not a Causal Measure:** High feature importance does not mean that the feature causes changes in the target variable.

5. **Model Dependent:** Importance values depend on the fitted tree and can change with different training data or hyperparameters.

---

## 7. Key Takeaways

- Feature Importance measures each feature's contribution to total impurity reduction.
- A feature's importance is the sum of its contributions across all nodes where it is used for splitting.
- Importance values are normalized so that their sum equals 1.
- Higher importance indicates a greater contribution to impurity reduction, not necessarily a stronger causal relationship with the target.
- In Scikit-learn, feature importance is available through `feature_importances_`.

## Techniques to Reduce Overfitting in Decision Trees

1. **Pre-Pruning:** Stop tree growth early by applying constraints using hyperparameters such as `max_depth`, `min_samples_split`, `min_samples_leaf`, and `min_impurity_decrease`.

2. **Post-Pruning:** First grow the tree and then remove unnecessary branches using Cost Complexity Pruning (`ccp_alpha`).

3. **Cross-Validation:** Use cross-validation to select suitable hyperparameter values and improve generalization.

4. **Feature Selection:** Remove irrelevant features, identifier columns, and data leakage features to avoid learning meaningless patterns.

# Pre-Pruning in Decision Trees

## 1. What is Pre-Pruning?

Pre-Pruning is a technique used to prevent overfitting by restricting the growth of a Decision Tree during training.

Instead of allowing the tree to grow until no further valid splits are possible, we apply certain conditions to stop unnecessary splitting early.

These conditions are controlled using hyperparameters.

## 2. Pre-Pruning Hyperparameters

### 2.1 max_depth

**Definition:** Controls the maximum depth (number of levels) of the Decision Tree.

- `max_depth=None`: No explicit depth restriction.
- Smaller values produce simpler trees.
- Larger values allow more complex trees.

**Example:**

```python
DecisionTreeClassifier(max_depth=3)
```

The tree cannot grow beyond a maximum depth of 3.

**Why use it?**

To prevent the tree from creating too many levels and learning overly specific patterns from training data.

---

### 2.2 min_samples_split

**Definition:** Specifies the minimum number of samples required at a node to make it eligible for splitting.

- `min_samples_split=2`: A node needs at least 2 samples to be considered for splitting.
- Higher values restrict splitting.
- If a node has fewer samples than the specified value, it cannot be split.

**Example:**

```python
DecisionTreeClassifier(min_samples_split=10)
```

A node must contain at least 10 samples to be eligible for splitting.

**Why use it?**

To prevent the tree from splitting nodes containing very few samples.

---

### 2.3 min_samples_leaf

**Definition:** Specifies the minimum number of samples that must be present in every leaf node.

- `min_samples_leaf=1`: A leaf can contain at least 1 sample.
- Higher values require more samples in each leaf.
- A split is allowed only if both resulting child nodes satisfy this minimum.

**Example:**

```python
DecisionTreeClassifier(min_samples_leaf=5)
```

Every leaf node must contain at least 5 training samples.

**Why use it?**

To prevent the creation of very small leaf nodes and make predictions less sensitive to individual observations.

---

### 2.4 max_leaf_nodes

**Definition:** Restricts the maximum number of leaf nodes in the Decision Tree.

- `max_leaf_nodes=None`: No explicit leaf count restriction.
- Smaller values produce simpler trees.
- Larger values allow more splits and more complex trees.

**Example:**

```python
DecisionTreeClassifier(max_leaf_nodes=10)
```

The tree can have a maximum of 10 leaf nodes.

**Why use it?**

To control overall tree complexity by limiting the number of terminal nodes.

---

### 2.5 min_impurity_decrease

**Definition:** A node is split only if the split produces an impurity reduction greater than or equal to the specified threshold.

The weighted impurity reduction is calculated as:

`Impurity Decrease = (N_t / N) × [I_t - (N_L / N_t × I_L) - (N_R / N_t × I_R)]`

Where:

- `N` = Total number of training samples.
- `N_t` = Number of samples reaching the current node.
- `N_L`, `N_R` = Samples in the left and right child nodes.
- `I_t` = Impurity of the current node.
- `I_L`, `I_R` = Impurity of the left and right child nodes.

**Example:**

```python
DecisionTreeClassifier(min_impurity_decrease=0.01)
```

A split is accepted only if its weighted impurity reduction is at least `0.01`.

**Why use it?**

To avoid splits that provide very little improvement in impurity.

---

### 2.6 max_features

**Definition:** Specifies the maximum number of features considered when searching for the best split at each node.

- `max_features=None`: All features are considered.
- Smaller values restrict the number of features considered at each split.
- It can introduce randomness into the tree-building process.

**Example:**

```python
DecisionTreeClassifier(max_features=3)
```

If the dataset contains 10 features, the algorithm considers at most 3 features when searching for the best split at each node.

**Why use it?**

- Reduces the number of features evaluated at each split.
- Can introduce randomness and reduce correlation between trees when used in ensemble methods.
- May reduce overfitting in some situations, but does not guarantee it.




## 3. Default Values in Scikit-learn

| Hyperparameter | Default Value |
|---|---|
| `max_depth` | None |
| `min_samples_split` | 2 |
| `min_samples_leaf` | 1 |
| `max_leaf_nodes` | None |
| `min_impurity_decrease` | 0.0 |

With these defaults, there are no explicit restrictions on maximum depth, minimum leaf size beyond one sample, or maximum leaf count.

However, the tree still stops when no valid split is available or no split provides positive impurity improvement.

## 4. Implementation

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5,
    max_leaf_nodes=20,
    min_impurity_decrease=0.01,
    random_state=42
)

model.fit(X_train, y_train)
```

## 5. Important Points

- Pre-Pruning controls tree growth during training.
- It can be bad if good nodes come after bad nodes , so before only the nodes are pruned.
- `max_depth` controls the maximum depth.
- `min_samples_split` controls whether a node is eligible for splitting.
- `min_samples_leaf` controls the minimum samples in each leaf.
- `max_leaf_nodes` controls the maximum number of leaf nodes.
- `min_impurity_decrease` controls the minimum impurity reduction required for a split.
- Hyperparameter values should be selected using cross-validation rather than arbitrary assumptions.
- Excessive pre-pruning can cause underfitting, while insufficient pre-pruning can lead to overfitting.