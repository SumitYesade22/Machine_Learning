# Decision Tree

A Decision Tree is a supervised, non-parametric machine learning algorithm used for both classification and regression tasks.

It works by recursively splitting the dataset into smaller subsets based on feature-based conditions to make predictions.

Each internal node represents a decision based on a feature, each branch represents the outcome of that decision, and each leaf node represents the final prediction.

**Key Characteristics:**
- Non-parametric model.
- White-box / interpretable model.
- Supports both classification and regression.
- Can capture non-linear relationships.
- Uses recursive binary splitting in CART.
- Requires little feature scaling or normalization.

## 1. Decision Tree Terminologies

| Terminology | Description |
|---|---|
| Root Node | The first node representing the entire dataset. |
| Decision Node (Internal Node) | A node where a feature-based condition is applied to split the data. |
| Leaf Node (Terminal Node) | The final node that provides the prediction. |
| Parent Node | A node that is divided into child nodes. |
| Child Node | A node created after splitting a parent node. |
| Branch | A connection representing the outcome of a decision. |
| Splitting | The process of dividing a node into smaller subsets based on a feature condition. |
| Impurity | A measure of how mixed the classes are within a node. |
| Pure Node | A node containing samples belonging to only one class. |
| Pruning | The process of removing branches to reduce tree complexity and prevent overfitting. |

## 2. Impurity Measures in Decision Trees

### 2.1 What is Impurity?

Impurity measures how mixed the target classes are within a Decision Tree node.

A node is considered pure when all samples belong to the same class. If a node contains samples from multiple classes, it is considered impure.

**Example:**

Consider a node containing 10 samples:

**Case 1: Pure Node**
- 10 Yes
- 0 No

All samples belong to the same class. Therefore, the node is completely pure.

**Case 2: Impure Node**
- 5 Yes
- 5 No

Both classes are equally represented. Therefore, the node has maximum impurity for binary classification.

**Case 3: Partially Impure Node**
- 8 Yes
- 2 No

One class dominates, but both classes are present. Therefore, the node is partially impure.

The objective of a Classification Decision Tree is to create child nodes with lower impurity by selecting suitable feature-based splits.

### 2.2 Types of Impurity Measures

Two commonly used impurity measures are:

1. Gini Impurity
2. Entropy

Both measure class impurity, but they use different mathematical formulas.

---

## 3. Gini Impurity

### 3.1 Definition

Gini Impurity is the probability of misclassifying a randomly selected sample when its class is randomly assigned according to the class distribution of the node.

It measures the impurity or class-mixing within a Decision Tree node.

### 3.2 Formula

Gini Impurity = 1 - Σ(p_i²)

Where:

- p_i = Proportion of samples belonging to class i.
- n = Total number of classes.

### 3.3 Interpretation

- Gini = 0: Completely pure node.
- Higher Gini: Greater class impurity.
- For binary classification, maximum Gini = 0.5 when both classes have equal proportions.

### 3.4 Numerical Examples

**Example 1: Pure Node**

Suppose a node contains:
- 10 Yes
- 0 No

p(Yes) = 10/10 = 1

p(No) = 0/10 = 0

Gini = 1 - (1² + 0²)

Gini = 0

The node is completely pure.

**Example 2: Equally Distributed Classes**

Suppose a node contains:
- 5 Yes
- 5 No

p(Yes) = 0.5

p(No) = 0.5

Gini = 1 - (0.5² + 0.5²)

Gini = 1 - (0.25 + 0.25)

Gini = 0.5

The node has maximum impurity for binary classification.

**Example 3: Unequally Distributed Classes**

Suppose a node contains:
- 8 Yes
- 2 No

p(Yes) = 0.8

p(No) = 0.2

Gini = 1 - (0.8² + 0.2²)

Gini = 1 - (0.64 + 0.04)

Gini = 0.32

The node is partially impure.

### 3.5 Summary

| Yes | No | Gini Impurity |
|---:|---:|---:|
| 10 | 0 | 0 |
| 8 | 2 | 0.32 |
| 5 | 5 | 0.50 |

A lower Gini value indicates a purer node.

---

## 4. Entropy

### 4.1 Definition

Entropy is a measure of uncertainty or randomness in the class distribution of a node.

It measures how mixed the classes are within a node.

### 4.2 Formula

Entropy = -Σ(p_i × log2(p_i))

Where:

- p_i = Proportion of samples belonging to class i.
- log2 = Logarithm with base 2.

Convention: If p_i = 0, its contribution to entropy is considered 0.

### 4.3 Interpretation

- Entropy = 0: Completely pure node.
- Higher entropy: Greater uncertainty.
- For binary classification, maximum entropy = 1 when both classes have equal proportions.

### 4.4 Numerical Examples

**Example 1: Pure Node**

Suppose a node contains:
- 10 Yes
- 0 No

Entropy = -[1 × log2(1) + 0 × log2(0)]

Entropy = 0

The node is completely pure.

**Example 2: Equally Distributed Classes**

Suppose a node contains:
- 5 Yes
- 5 No

Entropy = -[0.5 × log2(0.5) + 0.5 × log2(0.5)]

Entropy = -[0.5 × (-1) + 0.5 × (-1)]

Entropy = 1 bit

The node has maximum entropy for binary classification.

**Example 3: Unequally Distributed Classes**

Suppose a node contains:
- 8 Yes
- 2 No

Entropy = -[0.8 × log2(0.8) + 0.2 × log2(0.2)]

Entropy = -[0.8 × (-0.3219) + 0.2 × (-2.3219)]

Entropy = 0.7219 bits

The node is partially impure.

### 4.5 Summary

| Yes | No | Entropy |
|---:|---:|---:|
| 10 | 0 | 0 |
| 8 | 2 | 0.7219 |
| 5 | 5 | 1.0000 |

A lower entropy value indicates a purer node.

---

## 5. Gini Impurity vs Entropy

| Gini Impurity | Entropy |
|---|---|
| Measures class impurity. | Measures uncertainty in class distribution. |
| Formula uses squared probabilities. | Formula uses logarithmic probabilities. |
| Range for binary classification: 0 to 0.5. | Range for binary classification: 0 to 1 bit. |
| Computationally simpler. | Requires logarithmic calculations. |
| Lower impurity is preferred. | Lower impurity is preferred. |

Both measures help Decision Trees identify good splits.

---

## 6. Information Gain

### 6.1 Definition

Information Gain measures the reduction in entropy after splitting a parent node into child nodes.

It helps identify the feature and split that provide the maximum reduction in uncertainty.

### 6.2 Formula

Information Gain = Parent Entropy - Weighted Child Entropy

IG = H(parent) - Σ[(N_child / N_parent) × H(child)]

Where:

- H(parent) = Entropy of the parent node.
- H(child) = Entropy of a child node.
- N_child = Number of samples in the child node.
- N_parent = Number of samples in the parent node.

### 6.3 Why Weighted Entropy?

Different child nodes may contain different numbers of samples.

Therefore, we calculate a weighted average instead of taking a simple average.

A child containing 90 samples should contribute more to the final impurity calculation than a child containing 10 samples.

### 6.4 Numerical Example

Suppose a parent node contains:
- 6 Yes
- 4 No

Parent Entropy:

H(parent) = -[0.6 × log2(0.6) + 0.4 × log2(0.4)]

H(parent) = 0.971 bits

Now, suppose a split creates two child nodes.

**Left Child:**
- 4 Yes
- 0 No

Entropy(left) = 0

**Right Child:**
- 2 Yes
- 4 No

Entropy(right) = -[(2/6) × log2(2/6) + (4/6) × log2(4/6)]

Entropy(right) = 0.918 bits

Weighted Child Entropy:

= (4/10 × 0) + (6/10 × 0.918)

= 0.551 bits

Information Gain:

IG = 0.971 - 0.551

**Information Gain = 0.420 bits**

Interpretation:

The split reduced the uncertainty by approximately 0.420 bits.

A higher Information Gain indicates a better split.

---

## 7. Important Distinction

- Gini Impurity: Measures the impurity of a node.
- Entropy: Measures the uncertainty of a node.
- Information Gain: Measures how much entropy decreases after a split.

**Split selection:**

- Using Gini: Choose the split with the lowest weighted child Gini.
- Using Entropy: Choose the split with the highest Information Gain.

## 3. How Does a Decision Tree Select the Best Split?

A Decision Tree evaluates multiple candidate splits and selects the one that produces the **maximum reduction in impurity**.

The objective is to create child nodes that are more homogeneous (pure) than the parent node.

### Step 1: Generate Candidate Splits

The algorithm identifies possible splits using the available features.

- **Numerical features:** Candidate thresholds are generated between consecutive sorted values.
- **Categorical features:** Candidate splits are based on possible category partitions. The exact approach depends on the tree algorithm and implementation.

### Step 2: Calculate Child Node Impurity

For every candidate split:

1. Divide the parent dataset into left and right child nodes.
2. Calculate the impurity of each child using Gini or Entropy.
3. Calculate the weighted average of child impurities.

**Weighted Child Impurity:**

`Weighted Impurity = (N_left / N) × I_left + (N_right / N) × I_right`

Where:
- `N` = Total samples in the parent node
- `N_left`, `N_right` = Samples in the child nodes
- `I_left`, `I_right` = Impurity of the child nodes

### Step 3: Calculate Impurity Reduction

The algorithm measures how much impurity decreases after a split.

**For Gini:**

`Gini Reduction = Parent Gini - Weighted Child Gini`

**For Entropy:**

`Information Gain = Parent Entropy - Weighted Child Entropy`

### Step 4: Select the Best Split

The algorithm selects the candidate split that produces the **largest impurity reduction**.

This process is repeated recursively for the resulting child nodes until a stopping condition is reached.

**Important:** Decision Trees use a greedy approach. They select the best split at the current node rather than searching for the globally optimal tree.

## 4. CART Algorithm – Step-by-Step Working

**CART (Classification and Regression Trees)** is a tree-building algorithm that constructs binary decision trees by recursively splitting the dataset.

For classification, CART selects splits using Gini Impurity by default in Scikit-learn.

### Working Steps

1. **Start with the Root Node**
   - The entire training dataset is placed in the root node.
   - Calculate its impurity using the selected criterion (Gini or Entropy).

2. **Generate Candidate Splits**
   - Examine available features.
   - Generate possible split conditions for numerical and categorical features.

3. **Evaluate Candidate Splits**
   - Divide the dataset into two child nodes for each candidate split.
   - Calculate the weighted impurity of the child nodes.
   - Measure impurity reduction.

4. **Select the Best Split**
   - Select the split that produces the maximum impurity reduction.
   - Create two child nodes using that split.

5. **Repeat Recursively**
   - Apply the same process independently to each child node.
   - Each child node can become a parent node for further splits.

6. **Stop Splitting**
   - Stop when a stopping condition is reached, such as:
     - Maximum tree depth is reached (`max_depth`).
     - Minimum samples required to split a node are not available (`min_samples_split`).
     - Minimum samples required in a leaf are not available (`min_samples_leaf`).
     - No valid split provides sufficient impurity improvement (`min_impurity_decrease`).
     - The node is already pure or cannot be split further.

7. **Assign Class Predictions**
   - Each leaf node predicts the majority class among the training samples reaching that leaf.

### Important Characteristics

- CART creates **binary splits**: every decision node has at most two child nodes.
- It uses a **greedy approach**, selecting the best split at each node.
- Splitting continues recursively until stopping conditions are met.
- The final tree can be controlled using hyperparameters to reduce overfitting.

**Note:** CART supports both classification and regression. For classification, it predicts a class; for regression, it predicts a numerical value.